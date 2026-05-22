"""
基于 IoU matching 的自动框级评估。

目标：
1. 用 class-agnostic IoU 匹配分离检测误差和分类误差
2. 支持 YOLO baseline / Qwen 自适应路由 / Qwen 全量路由
3. 输出整体 matched-box 分类准确率，以及 routed subset 的分类准确率

用法：
python eval_iou_matching.py \
  --data_dir /path/to/VOCdevkit/VOC2012 \
  --model_path best.pt \
  --mode adaptive \
  --vlm_path /path/to/Qwen3.5-0.8B \
  --max_images 500
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
from PIL import Image
from ultralytics import YOLO

from eval_utils import (
    CONF_THRESHOLD,
    get_img_ids,
    greedy_match_boxes,
    load_qwen,
    load_voc_box_annotations,
    measure_vram,
    qwen_classify,
    warmup_vlm_qwen,
    warmup_yolo,
)


def build_predictions(result, yolo, img_pil, mode, processor=None, vlm=None):
    predictions = []
    routed_count = 0
    routed_times = []

    for box in result.boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        yolo_label = yolo.names[int(box.cls)]
        conf = float(box.conf)
        final_label = yolo_label
        routed = False

        should_route = mode == "full" or (mode == "adaptive" and conf < CONF_THRESHOLD)
        if should_route:
            crop = img_pil.crop((int(x1), int(y1), int(x2), int(y2)))
            if crop.size[0] > 0 and crop.size[1] > 0:
                routed = True
                routed_count += 1
                t0 = time.time()
                final_label = qwen_classify(crop, processor, vlm)
                routed_times.append((time.time() - t0) * 1000)

        predictions.append({
            "bbox": [x1, y1, x2, y2],
            "conf": conf,
            "yolo_label": yolo_label,
            "label": final_label,
            "routed": routed,
        })

    return predictions, routed_count, routed_times


def summarize_counter(counter):
    return {k: dict(v) for k, v in counter.items()}


def main(args):
    yolo = YOLO(args.model_path)
    val_txt = os.path.join(args.data_dir, "ImageSets/Main/val.txt")
    img_dir = os.path.join(args.data_dir, "JPEGImages")
    ann_dir = os.path.join(args.data_dir, "Annotations")
    img_ids = get_img_ids(val_txt)[:args.max_images]
    gt_map = load_voc_box_annotations(ann_dir, img_ids)

    processor = vlm = None
    vram_gb = None
    if args.mode in ("adaptive", "full"):
        if not args.vlm_path:
            raise ValueError("--vlm_path is required for adaptive/full mode")
        print("[Qwen3.5] 加载模型...")
        (processor, vlm), vram_gb = measure_vram(lambda: load_qwen(args.vlm_path))
        print(f"显存: {vram_gb} GB")

    print("[warmup] 预热中...")
    warmup_yolo(yolo, img_dir, img_ids, n=10)
    if vlm is not None:
        warmup_vlm_qwen(processor, vlm)

    gt_total = 0
    pred_total = 0
    matched_total = 0
    matched_cls_correct = 0
    e2e_correct = 0
    routed_total = 0
    routed_matched = 0
    routed_cls_correct = 0
    low_conf_matched = 0
    low_conf_cls_correct = 0
    vlm_times = []
    pair_confusion = defaultdict(lambda: defaultdict(int))

    for i, img_id in enumerate(img_ids):
        img_path = os.path.join(img_dir, img_id + ".jpg")
        if not os.path.exists(img_path):
            continue

        gt_boxes = gt_map.get(img_id, [])
        gt_total += len(gt_boxes)

        img = Image.open(img_path).convert("RGB")
        yolo_result = yolo(np.array(img), conf=args.det_conf, verbose=False)[0]
        pred_boxes, routed_count, routed_times = build_predictions(
            yolo_result, yolo, img, args.mode, processor, vlm
        )

        pred_total += len(pred_boxes)
        routed_total += routed_count
        vlm_times.extend(routed_times)

        matches = greedy_match_boxes(pred_boxes, gt_boxes, iou_threshold=args.iou_threshold)
        matched_total += len(matches)

        for match in matches:
            pred = pred_boxes[match["pred_idx"]]
            gt = gt_boxes[match["gt_idx"]]
            gt_label = gt["label"]
            pred_label = pred["label"]

            pair_confusion[pred_label][gt_label] += 1
            if pred_label == gt_label:
                matched_cls_correct += 1
                e2e_correct += 1

            if pred["routed"]:
                routed_matched += 1
                if pred_label == gt_label:
                    routed_cls_correct += 1

            if pred["conf"] < CONF_THRESHOLD:
                low_conf_matched += 1
                if pred_label == gt_label:
                    low_conf_cls_correct += 1

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(img_ids)}")

    det_precision = round(matched_total / pred_total * 100, 2) if pred_total else 0
    det_recall = round(matched_total / gt_total * 100, 2) if gt_total else 0
    matched_cls_acc = round(matched_cls_correct / matched_total * 100, 2) if matched_total else 0
    e2e_precision = round(e2e_correct / pred_total * 100, 2) if pred_total else 0
    e2e_recall = round(e2e_correct / gt_total * 100, 2) if gt_total else 0
    routed_cls_acc = round(routed_cls_correct / routed_matched * 100, 2) if routed_matched else 0
    low_conf_cls_acc = round(low_conf_cls_correct / low_conf_matched * 100, 2) if low_conf_matched else 0
    avg_vlm_ms = round(float(np.mean(vlm_times)), 1) if vlm_times else 0

    result = {
        "mode": args.mode,
        "det_conf": args.det_conf,
        "route_conf": CONF_THRESHOLD,
        "iou_threshold": args.iou_threshold,
        "gt_total": gt_total,
        "pred_total": pred_total,
        "matched_total": matched_total,
        "det_precision": det_precision,
        "det_recall": det_recall,
        "matched_cls_acc": matched_cls_acc,
        "e2e_precision": e2e_precision,
        "e2e_recall": e2e_recall,
        "routed_total": routed_total,
        "routed_match_total": routed_matched,
        "routed_match_cls_acc": routed_cls_acc,
        "low_conf_match_total": low_conf_matched,
        "low_conf_match_cls_acc": low_conf_cls_acc,
        "avg_vlm_ms_per_crop": avg_vlm_ms,
        "vram_gb": vram_gb,
        "matched_pair_confusion": summarize_counter(pair_confusion),
    }

    print(f"\n===== IoU Matching 评估 ({args.mode}) =====")
    print(f"GT框数:                {gt_total}")
    print(f"预测框数:              {pred_total}")
    print(f"成功匹配数:            {matched_total}")
    print(f"检测Precision:         {det_precision}%")
    print(f"检测Recall:            {det_recall}%")
    print(f"Matched框分类准确率:   {matched_cls_acc}%")
    print(f"端到端Precision:       {e2e_precision}%")
    print(f"端到端Recall:          {e2e_recall}%")
    if args.mode in ("adaptive", "full"):
        print(f"路由触发数:            {routed_total}")
        print(f"路由匹配框分类准确率:  {routed_cls_acc}% ({routed_cls_correct}/{routed_matched})")
        print(f"低置信匹配框准确率:    {low_conf_cls_acc}% ({low_conf_cls_correct}/{low_conf_matched})")
        print(f"VLM平均耗时:           {avg_vlm_ms} ms/crop")
        print(f"显存:                  {vram_gb} GB")

    out_path = args.out_path or f"results_iou_{args.mode}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/path/to/VOCdevkit/VOC2012")
    parser.add_argument("--model_path", default="best.pt")
    parser.add_argument("--mode", default="adaptive", choices=["yolo", "adaptive", "full"])
    parser.add_argument("--vlm_path", default=None)
    parser.add_argument("--max_images", type=int, default=500)
    parser.add_argument("--det_conf", type=float, default=0.25)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--out_path", default=None)
    args = parser.parse_args()
    main(args)
