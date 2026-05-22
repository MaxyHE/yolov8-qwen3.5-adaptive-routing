"""
阈值扫描实验：sweep routing threshold，记录准确率/调用比例/延迟
VLM只跑一次，阈值扫描纯分析计算。

用法（原版VLM）：
python sweep_threshold.py \
  --data_dir /path/to/VOCdevkit/VOC2012 \
  --model_path best.pt \
  --vlm_path /path/to/Qwen3.5-0.8B \
  --max_images 500

用法（LoRA版本，推荐）：
python sweep_threshold.py \
  --data_dir /path/to/VOCdevkit/VOC2012 \
  --model_path best.pt \
  --vlm_path /path/to/Qwen3.5-0.8B \
  --lora_path /path/to/qwen_lora/best \
  --max_images 500
"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image
from ultralytics import YOLO

from eval_utils import (
    get_img_ids, load_voc_annotations, measure_vram,
    load_qwen, qwen_classify, warmup_yolo, warmup_vlm_qwen,
)

DEFAULT_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
                      0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def load_vlm(args):
    if args.lora_path:
        import torch
        from transformers import AutoProcessor
        from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
        from peft import PeftModel
        processor = AutoProcessor.from_pretrained(args.vlm_path, trust_remote_code=True)
        base = Qwen3_5ForConditionalGeneration.from_pretrained(
            args.vlm_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).cuda()
        model = PeftModel.from_pretrained(base, args.lora_path).merge_and_unload().to(torch.bfloat16).eval()
        return processor, model
    else:
        return load_qwen(args.vlm_path)


def main(args):
    thresholds = sorted([float(t) for t in args.thresholds.split(",")])
    max_thresh = max(thresholds)

    yolo = YOLO(args.model_path)
    val_txt = os.path.join(args.data_dir, "ImageSets/Main/val.txt")
    img_dir = os.path.join(args.data_dir, "JPEGImages")
    ann_dir = os.path.join(args.data_dir, "Annotations")
    img_ids = get_img_ids(val_txt)[:args.max_images]
    gt_map = load_voc_annotations(ann_dir, img_ids)

    vlm_tag = "Qwen3.5-0.8B-LoRA" if args.lora_path else "Qwen3.5-0.8B"
    print(f"[{vlm_tag}] 加载模型...")
    (processor, model), vram_gb = measure_vram(lambda: load_vlm(args))
    print(f"显存: {vram_gb} GB")

    print("[warmup] 预热中...")
    warmup_yolo(yolo, img_dir, img_ids, n=10)
    warmup_vlm_qwen(processor, model)

    # ── Phase 1: YOLO推理，缓存所有框 ────────────────────────
    print("\n[Phase 1] YOLO推理...")
    all_boxes = []
    yolo_times = []

    for i, img_id in enumerate(img_ids):
        img_path = os.path.join(img_dir, img_id + ".jpg")
        if not os.path.exists(img_path):
            continue
        gt_classes = set(gt_map.get(img_id, []))
        if not gt_classes:
            continue
        img = Image.open(img_path).convert("RGB")

        t0 = time.time()
        results = yolo(np.array(img), conf=0.25, verbose=False)
        yolo_times.append((time.time() - t0) * 1000)

        for box in results[0].boxes:
            conf = float(box.conf)
            yolo_cls = yolo.names[int(box.cls)]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img.crop((x1, y1, x2, y2))
            if crop.size[0] < 1 or crop.size[1] < 1:
                continue
            all_boxes.append({
                "gt_classes": gt_classes,
                "conf": conf,
                "yolo_cls": yolo_cls,
                "crop": crop,
                "vlm_pred": None,
                "vlm_ms": None,
            })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(img_ids)}")

    avg_yolo_ms = round(np.mean(yolo_times), 1)
    print(f"YOLO完成: {len(all_boxes)} 个框, 平均 {avg_yolo_ms} ms/图")

    # ── Phase 2: VLM对conf < max_thresh的框推理，缓存结果 ────
    vlm_candidates = [b for b in all_boxes if b["conf"] < max_thresh]
    print(f"\n[Phase 2] VLM推理: {len(vlm_candidates)} 个框 (conf < {max_thresh})...")

    for i, b in enumerate(vlm_candidates):
        t0 = time.time()
        b["vlm_pred"] = qwen_classify(b["crop"], processor, model)
        b["vlm_ms"] = (time.time() - t0) * 1000
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(vlm_candidates)}")

    vlm_ms_all = [b["vlm_ms"] for b in vlm_candidates]
    avg_vlm_ms = round(np.mean(vlm_ms_all), 1) if vlm_ms_all else 0
    p95_vlm_ms = round(float(np.percentile(vlm_ms_all, 95)), 1) if vlm_ms_all else 0
    print(f"VLM完成: avg={avg_vlm_ms}ms  p95={p95_vlm_ms}ms")

    # ── Phase 3: 扫描阈值，纯分析计算 ────────────────────────
    print("\n[Phase 3] 扫描阈值...\n")
    print(f"{'thresh':>8}  {'acc':>7}  {'call_rate':>10}  {'est_ms/img':>12}")
    print("-" * 46)

    avg_boxes_per_img = len(all_boxes) / len(yolo_times) if yolo_times else 1
    sweep = []

    for thresh in thresholds:
        correct, total, vlm_calls = 0, 0, 0
        for b in all_boxes:
            total += 1
            if b["conf"] < thresh and b["vlm_pred"] is not None:
                pred = b["vlm_pred"]
                vlm_calls += 1
            else:
                pred = b["yolo_cls"]
            if pred in b["gt_classes"]:
                correct += 1

        acc = round(correct / total * 100, 2) if total else 0
        call_rate = round(vlm_calls / total * 100, 1) if total else 0
        # 估算每张图总延迟 = YOLO时间 + 该图平均VLM调用次数 × 单次VLM时间
        est_total_ms = round(avg_yolo_ms + (call_rate / 100) * avg_boxes_per_img * avg_vlm_ms, 1)

        sweep.append({
            "threshold": thresh,
            "acc": acc,
            "vlm_call_rate": call_rate,
            "vlm_calls": vlm_calls,
            "total_boxes": total,
            "est_total_ms_per_image": est_total_ms,
        })
        print(f"{thresh:>8.2f}  {acc:>6.2f}%  {call_rate:>9.1f}%  {est_total_ms:>10.1f}ms")

    output = {
        "vlm": vlm_tag,
        "vram_gb": vram_gb,
        "num_images": len(yolo_times),
        "avg_yolo_ms_per_image": avg_yolo_ms,
        "avg_vlm_ms_per_crop": avg_vlm_ms,
        "p95_vlm_ms_per_crop": p95_vlm_ms,
        "avg_boxes_per_image": round(avg_boxes_per_img, 1),
        "sweep": sweep,
    }
    with open("results_threshold_sweep.json", "w") as f:
        json.dump(output, f, indent=2)
    out_name = f"results_threshold_sweep_{'lora' if args.lora_path else 'base'}.json"
    with open(out_name, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存到 {out_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/path/to/VOCdevkit/VOC2012")
    parser.add_argument("--model_path", default="best.pt")
    parser.add_argument("--vlm_path", default="/path/to/Qwen3.5-0.8B")
    parser.add_argument("--lora_path", default="", help="LoRA权重路径，留空则用原版VLM")
    parser.add_argument("--max_images", type=int, default=500)
    parser.add_argument("--thresholds",
                        default=",".join(str(t) for t in DEFAULT_THRESHOLDS))
    args = parser.parse_args()
    main(args)
