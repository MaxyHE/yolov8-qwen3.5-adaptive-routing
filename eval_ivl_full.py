"""
YOLO + InternVL2 全量VLM评估（所有框都送VLM）
用法：
python eval_ivl_full.py \
  --data_dir /path/to/VOCdevkit/VOC2012 \
  --model_path best.pt \
  --vlm_path /path/to/internvl \
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

from eval_utils import (get_img_ids, load_voc_annotations, measure_vram,
                        load_internvl, internvl_classify, warmup_yolo,
                        warmup_vlm_internvl, update_confusion, print_confusion_summary)


def main(args):
    yolo = YOLO(args.model_path)
    val_txt = os.path.join(args.data_dir, "ImageSets/Main/val.txt")
    img_dir = os.path.join(args.data_dir, "JPEGImages")
    ann_dir = os.path.join(args.data_dir, "Annotations")
    img_ids = get_img_ids(val_txt)[:args.max_images]
    gt_map = load_voc_annotations(ann_dir, img_ids)

    print("[InternVL2] 加载模型...")
    (tokenizer, model), vram_gb = measure_vram(lambda: load_internvl(args.vlm_path))
    print(f"显存: {vram_gb} GB")

    print("[warmup] 预热中...")
    warmup_yolo(yolo, img_dir, img_ids, n=10)
    warmup_vlm_internvl(tokenizer, model)

    correct, total = 0, 0
    vlm_times = []
    confusion = defaultdict(lambda: defaultdict(int))

    for i, img_id in enumerate(img_ids):
        img_path = os.path.join(img_dir, img_id + ".jpg")
        if not os.path.exists(img_path):
            continue
        gt_classes = set(gt_map.get(img_id, []))
        if not gt_classes:
            continue
        img = Image.open(img_path).convert("RGB")
        results = yolo(np.array(img), conf=0.25, verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img.crop((x1, y1, x2, y2))
            total += 1
            if crop.size[0] > 0 and crop.size[1] > 0:
                t0 = time.time()
                pred = internvl_classify(crop, tokenizer, model)
                vlm_times.append((time.time() - t0) * 1000)
            else:
                pred = yolo.names[int(box.cls)]
            if pred in gt_classes:
                correct += 1
            update_confusion(confusion, pred, gt_classes)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(img_ids)}")

    acc = round(correct / total * 100, 2) if total else 0
    avg_ms = round(np.mean(vlm_times), 1) if vlm_times else 0

    result = {
        "model": "InternVL2-1B",
        "mode": "full",
        "acc": acc,
        "total_boxes": total,
        "avg_ms_per_crop": avg_ms,
        "vram_gb": vram_gb,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }

    print("\n===== InternVL2 全量VLM =====")
    print(f"分类准确率:    {acc}%  ({correct}/{total})")
    print(f"平均推理时间:  {avg_ms} ms/crop")
    print(f"显存:          {vram_gb} GB")
    print_confusion_summary(confusion)

    with open("results_ivl_full.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n结果已保存到 results_ivl_full.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/path/to/VOCdevkit/VOC2012")
    parser.add_argument("--model_path", default="best.pt")
    parser.add_argument("--vlm_path", default="/path/to/internvl")
    parser.add_argument("--max_images", type=int, default=500)
    args = parser.parse_args()
    main(args)
