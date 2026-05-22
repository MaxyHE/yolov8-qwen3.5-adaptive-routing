"""
纯YOLO评估：mAP@0.5, mAP@0.5:0.95, per-class AP, FPS, 分类准确率
用法：
python eval_yolo.py \
  --data_dir /path/to/VOCdevkit/VOC2012 \
  --model_path best.pt \
  --yaml_path VOC2012.yaml \
  --max_images 500
"""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image
from ultralytics import YOLO

from eval_utils import (get_img_ids, load_voc_annotations, warmup_yolo,
                        update_confusion, print_confusion_summary, VOC_CLASSES)


def main(args):
    yolo = YOLO(args.model_path)
    val_txt = os.path.join(args.data_dir, "ImageSets/Main/val.txt")
    img_dir = os.path.join(args.data_dir, "JPEGImages")
    ann_dir = os.path.join(args.data_dir, "Annotations")
    img_ids = get_img_ids(val_txt)[:args.max_images]
    gt_map = load_voc_annotations(ann_dir, img_ids)

    # mAP
    print("[mAP] 运行 YOLO val...")
    metrics = yolo.val(data=args.yaml_path, verbose=False)
    map50 = round(float(metrics.box.map50), 4)
    map50_95 = round(float(metrics.box.map), 4)
    per_class_ap50 = {}
    if hasattr(metrics.box, 'ap50') and metrics.box.ap50 is not None:
        for i, cls in enumerate(VOC_CLASSES):
            if i < len(metrics.box.ap50):
                per_class_ap50[cls] = round(float(metrics.box.ap50[i]), 4)

    # warmup
    print("[warmup] 预热中...")
    warmup_yolo(yolo, img_dir, img_ids, n=10)

    # FPS + 分类准确率（同一次遍历）
    print("[eval] 评估中...")
    correct, total = 0, 0
    times = []
    from collections import defaultdict
    confusion = defaultdict(lambda: defaultdict(int))

    for i, img_id in enumerate(img_ids):
        img_path = os.path.join(img_dir, img_id + ".jpg")
        if not os.path.exists(img_path):
            continue
        gt_classes = set(gt_map.get(img_id, []))
        if not gt_classes:
            continue
        img = np.array(Image.open(img_path).convert("RGB"))
        t0 = time.time()
        results = yolo(img, conf=0.25, verbose=False)
        times.append(time.time() - t0)
        for box in results[0].boxes:
            pred = yolo.names[int(box.cls)]
            total += 1
            if pred in gt_classes:
                correct += 1
            update_confusion(confusion, pred, gt_classes)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(img_ids)}")

    fps = round(1.0 / np.mean(times), 1) if times else 0
    cls_acc = round(correct / total * 100, 2) if total else 0

    result = {
        "mAP50": map50,
        "mAP50_95": map50_95,
        "per_class_ap50": per_class_ap50,
        "fps": fps,
        "cls_acc": cls_acc,
        "total_boxes": total,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }

    print("\n===== YOLO 评估结果 =====")
    print(f"mAP@0.5:       {map50}")
    print(f"mAP@0.5:0.95:  {map50_95}")
    print(f"FPS:           {fps}")
    print(f"分类准确率:    {cls_acc}%  ({correct}/{total})")
    print("\nPer-class AP@0.5:")
    for cls, ap in sorted(per_class_ap50.items(), key=lambda x: -x[1]):
        print(f"  {cls:15s}: {ap:.4f}")
    print_confusion_summary(confusion)

    with open("results_yolo.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n结果已保存到 results_yolo.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/path/to/VOCdevkit/VOC2012")
    parser.add_argument("--model_path", default="best.pt")
    parser.add_argument("--yaml_path", default="VOC2012.yaml")
    parser.add_argument("--max_images", type=int, default=500)
    args = parser.parse_args()
    main(args)
