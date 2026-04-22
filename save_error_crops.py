"""
保存YOLO判错的框的crop图片，按 pred__gt 分类存放。
只跑YOLO，不调VLM，纯可视化用途。
用法：
python save_error_crops.py \
  --data_dir /path/to/VOCdevkit/VOC2012 \
  --model_path best.pt \
  --out_dir /path/to/eval_results/error_crops \
  --max_images 500 \
  --max_per_pair 10
"""
import argparse
import os

import numpy as np
from PIL import Image
from ultralytics import YOLO

from eval_utils import get_img_ids, load_voc_annotations


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    yolo = YOLO(args.model_path)
    val_txt = os.path.join(args.data_dir, "ImageSets/Main/val.txt")
    img_dir = os.path.join(args.data_dir, "JPEGImages")
    ann_dir = os.path.join(args.data_dir, "Annotations")
    img_ids = get_img_ids(val_txt)[:args.max_images]
    gt_map = load_voc_annotations(ann_dir, img_ids)

    saved = {}  # (pred, gt) -> count

    for img_id in img_ids:
        img_path = os.path.join(img_dir, img_id + ".jpg")
        if not os.path.exists(img_path):
            continue
        gt_classes = set(gt_map.get(img_id, []))
        if not gt_classes:
            continue
        img = Image.open(img_path).convert("RGB")
        results = yolo(np.array(img), conf=0.25, verbose=False)
        for box in results[0].boxes:
            pred = yolo.names[int(box.cls)]
            if pred in gt_classes:
                continue
            # 判错了，找一个最近的GT类作为标注
            for gt in gt_classes:
                key = (pred, gt)
                cnt = saved.get(key, 0)
                if cnt >= args.max_per_pair:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img.crop((x1, y1, x2, y2))
                if crop.size[0] < 10 or crop.size[1] < 10:
                    continue
                pair_dir = os.path.join(args.out_dir, f"{pred}__{gt}")
                os.makedirs(pair_dir, exist_ok=True)
                conf = float(box.conf)
                crop.save(os.path.join(pair_dir, f"{img_id}_conf{conf:.2f}.jpg"))
                saved[key] = cnt + 1
                break  # 每个框只存一次

    total = sum(saved.values())
    print(f"共保存 {total} 张错误crop，目录: {args.out_dir}")
    print("Top误判对:")
    for (pred, gt), cnt in sorted(saved.items(), key=lambda x: -x[1])[:10]:
        print(f"  {pred} -> {gt}: {cnt}张")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/path/to/VOCdevkit/VOC2012")
    parser.add_argument("--model_path", default="best.pt")
    parser.add_argument("--out_dir", default="/path/to/eval_results/error_crops")
    parser.add_argument("--max_images", type=int, default=500)
    parser.add_argument("--max_per_pair", type=int, default=10)
    args = parser.parse_args()
    main(args)
