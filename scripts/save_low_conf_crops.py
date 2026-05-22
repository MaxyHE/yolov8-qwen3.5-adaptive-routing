"""
生成低置信度框的crop，用于人工标注。
只保存conf < 0.5的框，同时记录元数据到JSON。

用法：
python save_low_conf_crops.py \
  --data_dir /home/user/2026/yolo/VOCdevkit/VOC2012 \
  --model_path /home/user/2026/yolo/best.pt \
  --out_dir /home/user/2026/yolo/low_conf_crops \
  --max_images 500
"""
import argparse
import json
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

    metadata = []

    for i, img_id in enumerate(img_ids):
        img_path = os.path.join(img_dir, img_id + ".jpg")
        if not os.path.exists(img_path):
            continue
        gt_classes = gt_map.get(img_id, [])
        if not gt_classes:
            continue

        img = Image.open(img_path).convert("RGB")
        results = yolo(np.array(img), conf=0.25, verbose=False)

        for j, box in enumerate(results[0].boxes):
            conf = float(box.conf)
            if conf >= 0.5:
                continue
            yolo_cls = yolo.names[int(box.cls)]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img.crop((x1, y1, x2, y2))
            if crop.size[0] < 10 or crop.size[1] < 10:
                continue

            filename = f"{img_id}_{j}_conf{conf:.2f}_{yolo_cls}.jpg"
            crop.save(os.path.join(args.out_dir, filename))
            metadata.append({
                "filename": filename,
                "img_id": img_id,
                "box_idx": j,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": round(conf, 3),
                "yolo_pred": yolo_cls,
                "gt_classes": gt_classes,
                "true_label": None,  # 待人工标注
            })

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(img_ids)}  crops so far: {len(metadata)}")

    meta_path = os.path.join(args.out_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n共生成 {len(metadata)} 个低置信度crop")
    print(f"元数据保存到 {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/user/2026/yolo/VOCdevkit/VOC2012")
    parser.add_argument("--model_path", default="/home/user/2026/yolo/best.pt")
    parser.add_argument("--out_dir", default="/home/user/2026/yolo/low_conf_crops")
    parser.add_argument("--max_images", type=int, default=500)
    args = parser.parse_args()
    main(args)
