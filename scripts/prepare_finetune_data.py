"""
从VOC2012构造Qwen3.5-0.8B微调数据集。
对每张图跑YOLO，把所有检测框crop出来，配上GT标签，存成JSON。

用法：
python prepare_finetune_data.py \
  --data_dir /home/user/2026/yolo/VOCdevkit/VOC2012 \
  --model_path /home/user/2026/yolo/best.pt \
  --out_path /home/user/2026/yolo/finetune_data.json \
  --split train \
  --max_images 2000
"""
import argparse
import base64
import json
import os
from io import BytesIO

import numpy as np
from PIL import Image
from ultralytics import YOLO

from eval_utils import get_img_ids, load_voc_annotations, VOC_CLASSES, VOC_CLASSES_STR


def pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def main(args):
    yolo = YOLO(args.model_path)
    split_txt = os.path.join(args.data_dir, f"ImageSets/Main/{args.split}.txt")
    img_dir = os.path.join(args.data_dir, "JPEGImages")
    ann_dir = os.path.join(args.data_dir, "Annotations")

    img_ids = get_img_ids(split_txt)
    if args.max_images:
        img_ids = img_ids[:args.max_images]
    gt_map = load_voc_annotations(ann_dir, img_ids)

    samples = []
    for i, img_id in enumerate(img_ids):
        img_path = os.path.join(img_dir, img_id + ".jpg")
        if not os.path.exists(img_path):
            continue
        gt_classes = gt_map.get(img_id, [])
        if not gt_classes:
            continue

        img = Image.open(img_path).convert("RGB")
        results = yolo(np.array(img), conf=0.25, verbose=False)

        for box in results[0].boxes:
            yolo_cls = yolo.names[int(box.cls)]
            # 只保留YOLO预测在VOC类别内的框
            if yolo_cls not in VOC_CLASSES:
                continue
            # 只用高置信度框，标签直接用YOLO结果，避免噪声
            conf = float(box.conf)
            if conf < 0.5:
                continue
            label = yolo_cls

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img.crop((x1, y1, x2, y2))
            if crop.size[0] < 10 or crop.size[1] < 10:
                continue

            samples.append({
                "image_b64": pil_to_base64(crop),
                "label": label,
                "conf": round(conf, 3),
                "img_id": img_id,
            })

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(img_ids)}  samples so far: {len(samples)}")

    print(f"\n共构造 {len(samples)} 条样本")
    with open(args.out_path, "w") as f:
        json.dump(samples, f)
    print(f"已保存到 {args.out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/user/2026/yolo/VOCdevkit/VOC2012")
    parser.add_argument("--model_path", default="/home/user/2026/yolo/best.pt")
    parser.add_argument("--out_path", default="/home/user/2026/yolo/finetune_data.json")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--max_images", type=int, default=2000)
    args = parser.parse_args()
    main(args)
