import argparse
import random
import numpy as np
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on VOC2012")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="pretrained weights")
    parser.add_argument("--data", type=str, default="VOC2012.yaml", help="dataset config")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=str, default="runs/train", help="save directory")
    parser.add_argument("--name", type=str, default="yolov8s_voc2012", help="experiment name")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"[config] model={args.model}, data={args.data}, epochs={args.epochs}, "
          f"imgsz={args.imgsz}, batch={args.batch}, device={args.device}, seed={args.seed}")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        project=args.project,
        name=args.name,
    )

    metrics = results.results_dict
    print(f"[done] mAP@0.5={metrics.get('metrics/mAP50(B)', 'N/A'):.4f}  "
          f"mAP@0.5:0.95={metrics.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print(f"[done] best weights saved to: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
