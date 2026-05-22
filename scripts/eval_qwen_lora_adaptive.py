"""
先merge LoRA权重，再跑自适应路由评估。
用法：
python eval_qwen_lora_adaptive.py \
  --data_dir /home/user/2026/yolo/VOCdevkit/VOC2012 \
  --model_path /home/user/2026/yolo/best.pt \
  --base_vlm_path /home/user/library/code/Qwen3.5-0.8B \
  --lora_path /home/user/2026/yolo/qwen_lora/best \
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
                        warmup_yolo, update_confusion, print_confusion_summary,
                        CONF_THRESHOLD, VOC_CLASSES)


def load_qwen_lora(base_vlm_path, lora_path):
    import torch
    from transformers import AutoProcessor
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    from peft import PeftModel

    processor = AutoProcessor.from_pretrained(base_vlm_path, trust_remote_code=True)
    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_vlm_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload().to(torch.bfloat16).eval()
    return processor, model


def qwen_lora_classify(crop_pil, processor, model):
    import torch
    from eval_utils import VOC_CLASSES_STR
    question = f"Choose the most likely category from: {VOC_CLASSES_STR}. Answer with only the category name."
    messages = [{"role": "user", "content": [
        {"type": "image", "image": crop_pil},
        {"type": "text", "text": question}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=[text], images=[crop_pil], return_tensors="pt").to("cuda")
    gen_inputs = {k: v for k, v in inputs.items()
                  if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")}
    with torch.no_grad():
        out = model.generate(**gen_inputs, max_new_tokens=16, do_sample=False)
    result = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip().lower()
    for cls in VOC_CLASSES:
        if cls in result:
            return cls
    return result


def main(args):
    yolo = YOLO(args.model_path)
    val_txt = os.path.join(args.data_dir, "ImageSets/Main/val.txt")
    img_dir = os.path.join(args.data_dir, "JPEGImages")
    ann_dir = os.path.join(args.data_dir, "Annotations")
    img_ids = get_img_ids(val_txt)[:args.max_images]
    gt_map = load_voc_annotations(ann_dir, img_ids)

    print("[LoRA VLM] 加载并merge模型...")
    (processor, model), vram_gb = measure_vram(lambda: load_qwen_lora(args.base_vlm_path, args.lora_path))
    print(f"显存: {vram_gb} GB")

    print("[warmup] 预热中...")
    warmup_yolo(yolo, img_dir, img_ids, n=10)
    # warmup VLM
    dummy = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    qwen_lora_classify(dummy, processor, model)

    correct, total, vlm_calls = 0, 0, 0
    vlm_times = []
    img_times = []
    confusion = defaultdict(lambda: defaultdict(int))

    for i, img_id in enumerate(img_ids):
        img_path = os.path.join(img_dir, img_id + ".jpg")
        if not os.path.exists(img_path):
            continue
        gt_classes = set(gt_map.get(img_id, []))
        if not gt_classes:
            continue
        t_img_start = time.time()
        img = Image.open(img_path).convert("RGB")
        results = yolo(np.array(img), conf=0.25, verbose=False)
        for box in results[0].boxes:
            yolo_cls = yolo.names[int(box.cls)]
            conf = float(box.conf)
            total += 1
            if conf < CONF_THRESHOLD:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = img.crop((x1, y1, x2, y2))
                if crop.size[0] > 0 and crop.size[1] > 0:
                    vlm_calls += 1
                    t0 = time.time()
                    pred = qwen_lora_classify(crop, processor, model)
                    vlm_times.append((time.time() - t0) * 1000)
                else:
                    pred = yolo_cls
            else:
                pred = yolo_cls
            if pred in gt_classes:
                correct += 1
            update_confusion(confusion, pred, gt_classes)
        img_times.append((time.time() - t_img_start) * 1000)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(img_ids)}")

    acc = round(correct / total * 100, 2) if total else 0
    avg_ms = round(np.mean(vlm_times), 1) if vlm_times else 0
    avg_img_ms = round(np.mean(img_times), 1) if img_times else 0
    p95_img_ms = round(float(np.percentile(img_times, 95)), 1) if img_times else 0
    call_rate = round(vlm_calls / total * 100, 1) if total else 0

    result = {
        "model": "Qwen3.5-0.8B-LoRA",
        "mode": "adaptive",
        "conf_threshold": CONF_THRESHOLD,
        "acc": acc,
        "total_boxes": total,
        "vlm_calls": vlm_calls,
        "vlm_call_rate": call_rate,
        "avg_ms_per_crop": avg_ms,
        "avg_ms_per_image_e2e": avg_img_ms,
        "p95_ms_per_image_e2e": p95_img_ms,
        "vram_gb": vram_gb,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }

    print("\n===== Qwen3.5-LoRA 自适应路由 =====")
    print(f"分类准确率:        {acc}%  ({correct}/{total})")
    print(f"VLM调用数:         {vlm_calls} / {total} ({call_rate}%)")
    print(f"平均推理时间:      {avg_ms} ms/crop")
    print(f"端到端延迟(avg):   {avg_img_ms} ms/图")
    print(f"端到端延迟(p95):   {p95_img_ms} ms/图")
    print(f"显存:              {vram_gb} GB")
    print_confusion_summary(confusion)

    out_path = "results_qwen_lora_adaptive.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n结果已保存到 {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="/home/user/2026/yolo/VOCdevkit/VOC2012")
    parser.add_argument("--model_path", default="/home/user/2026/yolo/best.pt")
    parser.add_argument("--base_vlm_path", default="/home/user/library/code/Qwen3.5-0.8B")
    parser.add_argument("--lora_path", default="/home/user/2026/yolo/qwen_lora/best")
    parser.add_argument("--max_images", type=int, default=500)
    args = parser.parse_args()
    main(args)
