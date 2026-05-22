"""
基于人工标注的框级别精确评估。
对比YOLO、Qwen3.5 zero-shot、Qwen3.5 LoRA在低置信度框上的真实准确率。

用法：
python eval_box_level.py \
  --meta_path /home/user/2026/yolo/low_conf_crops/metadata.json \
  --crops_dir /home/user/2026/yolo/low_conf_crops \
  --base_vlm_path /home/user/library/code/Qwen3.5-0.8B \
  --lora_path /home/user/2026/yolo/qwen_lora/best
"""
import argparse
import json
import time

import numpy as np
import torch
from PIL import Image

from eval_utils import VOC_CLASSES, VOC_CLASSES_STR


def load_qwen_base(vlm_path):
    from transformers import AutoProcessor
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        vlm_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()
    return processor, model


def load_qwen_lora(base_vlm_path, lora_path):
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


def classify(crop_pil, processor, model):
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


def evaluate(samples, crops_dir, processor, model, label):
    correct, total = 0, 0
    per_class = {}
    for s in samples:
        true_label = s['true_label']
        img_path = f"{crops_dir}/{s['filename']}"
        crop = Image.open(img_path).convert("RGB")
        pred = classify(crop, processor, model)
        total += 1
        if pred == true_label:
            correct += 1
        if true_label not in per_class:
            per_class[true_label] = {'correct': 0, 'total': 0}
        per_class[true_label]['total'] += 1
        if pred == true_label:
            per_class[true_label]['correct'] += 1

    acc = correct / total * 100 if total else 0
    print(f"\n[{label}] 准确率: {acc:.2f}% ({correct}/{total})")
    print("难类表现:")
    hard_classes = ['sofa', 'chair', 'diningtable', 'bottle', 'pottedplant', 'boat']
    for cls in hard_classes:
        if cls in per_class:
            d = per_class[cls]
            print(f"  {cls:15s}: {d['correct']}/{d['total']} ({d['correct']/d['total']*100:.1f}%)")
    return acc, per_class


def main(args):
    with open(args.meta_path) as f:
        metadata = json.load(f)

    # 过滤ambiguous，只用有明确标签的框
    samples = [m for m in metadata if m['true_label'] not in (None, 'ambiguous')]
    print(f"有效样本: {len(samples)} / {len(metadata)} (排除ambiguous {len(metadata)-len(samples)}个)")

    # YOLO baseline（直接用yolo_pred对比true_label）
    yolo_correct = sum(1 for s in samples if s['yolo_pred'] == s['true_label'])
    yolo_acc = yolo_correct / len(samples) * 100
    print(f"\n[YOLO baseline] 准确率: {yolo_acc:.2f}% ({yolo_correct}/{len(samples)})")

    # Qwen3.5 zero-shot
    print("\n加载Qwen3.5 zero-shot...")
    processor, model = load_qwen_base(args.base_vlm_path)
    zs_acc, zs_per_class = evaluate(samples, args.crops_dir, processor, model, "Qwen3.5 zero-shot")
    del model
    torch.cuda.empty_cache()

    # Qwen3.5 LoRA
    print("\n加载Qwen3.5 LoRA...")
    processor, model = load_qwen_lora(args.base_vlm_path, args.lora_path)
    lora_acc, lora_per_class = evaluate(samples, args.crops_dir, processor, model, "Qwen3.5 LoRA")

    print("\n===== 框级别精确评估汇总 =====")
    print(f"YOLO baseline:      {yolo_acc:.2f}%")
    print(f"Qwen3.5 zero-shot:  {zs_acc:.2f}%")
    print(f"Qwen3.5 LoRA:       {lora_acc:.2f}%")

    result = {
        "total_samples": len(samples),
        "ambiguous_excluded": len(metadata) - len(samples),
        "yolo_acc": round(yolo_acc, 2),
        "zeroshot_acc": round(zs_acc, 2),
        "lora_acc": round(lora_acc, 2),
    }
    with open("results_box_level.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n结果已保存到 results_box_level.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", default="/home/user/2026/yolo/low_conf_crops/metadata.json")
    parser.add_argument("--crops_dir", default="/home/user/2026/yolo/low_conf_crops")
    parser.add_argument("--base_vlm_path", default="/home/user/library/code/Qwen3.5-0.8B")
    parser.add_argument("--lora_path", default="/home/user/2026/yolo/qwen_lora/best")
    args = parser.parse_args()
    main(args)
