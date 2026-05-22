"""
对ablation_lora里所有消融checkpoint做框级别精确评估。
同时重新评估r8_qv（原始LoRA）确认数字。

用法：
  python eval_ablation_box_level.py \
    --base_vlm_path /home/user/library/code/Qwen3.5-0.8B \
    --ablation_dir /home/user/2026/yolo/ablation_lora \
    --r8_qv_path /home/user/2026/yolo/qwen_lora/best \
    --meta_path /home/user/2026/yolo/low_conf_crops/metadata.json \
    --crops_dir /home/user/2026/yolo/low_conf_crops
"""
import argparse
import json
import os

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor
from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration

from eval_utils import VOC_CLASSES, VOC_CLASSES_STR


def load_lora(base_vlm_path, lora_path):
    processor = AutoProcessor.from_pretrained(base_vlm_path, trust_remote_code=True)
    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_vlm_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload().to(torch.bfloat16).eval()
    return processor, model


def classify(crop_pil, processor, model):
    question = (
        f"Choose the most likely category from: {VOC_CLASSES_STR}. "
        "Answer with only the category name."
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": crop_pil},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(text=[text], images=[crop_pil], return_tensors="pt").to("cuda")
    gen_inputs = {k: v for k, v in inputs.items()
                  if k in ("input_ids", "attention_mask", "pixel_values",
                           "image_grid_thw", "mm_token_type_ids")}
    with torch.no_grad():
        out = model.generate(**gen_inputs, max_new_tokens=16, do_sample=False)
    result = processor.batch_decode(
        out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )[0].strip().lower()
    for cls in VOC_CLASSES:
        if cls in result:
            return cls
    return result


def evaluate(samples, crops_dir, processor, model, label):
    correct, total = 0, 0
    for s in samples:
        crop = Image.open(f"{crops_dir}/{s['filename']}").convert("RGB")
        pred = classify(crop, processor, model)
        if pred == s["true_label"]:
            correct += 1
        total += 1
    acc = correct / total * 100 if total else 0
    print(f"[{label}] {acc:.2f}% ({correct}/{total})")
    return round(acc, 2)


def main(args):
    with open(args.meta_path) as f:
        metadata = json.load(f)
    samples = [m for m in metadata if m["true_label"] not in (None, "ambiguous")]
    print(f"有效评估样本: {len(samples)}\n")

    # 读取已有的ablation_results（训练时的val loss评估结果）
    ablation_json = os.path.join(args.ablation_dir, "ablation_results.json")
    with open(ablation_json) as f:
        ablation_meta = json.load(f)

    results = {}

    # r8_qv原始checkpoint
    print("=" * 50)
    print(f"评估 r8_qv (原始LoRA): {args.r8_qv_path}")
    processor, model = load_lora(args.base_vlm_path, args.r8_qv_path)
    acc = evaluate(samples, args.crops_dir, processor, model, "r8_qv")
    results["r8_qv"] = {
        "lora_r": 8,
        "target_modules": "q_proj+v_proj",
        "box_level_acc": acc,
    }
    del model
    torch.cuda.empty_cache()

    # 消融checkpoints
    ablation_configs = [
        {"name": "r4_qv",  "lora_r": 4,  "target_modules": "q_proj+v_proj"},
        {"name": "r16_qv", "lora_r": 16, "target_modules": "q_proj+v_proj"},
        {"name": "r8_qvk", "lora_r": 8,  "target_modules": "q_proj+v_proj+k_proj"},
    ]

    for cfg in ablation_configs:
        name = cfg["name"]
        lora_path = os.path.join(args.ablation_dir, name, "best")
        print("=" * 50)
        print(f"评估 {name}: {lora_path}")
        processor, model = load_lora(args.base_vlm_path, lora_path)
        acc = evaluate(samples, args.crops_dir, processor, model, name)
        results[name] = {
            "lora_r": cfg["lora_r"],
            "target_modules": cfg["target_modules"],
            "box_level_acc": acc,
        }
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 50)
    print("框级别精确评估汇总（187样本）")
    print("=" * 50)
    print(f"{'配置':<12} {'rank':<6} {'target_modules':<25} {'框级别准确率'}")
    print("-" * 60)
    for name, r in sorted(results.items()):
        print(f"{name:<12} {r['lora_r']:<6} {r['target_modules']:<25} {r['box_level_acc']:.2f}%")

    out_path = os.path.join(args.ablation_dir, "ablation_box_level_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_vlm_path", default="/home/user/library/code/Qwen3.5-0.8B")
    parser.add_argument("--ablation_dir",  default="/home/user/2026/yolo/ablation_lora")
    parser.add_argument("--r8_qv_path",    default="/home/user/2026/yolo/qwen_lora/best")
    parser.add_argument("--meta_path",     default="/home/user/2026/yolo/low_conf_crops/metadata.json")
    parser.add_argument("--crops_dir",     default="/home/user/2026/yolo/low_conf_crops")
    args = parser.parse_args()
    main(args)
