"""
LoRA消融实验：rank和target_modules的影响。

消融组合（共4组，加上已有的rank=8 qv作为baseline）：
  rank=4,  target=q_proj+v_proj
  rank=8,  target=q_proj+v_proj  （已有，跳过训练，直接评估）
  rank=16, target=q_proj+v_proj
  rank=8,  target=q_proj+v_proj+k_proj

用法：
  python ablation_lora.py \
    --data_path /home/user/2026/yolo/finetune_data.json \
    --vlm_path /home/user/library/code/Qwen3.5-0.8B \
    --output_dir /home/user/2026/yolo/ablation_lora \
    --existing_r8_qv /home/user/2026/yolo/qwen_lora/best \
    --meta_path /home/user/2026/yolo/low_conf_crops/metadata.json \
    --crops_dir /home/user/2026/yolo/low_conf_crops
"""
import argparse
import base64
import json
import os
import random
from io import BytesIO

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor
from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

from eval_utils import VOC_CLASSES, VOC_CLASSES_STR


# ── Dataset ──────────────────────────────────────────────────────────────────

def b64_to_pil(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


class VOCCropDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor
        self.question = (
            f"Choose the most likely category from: {VOC_CLASSES_STR}. "
            "Answer with only the category name."
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = b64_to_pil(s["image_b64"])
        label = s["label"]

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": self.question},
        ]}]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt + label

        inputs = self.processor(
            text=[full_text], images=[image], return_tensors="pt", padding=False
        )
        prompt_inputs = self.processor(
            text=[prompt], images=[image], return_tensors="pt", padding=False
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]

        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "image_grid_thw": inputs["image_grid_thw"].squeeze(0),
            "mm_token_type_ids": inputs["mm_token_type_ids"].squeeze(0),
            "labels": labels,
        }


def collate_fn(batch):
    assert len(batch) == 1, "batch_size must be 1"
    b = batch[0]
    return {k: v.unsqueeze(0) for k, v in b.items()}


# ── Training ──────────────────────────────────────────────────────────────────

def train(args, train_samples, val_samples, lora_r, target_modules, output_dir):
    processor = AutoProcessor.from_pretrained(args.vlm_path, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.vlm_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_loader = DataLoader(
        VOCCropDataset(train_samples, processor),
        batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=2,
    )
    val_loader = DataLoader(
        VOCCropDataset(val_samples, processor),
        batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-4, weight_decay=0.01,
    )

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss, steps = 0.0, 0
        for batch in train_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            steps += 1
            if steps % 100 == 0:
                print(f"  epoch {epoch+1} step {steps}  loss={total_loss/steps:.4f}")

        model.eval()
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.cuda() for k, v in batch.items()}
                val_loss += model(**batch).loss.item()
                val_steps += 1
        val_loss /= max(val_steps, 1)
        print(f"epoch {epoch+1}/{args.epochs}  train={total_loss/max(steps,1):.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(os.path.join(output_dir, "best"))
            processor.save_pretrained(os.path.join(output_dir, "best"))
            print(f"  -> saved (val_loss={val_loss:.4f})")

    del model
    torch.cuda.empty_cache()
    return os.path.join(output_dir, "best")


# ── Evaluation ────────────────────────────────────────────────────────────────

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


def evaluate_lora(lora_path, base_vlm_path, samples, crops_dir, label):
    processor = AutoProcessor.from_pretrained(base_vlm_path, trust_remote_code=True)
    base_model = Qwen3_5ForConditionalGeneration.from_pretrained(
        base_vlm_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload().to(torch.bfloat16).eval()

    correct, total = 0, 0
    for s in samples:
        crop = Image.open(f"{crops_dir}/{s['filename']}").convert("RGB")
        pred = classify(crop, processor, model)
        if pred == s["true_label"]:
            correct += 1
        total += 1

    acc = correct / total * 100 if total else 0
    print(f"[{label}] {acc:.2f}% ({correct}/{total})")
    del model
    torch.cuda.empty_cache()
    return round(acc, 2)


# ── Main ──────────────────────────────────────────────────────────────────────

ABLATION_CONFIGS = [
    {"lora_r": 4,  "target_modules": ["q_proj", "v_proj"],           "name": "r4_qv"},
    {"lora_r": 16, "target_modules": ["q_proj", "v_proj"],           "name": "r16_qv"},
    {"lora_r": 8,  "target_modules": ["q_proj", "v_proj", "k_proj"], "name": "r8_qvk"},
]


def main(args):
    random.seed(42)
    torch.manual_seed(42)

    print("[1/3] 加载数据...")
    with open(args.data_path) as f:
        samples = json.load(f)
    random.shuffle(samples)
    split = int(len(samples) * 0.9)
    train_samples, val_samples = samples[:split], samples[split:]
    print(f"  train: {len(train_samples)}  val: {len(val_samples)}")

    print("[2/3] 加载评估集...")
    with open(args.meta_path) as f:
        metadata = json.load(f)
    eval_samples = [m for m in metadata if m["true_label"] not in (None, "ambiguous")]
    print(f"  有效评估样本: {len(eval_samples)}")

    results = {}

    # 已有的r8_qv直接评估，跳过训练
    if args.existing_r8_qv:
        print("\n[r8_qv] 使用已有checkpoint，跳过训练...")
        acc = evaluate_lora(
            args.existing_r8_qv, args.vlm_path,
            eval_samples, args.crops_dir, "r8_qv (existing)"
        )
        results["r8_qv"] = {"lora_r": 8, "target_modules": "q_proj+v_proj", "acc": acc}

    print("[3/3] 开始消融训练...")
    for cfg in ABLATION_CONFIGS:
        name = cfg["name"]
        print(f"\n{'='*50}")
        print(f"消融组: {name}  rank={cfg['lora_r']}  targets={cfg['target_modules']}")
        print('='*50)

        out_dir = os.path.join(args.output_dir, name)
        best_path = os.path.join(out_dir, "best")

        if os.path.exists(best_path):
            print(f"  checkpoint已存在，跳过训练: {best_path}")
        else:
            best_path = train(
                args, train_samples, val_samples,
                cfg["lora_r"], cfg["target_modules"], out_dir
            )

        acc = evaluate_lora(
            best_path, args.vlm_path,
            eval_samples, args.crops_dir, name
        )
        results[name] = {
            "lora_r": cfg["lora_r"],
            "target_modules": "+".join(cfg["target_modules"]),
            "acc": acc,
        }

    print("\n" + "="*50)
    print("消融实验汇总（框级别准确率，187样本）")
    print("="*50)
    print(f"{'配置':<15} {'rank':<6} {'target_modules':<25} {'准确率'}")
    print("-"*60)
    for name, r in sorted(results.items()):
        print(f"{name:<15} {r['lora_r']:<6} {r['target_modules']:<25} {r['acc']:.2f}%")

    out_path = os.path.join(args.output_dir, "ablation_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到 {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",       default="/home/user/2026/yolo/finetune_data.json")
    parser.add_argument("--vlm_path",        default="/home/user/library/code/Qwen3.5-0.8B")
    parser.add_argument("--output_dir",      default="/home/user/2026/yolo/ablation_lora")
    parser.add_argument("--existing_r8_qv",  default="/home/user/2026/yolo/qwen_lora/best",
                        help="已有的rank=8 q+v checkpoint路径，跳过训练直接评估")
    parser.add_argument("--meta_path",       default="/home/user/2026/yolo/low_conf_crops/metadata.json")
    parser.add_argument("--crops_dir",       default="/home/user/2026/yolo/low_conf_crops")
    parser.add_argument("--epochs",          type=int, default=3)
    args = parser.parse_args()
    main(args)
