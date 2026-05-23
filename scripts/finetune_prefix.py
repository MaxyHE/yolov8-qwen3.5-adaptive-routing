"""
Prefix Tuning 微调 Qwen3.5-0.8B，与 LoRA 消融对比。

Prefix Tuning 在每层 attention 的 K/V 前拼接可训练的 prefix 向量，
模型权重完全冻结，只训练 prefix。参数量远少于 LoRA。

用法：
python scripts/finetune_prefix.py \
  --data_path /home/user/2026/yolo/finetune_data.json \
  --vlm_path /home/user/library/code/Qwen3.5-0.8B \
  --output_dir /home/user/2026/yolo/prefix_tuning \
  --num_virtual_tokens 20 \
  --epochs 3
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
from peft import PrefixTuningConfig, get_peft_model, TaskType

from eval_utils import VOC_CLASSES, VOC_CLASSES_STR


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


def evaluate(samples, crops_dir, processor, model):
    correct, total = 0, 0
    for s in samples:
        crop = Image.open(f"{crops_dir}/{s['filename']}").convert("RGB")
        pred = classify(crop, processor, model)
        if pred == s["true_label"]:
            correct += 1
        total += 1
    return correct / total * 100 if total else 0


def main(args):
    random.seed(42)
    torch.manual_seed(42)

    print("[1/4] 加载数据...")
    with open(args.data_path) as f:
        samples = json.load(f)
    random.shuffle(samples)
    split = int(len(samples) * 0.9)
    train_samples, val_samples = samples[:split], samples[split:]
    print(f"  train: {len(train_samples)}  val: {len(val_samples)}")

    print("[2/4] 加载模型...")
    processor = AutoProcessor.from_pretrained(args.vlm_path, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.vlm_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda()

    prefix_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.num_virtual_tokens,
        prefix_projection=False,  # True会加一个MLP投影层，参数更多
    )
    model = get_peft_model(model, prefix_config)
    model.print_trainable_parameters()

    print("[3/4] 构建DataLoader...")
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
        lr=args.lr, weight_decay=0.01,
    )

    print("[4/4] 开始训练...")
    os.makedirs(args.output_dir, exist_ok=True)
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
            model.save_pretrained(os.path.join(args.output_dir, "best"))
            processor.save_pretrained(os.path.join(args.output_dir, "best"))
            print(f"  -> saved (val_loss={val_loss:.4f})")

    # 框级别评估
    print("\n加载评估集...")
    with open(args.meta_path) as f:
        metadata = json.load(f)
    eval_samples = [m for m in metadata if m["true_label"] not in (None, "ambiguous")]
    print(f"有效评估样本: {len(eval_samples)}")

    # prefix tuning 不能 merge_and_unload，直接用 peft model 推理
    model.eval()
    acc = evaluate(eval_samples, args.crops_dir, processor, model)
    print(f"\n[Prefix Tuning num_virtual_tokens={args.num_virtual_tokens}] 框级别准确率: {acc:.2f}%")

    result = {
        "method": "prefix_tuning",
        "num_virtual_tokens": args.num_virtual_tokens,
        "prefix_projection": False,
        "box_level_acc": round(acc, 2),
    }
    out_path = os.path.join(args.output_dir, "prefix_results.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"结果已保存到 {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",         default="/home/user/2026/yolo/finetune_data.json")
    parser.add_argument("--vlm_path",          default="/home/user/library/code/Qwen3.5-0.8B")
    parser.add_argument("--output_dir",        default="/home/user/2026/yolo/prefix_tuning")
    parser.add_argument("--meta_path",         default="/home/user/2026/yolo/low_conf_crops/metadata.json")
    parser.add_argument("--crops_dir",         default="/home/user/2026/yolo/low_conf_crops")
    parser.add_argument("--num_virtual_tokens", type=int, default=20)
    parser.add_argument("--epochs",            type=int, default=3)
    parser.add_argument("--lr",                type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
