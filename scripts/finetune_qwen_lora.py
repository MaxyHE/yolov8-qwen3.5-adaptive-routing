"""
用LoRA微调Qwen3.5-0.8B，任务：VOC20类受限分类。

用法：
python finetune_qwen_lora.py \
  --data_path /home/user/2026/yolo/finetune_data.json \
  --vlm_path /home/user/library/code/Qwen3.5-0.8B \
  --output_dir /home/user/2026/yolo/qwen_lora \
  --epochs 3 \
  --batch_size 4
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
from peft import LoraConfig, get_peft_model, TaskType

from eval_utils import VOC_CLASSES_STR


def b64_to_pil(b64: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")


class VOCCropDataset(Dataset):
    def __init__(self, samples, processor):
        self.samples = samples
        self.processor = processor
        self.question = f"Choose the most likely category from: {VOC_CLASSES_STR}. Answer with only the category name."

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = b64_to_pil(s["image_b64"])
        label = s["label"]

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": self.question}
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
        labels[:prompt_len] = -100  # 只对answer部分计算loss

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "image_grid_thw": inputs["image_grid_thw"].squeeze(0),
            "mm_token_type_ids": inputs["mm_token_type_ids"].squeeze(0),
            "labels": labels,
        }


def collate_fn(batch):
    # pixel_values的visual token数量随图像尺寸变化，只支持batch_size=1
    assert len(batch) == 1, "batch_size must be 1 due to variable visual token length"
    b = batch[0]
    return {
        "input_ids": b["input_ids"].unsqueeze(0),
        "attention_mask": b["attention_mask"].unsqueeze(0),
        "pixel_values": b["pixel_values"].unsqueeze(0),
        "image_grid_thw": b["image_grid_thw"].unsqueeze(0),
        "mm_token_type_ids": b["mm_token_type_ids"].unsqueeze(0),
        "labels": b["labels"].unsqueeze(0),
    }


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

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[3/4] 构建DataLoader...")
    train_dataset = VOCCropDataset(train_samples, processor)
    val_dataset = VOCCropDataset(val_samples, processor)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, collate_fn=collate_fn, num_workers=2)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01
    )

    print("[4/4] 开始训练...")
    os.makedirs(args.output_dir, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss, steps = 0.0, 0
        for batch in train_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            steps += 1
            if steps % 50 == 0:
                print(f"  epoch {epoch+1} step {steps}  loss={total_loss/steps:.4f}")

        # validation
        model.eval()
        val_loss, val_steps = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                val_steps += 1
        val_loss /= max(val_steps, 1)
        train_loss = total_loss / max(steps, 1)
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(os.path.join(args.output_dir, "best"))
            processor.save_pretrained(os.path.join(args.output_dir, "best"))
            print(f"  -> 保存最优checkpoint (val_loss={val_loss:.4f})")

    print(f"\n训练完成，最优模型保存在 {args.output_dir}/best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/user/2026/yolo/finetune_data.json")
    parser.add_argument("--vlm_path", default="/home/user/library/code/Qwen3.5-0.8B")
    parser.add_argument("--output_dir", default="/home/user/2026/yolo/qwen_lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()
    main(args)
