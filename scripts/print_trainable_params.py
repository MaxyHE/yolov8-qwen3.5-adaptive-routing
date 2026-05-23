"""
打印各LoRA配置的可训练参数比例，不需要训练数据。

用法：
python scripts/print_trainable_params.py --vlm_path /home/user/library/code/Qwen3.5-0.8B
"""
import argparse
import torch
from transformers import AutoProcessor
from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType


CONFIGS = [
    {"name": "r4_qv",  "r": 4,  "targets": ["q_proj", "v_proj"]},
    {"name": "r8_qv",  "r": 8,  "targets": ["q_proj", "v_proj"]},
    {"name": "r16_qv", "r": 16, "targets": ["q_proj", "v_proj"]},
    {"name": "r8_qvk", "r": 8,  "targets": ["q_proj", "v_proj", "k_proj"]},
]


def main(args):
    print(f"加载基础模型: {args.vlm_path}")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.vlm_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"基础模型总参数: {total_params:,} ({total_params/1e6:.1f}M)\n")

    print(f"{'配置':<12} {'rank':>6} {'target_modules':<30} {'可训练参数':>12} {'比例':>8}")
    print("-" * 72)

    for cfg in CONFIGS:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg["r"],
            lora_alpha=cfg["r"] * 2,
            lora_dropout=0.05,
            target_modules=cfg["targets"],
            bias="none",
        )
        peft_model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        ratio = trainable / total_params * 100
        targets_str = "+".join(t.replace("_proj", "") for t in cfg["targets"])
        print(f"{cfg['name']:<12} {cfg['r']:>6} {targets_str:<30} {trainable:>12,} {ratio:>7.4f}%")

        # 重置为原始模型（去掉LoRA层）
        model = peft_model.base_model.model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_path", default="/home/user/library/code/Qwen3.5-0.8B")
    args = parser.parse_args()
    main(args)
