"""
准确率对比图：框级别 + 图像级别sweep曲线
用法：python plot_accuracy_comparison.py
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── 框级别数据（低置信度框上的精确评估）────────────────────────
box_yolo     = 72.19
box_zeroshot = 78.61
box_lora     = 90.37   # 修thinking bug后的正确数字

# ── 图像级别sweep数据 ─────────────────────────────────────────
with open("results_threshold_sweep_base.json") as f:
    base = json.load(f)
with open("results_threshold_sweep_lora.json") as f:
    lora = json.load(f)

thresholds  = [d["threshold"] for d in base["sweep"]]
base_acc    = [d["acc"]       for d in base["sweep"]]
lora_acc    = [d["acc"]       for d in lora["sweep"]]

# YOLO-only baseline（sweep里call_rate=0时等价）
# 用sweep结果中最低阈值以下近似：直接用yolo_acc from results_yolo
yolo_img_acc = 88.38   # from results_yolo.json cls_acc

fig, (ax_box, ax_sweep) = plt.subplots(1, 2, figsize=(13, 5))

# ── 左图：框级别三方对比柱状图 ───────────────────────────────
labels = ["YOLO\nbaseline", "Qwen3.5\nZero-shot", "Qwen3.5\nLoRA"]
values = [box_yolo, box_zeroshot, box_lora]
colors = ["#6B7280", "#2563EB", "#7C3AED"]
bars = ax_box.bar(labels, values, color=colors, width=0.5, zorder=3)

for bar, val in zip(bars, values):
    ax_box.text(bar.get_x() + bar.get_width() / 2,
                val + 0.4, f"{val:.2f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

# 标注提升幅度
ax_box.annotate("",
    xy=(2, box_lora), xytext=(0, box_yolo),
    arrowprops=dict(arrowstyle="<->", color="#DC2626", lw=1.5,
                    connectionstyle="arc3,rad=0.0"))
ax_box.text(2.28, (box_yolo + box_lora) / 2,
            f"+{box_lora - box_yolo:.2f}pp", color="#DC2626",
            fontsize=9, va="center")

ax_box.set_ylim(65, 96)
ax_box.set_ylabel("Accuracy (%)", fontsize=11)
ax_box.set_title("Box-level Accuracy\n(Low-confidence crops, 187 samples)", fontsize=11)
ax_box.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
ax_box.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

# ── 右图：图像级别sweep曲线 ──────────────────────────────────
ax_sweep.axhline(yolo_img_acc, color="#6B7280", lw=1.5, linestyle=":",
                 label=f"YOLO only ({yolo_img_acc}%)")
ax_sweep.plot(thresholds, base_acc, color="#2563EB", marker="o", lw=2,
              label="Zero-shot (adaptive)")
ax_sweep.plot(thresholds, lora_acc, color="#7C3AED", marker="o", lw=2,
              linestyle="--", label="LoRA (adaptive)")

# 标注base峰值
base_peak_idx = int(np.argmax(base_acc))
ax_sweep.annotate(
    f"Base peaks\n@ {thresholds[base_peak_idx]:.2f} → {base_acc[base_peak_idx]}%",
    xy=(thresholds[base_peak_idx], base_acc[base_peak_idx]),
    xytext=(thresholds[base_peak_idx] - 0.22, base_acc[base_peak_idx] - 0.8),
    fontsize=8, color="#2563EB",
    arrowprops=dict(arrowstyle="->", color="#2563EB", lw=0.8),
)
# 标注LoRA在最高阈值的结果
ax_sweep.annotate(
    f"LoRA: {lora_acc[-1]}%",
    xy=(thresholds[-1], lora_acc[-1]),
    xytext=(thresholds[-1] - 0.22, lora_acc[-1] - 0.8),
    fontsize=8, color="#7C3AED",
    arrowprops=dict(arrowstyle="->", color="#7C3AED", lw=0.8),
)
# 标注threshold=0.5
ax_sweep.axvline(0.5, color="#DC2626", lw=1, linestyle="--", alpha=0.6)
ax_sweep.text(0.51, 88.6, "threshold=0.50\n(default)", fontsize=8,
              color="#DC2626", va="bottom")

ax_sweep.set_xlabel("Routing Threshold", fontsize=11)
ax_sweep.set_ylabel("Image-level Accuracy (%)", fontsize=11)
ax_sweep.set_title("Image-level Accuracy vs Threshold\n(VOC2012 val, 500 images)", fontsize=11)
ax_sweep.set_ylim(87.5, 94.5)
ax_sweep.set_xticks(thresholds[::2])
ax_sweep.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax_sweep.grid(linestyle="--", alpha=0.3)
ax_sweep.legend(fontsize=9, loc="lower right")

plt.suptitle(
    "YOLO + Qwen3.5-0.8B Adaptive Routing — Accuracy Comparison\n"
    "Left: box-level on hard cases  |  Right: image-level sweep",
    fontsize=11, y=1.02
)
plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=150, bbox_inches="tight")
print("图已保存到 accuracy_comparison.png")
