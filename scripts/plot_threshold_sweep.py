"""
画阈值扫描对比图：Base vs LoRA
用法：python plot_threshold_sweep.py
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

with open("results_threshold_sweep_base.json") as f:
    base = json.load(f)
with open("results_threshold_sweep_lora.json") as f:
    lora = json.load(f)

def extract(data):
    s = data["sweep"]
    return (
        [d["threshold"]               for d in s],
        [d["acc"]                     for d in s],
        [d["vlm_call_rate"]           for d in s],
        [d["est_total_ms_per_image"]  for d in s],
    )

thresholds, base_acc, call_rates, base_lat = extract(base)
_,          lora_acc, _,          lora_lat = extract(lora)

fig, (ax_acc, ax_lat) = plt.subplots(1, 2, figsize=(13, 5))

# ── 左图：准确率对比 + 调用率 ────────────────────────────────
ax_cr = ax_acc.twinx()

ax_acc.plot(thresholds, base_acc, color="#2563EB", marker="o", lw=2,   label="Base accuracy")
ax_acc.plot(thresholds, lora_acc, color="#7C3AED", marker="o", lw=2,   label="LoRA accuracy", linestyle="--")
ax_cr.plot( thresholds, call_rates, color="#9CA3AF", marker="", lw=1.5, linestyle=":", label="VLM call rate")

# 标注 base 峰值
base_peak_idx = int(np.argmax(base_acc))
ax_acc.annotate(
    f"Base peak\n{thresholds[base_peak_idx]:.2f} → {base_acc[base_peak_idx]}%",
    xy=(thresholds[base_peak_idx], base_acc[base_peak_idx]),
    xytext=(thresholds[base_peak_idx] - 0.18, base_acc[base_peak_idx] - 1.2),
    fontsize=8, color="#2563EB",
    arrowprops=dict(arrowstyle="->", color="#2563EB", lw=0.8),
)
# 标注 threshold=0.5 原始结果
idx50 = thresholds.index(0.5)
ax_acc.axvline(0.5, color="#DC2626", lw=1, linestyle="--", alpha=0.5)
ax_acc.annotate(
    f"Original\n0.50",
    xy=(0.5, base_acc[idx50]),
    xytext=(0.52, base_acc[idx50] - 1.5),
    fontsize=8, color="#DC2626",
    arrowprops=dict(arrowstyle="->", color="#DC2626", lw=0.8),
)

ax_acc.set_xlabel("Routing Threshold", fontsize=11)
ax_acc.set_ylabel("Accuracy (%)", fontsize=11)
ax_cr.set_ylabel("VLM Call Rate (%)", color="#9CA3AF", fontsize=10)
ax_acc.set_ylim(87.5, 94.5)
ax_cr.set_ylim(0, 120)
ax_acc.set_xticks(thresholds[::2])
ax_acc.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax_acc.grid(linestyle="--", alpha=0.3)
ax_acc.set_title("Accuracy vs Threshold (Base vs LoRA)", fontsize=11)

lines1 = ax_acc.get_lines() + ax_cr.get_lines()
ax_acc.legend(lines1, [l.get_label() for l in lines1], fontsize=8, loc="lower right")

# ── 右图：延迟对比 ────────────────────────────────────────────
ax_lat.plot(thresholds, base_lat, color="#2563EB", marker="s", lw=2,   label=f"Base (avg VLM {base['avg_vlm_ms_per_crop']}ms)")
ax_lat.plot(thresholds, lora_lat, color="#7C3AED", marker="s", lw=2,   label=f"LoRA (avg VLM {lora['avg_vlm_ms_per_crop']}ms)", linestyle="--")
ax_lat.axvline(0.5, color="#DC2626", lw=1, linestyle="--", alpha=0.5, label="Original threshold=0.50")

ax_lat.set_xlabel("Routing Threshold", fontsize=11)
ax_lat.set_ylabel("Est. Latency (ms/image)", fontsize=11)
ax_lat.set_xticks(thresholds[::2])
ax_lat.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax_lat.grid(linestyle="--", alpha=0.3)
ax_lat.set_title("Estimated Latency vs Threshold", fontsize=11)
ax_lat.legend(fontsize=8)

plt.suptitle("Routing Threshold Sweep — Qwen3.5-0.8B Base vs LoRA\n(VOC2012 val, 500 images, image-level accuracy)",
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig("threshold_sweep.png", dpi=150, bbox_inches="tight")
print("图已保存到 threshold_sweep.png")
