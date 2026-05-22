"""
生成自适应路由流程图
输出: eval_results/pipeline.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 5)
ax.axis("off")

def box(ax, x, y, w, h, text, color, fontsize=10):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                          facecolor=color, edgecolor="gray", linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", wrap=True)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

def label(ax, x, y, text, color="black", fontsize=9):
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=color)

# 节点
box(ax, 0.3, 1.8, 1.8, 1.4, "Input\nImage", "#AED6F1")
box(ax, 2.5, 1.8, 1.8, 1.4, "YOLOv8s\nDetector", "#A9DFBF")
box(ax, 4.8, 2.8, 2.0, 1.0, "conf ≥ 0.5?", "#F9E79F", fontsize=10)
box(ax, 4.8, 0.5, 2.0, 1.2, "Crop ROI\n→ Qwen3.5-0.8B\n(VLM Classify)", "#F1948A", fontsize=9)
box(ax, 8.2, 2.8, 1.8, 1.0, "YOLO\nLabel", "#A9DFBF")
box(ax, 8.2, 0.5, 1.8, 1.2, "VLM\nLabel", "#F1948A")
box(ax, 10.3, 1.5, 1.4, 2.0, "Final\nDetection\nResult", "#D2B4DE")

# 箭头
arrow(ax, 2.1, 2.5, 2.5, 2.5)
arrow(ax, 4.3, 2.5, 4.8, 3.3)
arrow(ax, 5.8, 2.8, 5.8, 1.7)
arrow(ax, 6.8, 3.3, 8.2, 3.3)
arrow(ax, 6.8, 1.1, 8.2, 1.1)
arrow(ax, 10.0, 3.3, 10.3, 2.5)
arrow(ax, 10.0, 1.1, 10.3, 1.8)

# 标注
label(ax, 7.5, 3.55, "Yes (high conf)", color="#1a7a1a", fontsize=9)
label(ax, 6.15, 2.2, "No\n(low conf)", color="#c0392b", fontsize=9)

ax.set_title("Adaptive VLM Routing Pipeline (YOLO + Qwen3.5-0.8B)",
             fontsize=13, fontweight="bold", pad=15)

plt.tight_layout()
plt.savefig("eval_results/pipeline.png", dpi=150, bbox_inches="tight")
print("saved: eval_results/pipeline.png")
