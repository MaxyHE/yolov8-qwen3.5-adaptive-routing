"""
评估结果分析：per-class分类准确率对比 + 可视化
生成：
  - eval_results/per_class_acc.json
  - eval_results/per_class_acc.png
  - eval_results/summary_table.txt
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results")
OUT_DIR = RESULTS_DIR

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

HARD_CLASSES = {"bottle", "pottedplant", "chair", "boat", "diningtable", "sofa"}


def load(fname):
    with open(os.path.join(RESULTS_DIR, fname)) as f:
        return json.load(f)


def per_class_acc(confusion, total_boxes):
    """从confusion matrix算每类的分类准确率（以该类为GT的框里预测正确的比例）"""
    gt_total = {cls: 0 for cls in VOC_CLASSES}
    gt_correct = {cls: 0 for cls in VOC_CLASSES}
    for pred, gt_dict in confusion.items():
        for gt, cnt in gt_dict.items():
            if gt in gt_total:
                gt_total[gt] += cnt
                if pred == gt:
                    gt_correct[gt] += cnt
    acc = {}
    for cls in VOC_CLASSES:
        acc[cls] = round(gt_correct[cls] / gt_total[cls] * 100, 1) if gt_total[cls] else 0.0
    return acc


def main():
    yolo = load("results_yolo.json")
    qwen_ada = load("results_qwen_adaptive.json")
    qwen_full = load("results_qwen_full.json")
    ivl_ada = load("results_ivl_adaptive.json")

    acc_yolo = per_class_acc(yolo["confusion"], yolo["total_boxes"])
    acc_qwen_ada = per_class_acc(qwen_ada["confusion"], qwen_ada["total_boxes"])
    acc_qwen_full = per_class_acc(qwen_full["confusion"], qwen_full["total_boxes"])
    acc_ivl_ada = per_class_acc(ivl_ada["confusion"], ivl_ada["total_boxes"])

    # 按YOLO准确率升序排列，难类在左
    sorted_cls = sorted(VOC_CLASSES, key=lambda c: acc_yolo[c])

    # ---- 汇总表 ----
    lines = []
    lines.append(f"{'Class':<15} {'YOLO':>6} {'Qwen-Ada':>9} {'Qwen-Full':>10} {'IVL-Ada':>8}  {'AP@0.5':>7}")
    lines.append("-" * 62)
    for cls in sorted_cls:
        ap = yolo["per_class_ap50"].get(cls, 0)
        hard = " *" if cls in HARD_CLASSES else ""
        lines.append(
            f"{cls:<15} {acc_yolo[cls]:>5.1f}% {acc_qwen_ada[cls]:>8.1f}% "
            f"{acc_qwen_full[cls]:>9.1f}% {acc_ivl_ada[cls]:>7.1f}%  {ap:>6.4f}{hard}"
        )
    lines.append("-" * 62)
    lines.append(
        f"{'Overall':<15} {yolo['cls_acc']:>5.1f}% {qwen_ada['acc']:>8.1f}% "
        f"{qwen_full['acc']:>9.1f}% {ivl_ada['acc']:>7.1f}%"
    )
    lines.append("\n* = hard class (AP@0.5 < 0.60)")

    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(OUT_DIR, "summary_table.txt"), "w") as f:
        f.write(summary)

    # ---- per-class acc json ----
    out = {cls: {
        "yolo": acc_yolo[cls],
        "qwen_adaptive": acc_qwen_ada[cls],
        "qwen_full": acc_qwen_full[cls],
        "ivl_adaptive": acc_ivl_ada[cls],
        "ap50": yolo["per_class_ap50"].get(cls, 0),
    } for cls in VOC_CLASSES}
    with open(os.path.join(OUT_DIR, "per_class_acc.json"), "w") as f:
        json.dump(out, f, indent=2)

    # ---- 柱状图 ----
    x = np.arange(len(sorted_cls))
    w = 0.22
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.bar(x - 1.5*w, [acc_yolo[c] for c in sorted_cls],       w, label="YOLO only",        color="#4C72B0")
    ax.bar(x - 0.5*w, [acc_qwen_ada[c] for c in sorted_cls],   w, label="Qwen3.5 Adaptive", color="#55A868")
    ax.bar(x + 0.5*w, [acc_qwen_full[c] for c in sorted_cls],  w, label="Qwen3.5 Full",     color="#C44E52")
    ax.bar(x + 1.5*w, [acc_ivl_ada[c] for c in sorted_cls],    w, label="InternVL2 Adaptive",color="#8172B2")

    # 标注难类
    for i, cls in enumerate(sorted_cls):
        if cls in HARD_CLASSES:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.08, color="red")

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_cls, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Classification Accuracy (%)")
    ax.set_title("Per-Class Classification Accuracy: YOLO vs VLM Routing\n(sorted by YOLO acc, red shading = hard classes AP<0.60)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "per_class_acc.png"), dpi=150)
    print(f"\n图已保存: {OUT_DIR}/per_class_acc.png")
    print(f"表已保存: {OUT_DIR}/summary_table.txt")


if __name__ == "__main__":
    main()
