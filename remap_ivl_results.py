"""
对InternVL2结果做同义词映射后处理，重新计算准确率。
读取 eval_results/results_ivl_*.json，输出 eval_results/results_ivl_*_remapped.json
"""
import json
import os

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
VOC_SET = set(VOC_CLASSES)

SYNONYM_MAP = {
    "airplane": "aeroplane", "plane": "aeroplane",
    "motorcycle": "motorbike", "motocycle": "motorbike",
    "dining table": "diningtable", "dinnertable": "diningtable", "dine table": "diningtable",
    "potted plant": "pottedplant", "flower pot": "pottedplant",
    "tv": "tvmonitor", "television": "tvmonitor", "monitor": "tvmonitor",
    "baby": "person", "child": "person", "girl": "person", "boy": "person",
    "man": "person", "woman": "person", "kid": "person", "toddler": "person",
    "police officer": "person", "musician": "person", "bride and groom": "person",
    "boy scouts": "person", "band": "person", "saxophone player": "person",
    "bass guitar": "person",
    "couch": "sofa",
}


def remap(pred):
    if pred in VOC_SET:
        return pred
    if pred in SYNONYM_MAP:
        return SYNONYM_MAP[pred]
    for syn, voc in SYNONYM_MAP.items():
        if syn in pred:
            return voc
    for cls in VOC_CLASSES:
        if cls in pred:
            return cls
    return pred


def recompute(result):
    old_confusion = result["confusion"]
    total = result["total_boxes"]
    new_confusion = {}
    correct = 0

    for pred, gt_dict in old_confusion.items():
        new_pred = remap(pred)
        if new_pred not in new_confusion:
            new_confusion[new_pred] = {}
        for gt, cnt in gt_dict.items():
            new_confusion[new_pred][gt] = new_confusion[new_pred].get(gt, 0) + cnt
            if new_pred == gt:
                correct += cnt

    result["confusion"] = new_confusion
    result["acc_remapped"] = round(correct / total * 100, 2) if total else 0
    result["acc_original"] = result.pop("acc")
    return result, correct, total


results_dir = "/path/to/eval_results"

for fname in ("results_ivl_full.json", "results_ivl_adaptive.json"):
    src = os.path.join(results_dir, fname)
    if not os.path.exists(src):
        print(f"跳过（不存在）: {fname}")
        continue

    with open(src) as f:
        result = json.load(f)

    result, correct, total = recompute(result)

    dst = os.path.join(results_dir, fname.replace(".json", "_remapped.json"))
    with open(dst, "w") as f:
        json.dump(result, f, indent=2)

    print(f"{fname}")
    print(f"  原始准确率: {result['acc_original']}%")
    print(f"  映射后准确率: {result['acc_remapped']}%  ({correct}/{total})")
