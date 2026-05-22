import os
import time
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
VOC_CLASSES_STR = ", ".join(VOC_CLASSES)
CONF_THRESHOLD = 0.5

# 模块级别，避免每次推理重复初始化
_INTERNVL_TRANSFORM = T.Compose([
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])


def get_img_ids(val_txt):
    with open(val_txt) as f:
        return [l.strip() for l in f if l.strip()]


def load_voc_annotations(ann_dir, img_ids):
    gt = {}
    for img_id in img_ids:
        xml_path = os.path.join(ann_dir, img_id + ".xml")
        if not os.path.exists(xml_path):
            continue
        root = ET.parse(xml_path).getroot()
        gt[img_id] = [obj.find("name").text for obj in root.findall("object")]
    return gt


def load_voc_box_annotations(ann_dir, img_ids):
    gt = {}
    for img_id in img_ids:
        xml_path = os.path.join(ann_dir, img_id + ".xml")
        if not os.path.exists(xml_path):
            continue
        root = ET.parse(xml_path).getroot()
        objects = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            if xmax <= xmin or ymax <= ymin:
                continue
            objects.append({
                "label": name,
                "bbox": [xmin, ymin, xmax, ymax],
            })
        gt[img_id] = objects
    return gt


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def greedy_match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5):
    candidates = []
    for pred_idx, pred in enumerate(pred_boxes):
        for gt_idx, gt in enumerate(gt_boxes):
            iou = box_iou(pred["bbox"], gt["bbox"])
            if iou >= iou_threshold:
                candidates.append((iou, pred_idx, gt_idx))

    candidates.sort(reverse=True)
    used_pred = set()
    used_gt = set()
    matches = []
    for iou, pred_idx, gt_idx in candidates:
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append({
            "pred_idx": pred_idx,
            "gt_idx": gt_idx,
            "iou": iou,
        })
    return matches


def measure_vram(load_fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    obj = load_fn()
    vram_gb = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
    return obj, vram_gb


def load_qwen(vlm_path):
    from transformers import AutoProcessor
    from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration
    processor = AutoProcessor.from_pretrained(vlm_path, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        vlm_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()
    return processor, model


def load_internvl(vlm_path):
    from transformers import AutoTokenizer, AutoModel
    from transformers.modeling_utils import PreTrainedModel

    # patch仅在加载时生效，加载完成后恢复原始方法，避免污染全局状态
    original_mark = PreTrainedModel.mark_tied_weights_as_initialized
    def patched_mark(self, loading_info):
        # 确保属性存在且为dict，避免transformers内部访问时触发torch __getattr__报错
        if not isinstance(getattr(self, 'all_tied_weights_keys', None), dict):
            object.__setattr__(self, 'all_tied_weights_keys', {})
        original_mark(self, loading_info)
    PreTrainedModel.mark_tied_weights_as_initialized = patched_mark
    try:
        tokenizer = AutoTokenizer.from_pretrained(vlm_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            vlm_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).cuda().eval()
    finally:
        PreTrainedModel.mark_tied_weights_as_initialized = original_mark
    return tokenizer, model


def qwen_classify(crop_pil, processor, model):
    question = f"Choose the most likely category from: {VOC_CLASSES_STR}. Answer with only the category name."
    messages = [{"role": "user", "content": [
        {"type": "image", "image": crop_pil},
        {"type": "text", "text": question}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=[text], images=[crop_pil], return_tensors="pt").to("cuda")
    gen_inputs = {k: v for k, v in inputs.items()
                  if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
    with torch.no_grad():
        out = model.generate(**gen_inputs, max_new_tokens=16, do_sample=False)
    result = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip().lower()
    for cls in VOC_CLASSES:
        if cls in result:
            return cls
    return result


def internvl_classify(crop_pil, tokenizer, model):
    pixel_values = _INTERNVL_TRANSFORM(crop_pil.convert("RGB")).unsqueeze(0).to(torch.bfloat16).cuda()
    question = f"<image>\nChoose the most likely category from: {VOC_CLASSES_STR}. Answer with only the category name."
    with torch.no_grad():
        result = model.chat(tokenizer, pixel_values, question, dict(max_new_tokens=16, do_sample=False))
    result = result.strip().lower()
    for cls in VOC_CLASSES:
        if cls in result:
            return cls
    return result


def warmup_yolo(yolo, img_dir, img_ids, n=5):
    for img_id in img_ids[:n]:
        img_path = os.path.join(img_dir, img_id + ".jpg")
        if os.path.exists(img_path):
            yolo(np.array(Image.open(img_path).convert("RGB")), conf=0.25, verbose=False)


def warmup_vlm_qwen(processor, model):
    dummy = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    qwen_classify(dummy, processor, model)


def warmup_vlm_internvl(tokenizer, model):
    dummy = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    internvl_classify(dummy, tokenizer, model)


def update_confusion(confusion, pred, gt_classes):
    """框级confusion matrix：pred与每个GT类别都计一次"""
    for gt_cls in gt_classes:
        confusion[pred][gt_cls] += 1


def print_confusion_summary(confusion, top_n=10):
    errors = []
    for pred, gt_dict in confusion.items():
        for gt, cnt in gt_dict.items():
            if pred != gt:
                errors.append((cnt, pred, gt))
    errors.sort(reverse=True)
    print(f"\nTop-{top_n} 误判对 (预测→真实):")
    for cnt, pred, gt in errors[:top_n]:
        print(f"  {pred:15s} → {gt:15s}: {cnt}次")
