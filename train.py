from ultralytics import YOLO

model = YOLO('yolov8s.pt')

results = model.train(
    data='VOC2012.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=4,
    optimizer='SGD',
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    cos_lr=True,
    mixup=0.1,
    copy_paste=0.1,
)
