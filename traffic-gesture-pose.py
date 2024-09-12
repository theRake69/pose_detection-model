from ultralytics import YOLO

model = YOLO('weights/best.pt')

data_yaml = '/Users/administrator/Documents/python_projects/traffic-gesture-model/human-pose/final_version.yolov8/data.yaml'

num_epochs = 450  
batch_size = 8 
initial_lr = 0.0001  # Lower initial learning rate for fine-tuning

model.train(
    data=data_yaml,
    epochs=num_epochs,
    imgsz=352,
    batch=batch_size,
    lr0=initial_lr,
    hsv_h=0.02,
    hsv_s=0.5,
    hsv_v=0.5,
    degrees=10,
    translate=0.1,
    scale=0.6,
    shear=10,
    perspective=0.0005,
    flipud=0.0,
    fliplr=0.5,
    bgr=0.0,
    copy_paste=0.2,
    erasing=0.5,
    crop_fraction=0.9,
    val=True,
    resume=False,  # Start a new training session
    mosaic=1.0,  # Increase mosaic augmentation strength
    mixup=0.5,  # Increase mixup augmentation strength
    auto_augment='randaugment',  # Apply random augmentations
    label_smoothing=0.1,  # Apply label smoothing
    dropout=0.2,  # Apply dropout for regularization
    weight_decay=0.0005,  # Apply weight decay
    warmup_epochs=5,  # Increase warmup epochs
    warmup_momentum=0.8,  # Set warmup momentum
    warmup_bias_lr=0.1,  # Set warmup bias learning rate
    patience=20,  # Early stopping patience to avoid overfitting
    cos_lr=True  # Use cosine learning rate scheduler
)
