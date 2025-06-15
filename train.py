import os
from ultralytics import YOLO

DATA_PATH = "./dataset/data.yaml"

model = YOLO('yolov8s.pt')

training_params = {
    # === Core Training ===
    'data': DATA_PATH,
    'epochs': 100,                    
    'batch': 8,                        
    'imgsz': 640,                     
    'lr0': 0.005,                       
    'lrf': 0.1,                        
    'momentum': 0.937,                 
    'weight_decay': 0.001,
    'warmup_epochs': 5.0,              
    'warmup_momentum': 0.8,

    # === Loss Weights ===
    'box': 4.0,                        
    'cls': 1.0,                       
    'dfl': 1.5,

    # === Hardware/Setup ===
    'device': "cpu",                
    'name': "rasoi_yolov8s_2_clean",
    'pretrained': True,               

    # === Augmentations ===
    'hsv_h': 0.015,                  
    'hsv_s': 0.7,                    
    'hsv_v': 0.4,
    'degrees': 15.0,
    'translate': 0.1,
    'scale': 0.5,                     
    'shear': 1.0,
    'perspective': 0.0003,
    'flipud': 0.0,
    'fliplr': 0.7,
    'mosaic': 1.0,                     
    'mixup': 0.2,                     
    'copy_paste': 0.2,                
    'erasing': 0.1,                    

    # === Advanced ===
    'close_mosaic': 15,                
    'overlap_mask': True,
    'single_cls': False,
    'save_period': -1,
    'seed': 42,
    'patience': 50,                    
    'workers': 4                       
}

# Start training
results = model.train(**training_params)

model_path = "./models/model.pt"
model.save(model_path)
print(f"Model saved to {model_path}")