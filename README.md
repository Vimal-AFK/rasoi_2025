# 🥗 RASOI - YOLOv8 Model Evaluation Tool

## 📌 Overview

This tool evaluates a trained YOLOv8 model on a food object detection dataset. It compares predictions against annotated ground truth labels (from an Excel file) and calculates:

- ✅ Precision  
- ✅ Recall  
- ✅ F1 Score  
- ✅ Class-wise AP (Average Precision)  
- ✅ mAP@0.5 (mean Average Precision)

---

## 📂 Folder Structure

```
project/
├── models/
│   └── rasoi_yolov8s.pt         # Trained YOLOv8 model
├── validation_dataset/
│   ├── val_images/              # Test images (.jpg/.png/.jpeg)
│   └── val_labels.xlsx          # Ground truth annotations
├── predictions.csv              # Output predictions
├── metrics.json                 # Output evaluation metrics
└── inference.py                  # Evaluation script
```

---

## 📥 Input Format

- **Model Path**: YOLOv8 `.pt` file  
- **Image Folder**: Directory with test images  
- **Excel File**: Ground truth with bounding boxes and class labels

### 📝 Ground Truth Excel Format

| Filename    | Region Shape Attributes              | Region Attributes         |
|-------------|--------------------------------------|---------------------------|
| image1.jpg  | {"x": 100, "y": 120, "width": 50...} | {"name": "banana"}        |
| image2.jpg  | {"x": 80, "y": 95, "width": 40...}  | {"name": "apple"}         |

- `Region Shape Attributes`: JSON with `x`, `y`, `width`, `height`  
- `Region Attributes`: JSON with `"name"` key for class

---

## 🧪 Metrics Generated

| Metric     | Description                                  |
|------------|----------------------------------------------|
| Precision  | Correct predictions / Total predictions      |
| Recall     | Correct predictions / Total ground truths    |
| F1 Score   | Harmonic mean of Precision and Recall        |
| AP         | Average Precision for each class             |
| mAP@0.5    | Mean of all class AP scores at IoU=0.5       |

---

## 📊 Output Files

- **`predictions.csv`** – YOLO predictions with bounding boxes, classes, and confidence scores  
- **`metrics.json`** – Precision, Recall, F1, mAP, and per-class metrics

---

## 🚀 How to Run

```bash
python evaluate.py
```

Modify the following lines in `evaluate.py` if your paths differ:

```python
model_path = "./models/rasoi_yolov8s.pt"
images_path = "./validation_dataset/val_images"
labels_path = "./validation_dataset/val_labels.xlsx"
main(model_path, images_path, labels_path)
```

---

## 📦 Dependencies

```bash
pip install ultralytics pandas scikit-learn tqdm openpyxl gdown
```

---

## 🔗 Model File (Google Drive)

Due to GitHub's 100MB file limit, the model file is hosted externally.

📥 **[Download rasoi_yolov8s.pt]([https://drive.google.com/file/d/YOUR_FILE_ID/view?usp=sharing](https://drive.google.com/drive/folders/1JDvexVgl-zk5sE3HnEOJ3idFb8Tf4o3g?usp=sharing))**

To download via script:

```python
import gdown
file_id = "YOUR_FILE_ID"
gdown.download(f"https://drive.google.com/uc?id={file_id}", "rasoi_yolov8s.pt", quiet=False)
```

---

## ✅ Example Output

```bash
✅ Predictions saved to ./predictions.csv
✅ Metrics saved to ./metrics.json

=== Class-wise Metrics ===
banana: AP=0.7031, Precision=0.7647, Recall=0.6364
apple:  AP=0.6243, Precision=0.8125, Recall=0.5652

=== Evaluation Metrics ===
Precision: 0.6878
Recall: 0.7157
F1 Score: 0.7015
mAP@0.5: 0.6913
```

---

