import argparse
import os
import pandas as pd
import json
import numpy as np
from glob import glob
from collections import defaultdict
from ultralytics import YOLO
from sklearn.metrics import precision_recall_curve, auc
from tqdm import tqdm

class FoodDetectionEvaluator:
    def __init__(self, model_path, image_folder, label_path):
        self.model = YOLO(model_path)
        self.image_folder = image_folder
        self.label_path = label_path
        self.class_names = self.model.names
        self.ground_truths = self._parse_ground_truth()
        
    def _parse_ground_truth(self):
        try:
            df = pd.read_excel(self.label_path)
            ground_truths = {}
            
            for _, row in df.iterrows():
                filename = row[0].strip()
                shape_attr = json.loads(row[1])
                class_attr = json.loads(row[2])
                
                bbox = [
                    shape_attr['x'],
                    shape_attr['y'],
                    shape_attr['x'] + shape_attr['width'],
                    shape_attr['y'] + shape_attr['height']
                ]
                class_name = class_attr['name'].strip()
                
                if filename not in ground_truths:
                    ground_truths[filename] = []
                ground_truths[filename].append({'bbox': bbox, 'class': class_name})
                
            return ground_truths
        except Exception as e:
            raise ValueError(f"Error parsing ground truth file: {str(e)}")

    @staticmethod
    def _compute_iou(box1, box2):
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = min(box1[2], box2[2])
        yi2 = min(box1[3], box2[3])
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area else 0.0

    def _calculate_ap(self, y_true, y_scores):
        if not y_true:
            return 0.0
            
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        return auc(recall, precision)

    def evaluate(self):
        pred_records = []
        metrics = {
            'total_gt': 0,
            'total_pred': 0,
            'total_correct': 0,
            'class_stats': defaultdict(lambda: {
                'gt': 0, 
                'pred': 0, 
                'correct': 0,
                'y_true': [],
                'y_scores': []
            })
        }
        
        image_paths = glob(os.path.join(self.image_folder, "*.[jJ][pP][gG]")) + \
                     glob(os.path.join(self.image_folder, "*.[pP][nN][gG]")) + \
                     glob(os.path.join(self.image_folder, "*.[jJ][pP][eE][gG]"))
        
        if not image_paths:
            raise ValueError(f"No images found in {self.image_folder}")
        
        for image_path in tqdm(image_paths, desc="Processing images"):
            image_name = os.path.basename(image_path)
            try:
                results = self.model(image_path)[0]
                gt_list = self.ground_truths.get(image_name, [])
                matched_gt = set()
                metrics['total_gt'] += len(gt_list)
                
                for gt in gt_list:
                    metrics['class_stats'][gt['class']]['gt'] += 1
                
                for box, score, cls_id in zip(results.boxes.xyxy.cpu().numpy(),
                                            results.boxes.conf.cpu().numpy(),
                                            results.boxes.cls.cpu().numpy().astype(int)):
                    pred_class = self.class_names[cls_id]
                    pred_bbox = list(map(float, box))
                    
                    pred_records.append({
                        'filename': image_name,
                        'class': pred_class,
                        'x1': pred_bbox[0],
                        'y1': pred_bbox[1],
                        'x2': pred_bbox[2],
                        'y2': pred_bbox[3],
                        'confidence': float(score)
                    })
                    
                    matched = False
                    for j, gt in enumerate(gt_list):
                        if gt['class'] == pred_class and j not in matched_gt:
                            iou = self._compute_iou(pred_bbox, gt['bbox'])
                            if iou >= 0.5:
                                metrics['total_correct'] += 1
                                metrics['class_stats'][pred_class]['correct'] += 1
                                matched_gt.add(j)
                                matched = True
                                break
                    
                    metrics['total_pred'] += 1
                    metrics['class_stats'][pred_class]['pred'] += 1
                    metrics['class_stats'][pred_class]['y_true'].append(1 if matched else 0)
                    metrics['class_stats'][pred_class]['y_scores'].append(float(score))
            
            except Exception as e:
                print(f"Error processing {image_name}: {str(e)}")
                continue
        
        return metrics, pred_records

    def calculate_metrics(self, metrics):
        precision = metrics['total_correct'] / metrics['total_pred'] if metrics['total_pred'] else 0
        recall = metrics['total_correct'] / metrics['total_gt'] if metrics['total_gt'] else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
        
        class_metrics = {}
        ap_scores = []
        
        for cls, stats in metrics['class_stats'].items():
            cls_precision = stats['correct'] / stats['pred'] if stats['pred'] else 0
            cls_recall = stats['correct'] / stats['gt'] if stats['gt'] else 0
            cls_f1 = 2 * (cls_precision * cls_recall) / (cls_precision + cls_recall) if (cls_precision + cls_recall) else 0
            cls_ap = self._calculate_ap(stats['y_true'], stats['y_scores'])
            
            class_metrics[cls] = {
                'precision': cls_precision,
                'recall': cls_recall,
                'f1': cls_f1,
                'ap': cls_ap
            }
            ap_scores.append(cls_ap)
        
        map_score = np.mean(ap_scores) if ap_scores else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'map': map_score,
            'class_metrics': class_metrics
        }

def main(model_path, images_path, labels_path):
    output_pred = "./predictions.csv"
    output_metrics = "./metrics.json"

    try:
        evaluator = FoodDetectionEvaluator(model_path, images_path, labels_path)
        metrics, pred_records = evaluator.evaluate()
        results = evaluator.calculate_metrics(metrics)
        
        pd.DataFrame(pred_records).to_csv(output_pred, index=False)
        with open(output_metrics, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"✅ Predictions saved to {output_pred}")
        print(f"✅ Metrics saved to {output_metrics}")
        
        print("\n=== Class-wise Metrics ===")
        for cls, metrics in results['class_metrics'].items():
            print(f"{cls}: AP={metrics['ap']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
        
        print("\n=== Evaluation Metrics ===")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1']:.4f}")
        print(f"mAP@0.5: {results['map']:.4f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

model_path = "./models/rasoi_yolov8s.pt"
images_path = "./validation_dataset/val_images"
labels_path = "./validation_dataset/val_labels.xlsx"
main(model_path, images_path, labels_path)
