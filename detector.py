from ultralytics import YOLO
import cv2

class WeaponDetector:
    def __init__(self):
        # Load the pre-trained YOLOv8 model
        self.model = YOLO('yolov8n.pt')
        # Define weapon classes (from COCO dataset)
        self.weapon_classes = [43, 44, 45, 46]  # knife, scissors, etc.
        
    def detect(self, frame):
        results = self.model(frame, verbose=False)
        weapons = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls in self.weapon_classes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    weapons.append({
                        'class': cls,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
                    
        return weapons
