
import cv2
import torch
import numpy as np
import os

# --- Configuration ---
MODEL_PATH = "yolov5s.pt" # Path to pre-trained YOLOv5 model
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# --- Load YOLOv5 Model ---
def load_yolov5_model(path=MODEL_PATH):
    try:
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        print(f"YOLOv5 model loaded from {path}")
        return model
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        print("Attempting to download model...")
        # Fallback if local model not found or error
        model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
        return model

model = load_yolov5_model()

# --- Perform Object Detection ---
def detect_objects(image_path, model):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    # Inference
    results = model(img)

    # Process results
    detections = results.pandas().xyxy[0]
    
    # Filter by confidence
    detections = detections[detections["confidence"] > CONF_THRESHOLD]

    # Draw bounding boxes and labels
    for *xyxy, conf, cls, name in detections.values:
        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{name} {conf:.2f}"
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) # Random color
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img

# --- Main Execution ---
if __name__ == "__main__":
    # Create a dummy image for demonstration
    dummy_image_path = "dummy_image.jpg"
    if not os.path.exists(dummy_image_path):
        dummy_image = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.putText(dummy_image, "Hello YOLO!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.imwrite(dummy_image_path, dummy_image)
        print(f"Created dummy image: {dummy_image_path}")

    print(f"Performing object detection on {dummy_image_path}...")
    output_image = detect_objects(dummy_image_path, model)

    if output_image is not None:
        output_path = "detected_objects.jpg"
        cv2.imwrite(output_path, output_image)
        print(f"Detection results saved to {output_path}")
    else:
        print("Object detection failed.")

# Commit 1 marker: 2023-04-10 10:00:00
