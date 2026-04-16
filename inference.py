"""
Inference Pipeline for Bangladeshi Taka Note Detection
Loads trained YOLOv8 model and runs detection on a single image.

Usage:
    python inference.py <image_path>
    python inference.py test_images/sample1.jpg
"""

import sys
import os
from ultralytics import YOLO

# path to trained model weights
MODEL_PATH = "best.pt"


def load_model(weights_path):
    """Load the trained YOLOv8 model."""
    if not os.path.exists(weights_path):
        print(f"Error: Model weights not found at '{weights_path}'")
        sys.exit(1)
    model = YOLO(weights_path)
    print(f"Model loaded from: {weights_path}")
    return model


def run_inference(model, image_path, confidence=0.25):
    """Run detection on a single image and return results."""
    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'")
        sys.exit(1)

    results = model.predict(source=image_path, conf=confidence, verbose=False)[0]
    return results


def print_results(results, image_path):
    """Print detected classes, confidence scores, and bounding boxes."""
    print(f"\nImage: {os.path.basename(image_path)}")
    print("=" * 60)

    boxes = results.boxes
    if len(boxes) == 0:
        print("No Taka notes detected.")
        return

    print(f"Found {len(boxes)} detection(s):\n")
    for i, box in enumerate(boxes):
        class_id = int(box.cls[0])
        class_name = results.names[class_id]
        confidence = float(box.conf[0])
        bbox = [int(x) for x in box.xyxy[0].tolist()]

        print(f"  Detection {i + 1}:")
        print(f"    Class:      {class_name}")
        print(f"    Confidence: {confidence:.2%}")
        print(f"    Bbox:       {bbox}  (x1, y1, x2, y2)")
        print()


def save_result_image(results, image_path):
    """Save the image with bounding boxes drawn on it."""
    output_name = f"result_{os.path.basename(image_path)}"
    annotated = results.plot()

    import cv2
    cv2.imwrite(output_name, annotated)
    print(f"Result image saved: {output_name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        print("Example: python inference.py test_images/sample1.jpg")
        sys.exit(1)

    image_path = sys.argv[1]

    # Step 1: Load model
    model = load_model(MODEL_PATH)

    # Step 2: Run inference
    results = run_inference(model, image_path)

    # Step 3: Print results
    print_results(results, image_path)

    # Step 4: Save annotated image
    save_result_image(results, image_path)
