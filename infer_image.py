# File: infer_image.py
# Usage: python infer_image.py --source path/to/image.jpg
# This script runs inference on a single image using the trained model.

import argparse
from ultralytics import YOLO

def main(args):
    """Runs inference on a single image."""
    model = YOLO(args.model)
    model.predict(
        source=args.source,
        save=True,
        show=args.view,
        project='artifacts',
        name='image_inference_results',
        exist_ok=True
    )
    print("\nInference complete. Annotated image saved in 'artifacts/image_inference_results'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on a single image.")
    parser.add_argument('--model', type=str, default='models/best.pt', help="Path to the trained .pt model file.")
    parser.add_argument('--source', type=str, required=True, help="Path to the source image.")
    parser.add_argument('--view', action='store_true', help="Display the image after inference.")
    args = parser.parse_args()
    main(args)