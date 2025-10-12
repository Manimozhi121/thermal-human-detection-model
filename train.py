# File: train.py
# Usage: python train.py --data path/to/data.yaml --epochs 30
# This script fine-tunes a pretrained YOLOv8n model on the thermal dataset.

import argparse
from ultralytics import YOLO

def main(args):
    """Fine-tunes a YOLOv8n model on the thermal dataset."""
    print("Starting Model Training")    
    model = YOLO('yolov8n.pt')
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=640,
        project='models',
        name='training_run',
        exist_ok=True  
    )
    print("\nTraining complete.")
    print("Best model weights saved in 'models/training_run/weights/best.pt'")
    print("Please copy the 'best.pt' file to 'models/best.pt' for use with other scripts.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8n on a custom dataset.")
    parser.add_argument('--data', type=str, required=True, help="Path to the data.yaml configuration file.")
    parser.add_argument('--epochs', type=int, default=30, help="Number of training epochs.")
    args = parser.parse_args()
    main(args)