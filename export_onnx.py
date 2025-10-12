# File: export_onnx.py
# Usage: python export_onnx.py
# Exports the trained PyTorch model to ONNX format for deployment.

import argparse
from ultralytics import YOLO

def main(args):
    """Exports a PyTorch model to ONNX format."""
    print("Exporting model to ONNX")
    model = YOLO(args.model)
    model.export(format='onnx')
    print("\nExport complete. ONNX model saved in the 'models/' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to ONNX.")
    parser.add_argument('--model', type=str, default='models/best.pt', help="Path to the trained .pt model file.")
    args = parser.parse_args()
    main(args)