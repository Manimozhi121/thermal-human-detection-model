# File: infer_folder.py
# Usage: python infer_folder.py --source path/to/folder --to_video 25
# Runs inference on a folder of images and optionally stitches them into a video.

import os
import cv2
import argparse
from ultralytics import YOLO

def main(args):
    """Runs inference on a folder and optionally creates a video."""
    model = YOLO(args.model)
    
    # Directory to save annotated frames
    output_dir = 'artifacts/folder_inference_results'
    model.predict(
        source=args.source,
        save=True,
        project='artifacts',
        name='folder_inference_results',
        exist_ok=True
    )
    print(f"\nInference complete. Annotated images saved in '{output_dir}'.")

    if args.to_video:
        print(f"\n--- Stitching images into video at {args.to_video} FPS ---")
        subdirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith('predict')]
        if not subdirs:
            print("Could not find prediction output directory.")
            return
            
        latest_subdir = max(subdirs, key=os.path.getmtime)
        print(f"Reading frames from: {latest_subdir}")

        images = [img for img in os.listdir(latest_subdir) if img.endswith((".jpg", ".jpeg", ".png"))]
        images.sort() 
        
        if not images:
            print("No images found in the output folder to create a video.")
            return

        frame = cv2.imread(os.path.join(latest_subdir, images[0]))
        height, width, layers = frame.shape
        
        video_path = os.path.join('artifacts', 'output_video.mp4')
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), args.to_video, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(latest_subdir, image)))
        
        cv2.destroyAllWindows()
        video.release()
        print(f"Video saved successfully to '{video_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference on a folder.")
    parser.add_argument('--model', type=str, default='models/best.pt', help="Path to the trained .pt model file.")
    parser.add_argument('--source', type=str, required=True, help="Path to the folder of source images.")
    parser.add_argument('--to_video', type=int, help="Optional: FPS to stitch output images into an MP4 video.")
    args = parser.parse_args()
    main(args)