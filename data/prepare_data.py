# File: data/prepare_data.py
import os
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
WORKSPACE = "thermaldetection-dyazw"
PROJECT = "final-flir-dataset-hisfz-vfsor"
VERSION = 1

def download_dataset():
    """Downloads and prepares the thermal dataset from Roboflow."""
    print("--- Starting Data Preparation ---")
    if not ROBOFLOW_API_KEY:
        print("ERROR: Roboflow API key not found. Please create a .env file and add your key.")
        return

    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(WORKSPACE).project(PROJECT)
        dataset = project.version(VERSION).download("yolov8")
        print(f"\nDataset downloaded successfully to: {dataset.location}")
        print(f"The YAML file for training is at: {os.path.join(dataset.location, 'data.yaml')}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_dataset()