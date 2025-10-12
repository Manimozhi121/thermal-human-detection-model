# Dataset Specification and Preparation Pipeline

This document outlines the data sourcing, preprocessing, and augmentation pipeline for the thermal human detection model. The methodology ensures reproducibility, data integrity, and model robustness for the target surveillance application.

---

## 1. Dataset Specification

### Source
The project utilizes the **Final FLIR Dataset**, accessed via a pre-processed version hosted on [Roboflow Universe](https://universe.roboflow.com/thermaldetection-dyazw/final-flir-dataset-hisfz-vfsor/1). Utilizing a version-controlled, pre-vetted dataset provides a stable baseline and mitigates risks associated with manual data parsing and cleaning.

### Class Distribution
The dataset has been explicitly filtered to **two classes: `person` and `car`**. This focuses the model's learning objective entirely on human-focused detection, eliminating confounding variables from irrelevant classes present in multi-class dataset.

### Data Splits
The dataset is partitioned into three standard splits to ensure robust training and unbiased evaluation:
* **Training Set:** Used for model weight optimization.
* **Validation Set:** Used for hyperparameter tuning and selecting the best model epoch (`best.pt`).
* **Test Set:** A held-out set for final, unbiased performance reporting.

The splitting strategy employed by the Roboflow platform helps mitigate data leakage by attempting to group images from the same video sequence into the same split, which is critical for datasets derived from video.

---

## 2. Preprocessing & Augmentation Pipeline

A deterministic preprocessing and augmentation pipeline was applied to the training split to enhance model generalization.

### Preprocessing
A single, non-destructive preprocessing step is applied to all images across all splits:
* **Resize (with Letterboxing):** All input images are resized to a fixed `640x640` dimension. This is a mandatory input requirement for the YOLOv8 CNN backbone. **Letterboxing** is used to pad the shorter dimension, which critically maintains the original aspect ratio. This prevents geometric distortion of human figures, ensuring the integrity of learned features.

### Augmentation Strategy
The following augmentations are applied on-the-fly to the training set to simulate real-world surveillance conditions and increase data variance.

| Augmentation | Parameters | Rationale |
| :--- | :--- | :--- |
| **Random Brightness/Contrast**| `±15%` | Simulates variations in thermal sensor calibration and ambient temperature conditions (e.g., day vs. night). |
| **Motion Blur** | `Kernel size upto 2.5px` | Simulates motion artifacts from a moving camera platform (e.g., drone, vehicle) or a fast-moving target. |
| **Gaussian Noise** | `Upto 0.1% of pixels`| Simulates electronic noise inherent in thermal imaging sensors and atmospheric interference. |
| **Flip** | `Horizontal` | Simulates non-ideal camera angles and slight perspective shifts common in dynamic surveillance scenarios. |

---

## 3. Data Acquisition Workflow

The entire data preparation process is encapsulated within the `prepare_data.py` script for full reproducibility.

### Prerequisites
1.  **Python Environment:** A working Python 3.8+ environment.
2.  **Dependencies:** All required libraries must be installed via `pip install -r requirements.txt`.
3.  **API Key:** A valid Roboflow API key must be provided. For security, this is managed via a `.env` file in the project root.

### Execution
1.  **Create `.env` file:** In the project root, create a file named `.env` and add your key:
    ```
    ROBOFLOW_API_KEY="YOUR_API_KEY_HERE"
    ```
2.  **Run the script:** From the project's root directory, execute the following command:
    ```bash
    python data/prepare_data.py
    ```

The script will authenticate with Roboflow, download the specified dataset version, and extract it into a new directory. This output folder is correctly structured for immediate use with the `train.py` script.