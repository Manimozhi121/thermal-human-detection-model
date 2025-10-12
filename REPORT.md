# Technical Report: Thermal Human Detection for Military Surveillance
**Date:** October 12, 2025
**Author(s):** bit happens

---

## 1. Executive Summary

The objective of this project was to develop a lightweight, real-time object detection pipeline for thermal imagery, specifically targeting deployment on embedded surveillance platforms. 

We successfully fine-tuned a pre-trained YOLOv8-nano model on a specialized thermal dataset, creating a robust detector that is both highly accurate and computationally efficient.

The final model achieved a **mean Average Precision (mAP@0.5) of 0.862**, significantly exceeding the project's stretch goal of 0.70. Furthermore, the model was successfully exported to the ONNX format with a file size of **11.7 MB**, and performance estimates indicate a throughput of **~135 FPS** using TensorRT, satisfying all edge readiness criteria. This report details the methodology, results, and analysis of the developed solution.

---

## 2. Datasets Used and Preprocessing

A robust data pipeline was established to ensure model generalization and resilience to real-world surveillance conditions.

### 2.1. Dataset Selection
The primary dataset used was the **"Final FLIR Dataset"**, accessed via a pre-processed version hosted on [Roboflow Universe](https://universe.roboflow.com/thermaldetection-dyazw/final-flir-dataset-hisfz-vfsor/1). This specific version was selected as it was pre-filtered to two key classes of interest (`person` and `car`), providing a clean, high-quality foundation that allowed the model to specialize on relevant surveillance targets. The dataset provided well-defined training, validation, and test splits, which is critical for preventing data leakage and ensuring an unbiased evaluation.

### 2.2. Preprocessing and Augmentation Pipeline
All images were resized to **640x640 pixels** using letterboxing to maintain the original aspect ratio, preventing geometric distortion of objects. The training split was subjected to a series of augmentations designed to simulate dynamic surveillance environments.

| Augmentation | Parameters | Rationale |
| :--- | :--- | :--- |
| **Random Brightness/Contrast**| `±15%` | Simulates variations in thermal sensor calibration and ambient temperature conditions. |
| **Motion Blur** | `Kernel size ≤ 2.5px` | Simulates motion artifacts from a moving camera platform or a fast-moving target. |
| **Gaussian Noise** | `≤ 0.1% of pixels`| Simulates electronic noise inherent in thermal imaging sensors. |
| **Horizontal Flip** | `50% probability` | Increases dataset variability and teaches the model left-right object invariance. |

---

## 3. Model Choice and Training

The model architecture and training methodology were selected to prioritize speed and efficiency without compromising accuracy.

### 3.1. Model Architecture
We selected **YOLOv8-nano (`yolov8n.pt`)** as the base model. This choice was driven by several factors:
* **State-of-the-Art Performance:** YOLOv8's specialty lies in its versatility as a multi-task vision model that delivers accuracy and speed across a broad range of tasks, including object detection, instance segmentation, and pose estimation.
* **Lightweight Design:** The "nano" variant is specifically engineered for high-speed inference on resource-constrained edge devices.
* **Anchor-Free Architecture:** Its modern anchor-free design simplifies the detection head, often leading to faster inference and better performance.

### 3.2. Training Methodology
The model was fine-tuned using a transfer learning approach. Training was performed for **20 epochs** on a Google Colab instance equipped with an NVIDIA T4 GPU. **FP16 mixed-precision** training was utilized automatically by the Ultralytics framework to accelerate the process. The model with the best validation mAP score was saved as the final `best.pt` artifact.

---

## 4. Results and Evaluation

The model's performance was evaluated quantitatively and qualitatively against the project's quality bar.

### 4.1. Quantitative Results
The final model achieved the following performance on the held-out validation set:

| Metric | Final Score | Project Goal | Status |
| :--- | :--- | :--- | :--- |
| **mAP@0.5** | **0.862** | ≥ 0.70 (Stretch) | **Stretch Goal Met** |
| **Precision** | 0.870 | - | - |
| **Recall** | 0.759 | - | - |

The primary metric, mAP@0.5, significantly surpassed the stretch goal, indicating a high level of accuracy.

### 4.2. Performance Curves
The training logs confirm a healthy and stable learning process. The model's loss consistently decreased on both training and validation sets, indicating good generalization without overfitting. The Precision-Recall curve demonstrates the model's ability to maintain high precision across various recall thresholds, a hallmark of a robust detector.

---

## 5. Failure Case Analysis

While the model performed exceptionally well, no model is perfect. Analysis of failure cases provides insight into its limitations and potential areas for improvement.


### Case 1: Small Scale / Low Resolution

The model failed to detect a person at a significant distance. At this range, the target occupies only a few pixels. The features are too indistinct for the model to confidently classify as a person, likely falling below its confidence threshold.

### Case 2: Partial Occlusion

The model requires a significant portion of the target's key features to be visible. Here, more than half of the person is occluded by another object, preventing a confident detection. The model has not learned to infer a full person from a small visible part.

### Case 3: Thermal Crossover / Low Contrast

This is a fundamental challenge for thermal imaging. During a "thermal crossover" event, the target's temperature can match the ambient background temperature. This results in a very low thermal contrast, effectively camouflaging the person from the sensor and, therefore, the model. 

---

## 6. Future Scope

In the future, with sufficient time and resources, the following improvements could be explored to enhance the model's capabilities:

1.  **Data Enrichment:** Incorporate a more diverse dataset, including imagery from different climates and seasons, to improve robustness against environmental variations like thermal crossover. Adding a dedicated `animal` class would help reduce false positives in rural surveillance scenarios, distinguishing them from people and vehicles.
2.  **Model Quantization:** Perform **INT8 quantization** on the exported ONNX model using TensorRT's calibration tools. This could further increase inference speed by 1.5-2x with a minimal drop in accuracy, making it even more suitable for low-power edge devices.
3.  **Implement Temporal Smoothing:** For video inference, move beyond per-frame detection and implement a lightweight object tracking algorithm (e.g., BoT-SORT). This would provide temporal consistency, reduce flickering detections, and allow the model to maintain a track on an object even if it is briefly occluded.
