# Performance Benchmarks

This report details the accuracy, model size, and estimated edge performance of the fine-tuned YOLOv8n model.

## 1. Accuracy

| Metric  | Score      | Goal    | Status |
|---------|------------|---------|--------|
| mAP@0.5 | **0.862** | >= 0.70 | Stretch Goal Met |

**Source:** Final epoch of the training run logs (`results.csv`), `metrics/mAP50(B)` column.

---

## 2. Edge Readiness

| Metric          | Value        | Goal      | Status |
|-----------------|--------------|-----------|--------|
| Model Size (ONNX) | **11.6 MB** | <= 25 MB  | Pass  |
| Speed (TensorRT Est.) | **~135 FPS** | >= 25 FPS | Stretch Goal Met |

### Interpretation of Edge Readiness

* **Model Size:** The model was successfully exported to the ONNX format. The final file size is **11.6 MB**, which is less than half of the **≤ 25 MB** requirement, making it highly suitable for deployment on devices with limited storage. The source for this value is the output log from the `export_onnx.py` script.

* **Speed Estimate:** The speed was estimated using official performance data from Ultralytics, the creators of YOLOv8. The benchmark for a **YOLOv8n** model using **FP16 precision** on a representative embedded platform (**NVIDIA Jetson AGX Orin**) reports a latency of **7.4 ms**. This translates to a throughput of approximately **135 FPS** (`1000 ms / 7.4 ms`). This value significantly exceeds the **≥ 25 FPS** stretch goal, confirming the model's capability for real-time inference.