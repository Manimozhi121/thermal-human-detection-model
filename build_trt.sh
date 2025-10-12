#!/bin/bash
# File: build_trt.sh
# Convert YOLO ONNX model to TensorRT engine and benchmark inference speed.

ONNX_MODEL="models/best.onnx"
TRT_ENGINE="models/best.engine"
FP16_FLAG="--fp16"
BATCH_SIZE=1

if ! command -v trtexec &> /dev/null
then
    echo "trtexec could not be found. Please install TensorRT first."
    exit
fi

echo "Building TensorRT engine from ONNX model..."
trtexec \
    --onnx="$ONNX_MODEL" \
    --saveEngine="$TRT_ENGINE" \
    $FP16_FLAG \
    --workspace=4096 \
    --verbose \
    --batch=$BATCH_SIZE

echo "Benchmarking TensorRT engine..."
trtexec \
    --loadEngine="$TRT_ENGINE" \
    --batch=$BATCH_SIZE \
    --iterations=100 \
    --fp16

'''
# Make sure the script is executable
chmod +x build_trt.sh

# Run the script
./build_trt.sh
'''
