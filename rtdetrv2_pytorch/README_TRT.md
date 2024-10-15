# How to convert model ONNX -> TensorRT
Best practice for converting ONNX model to TensorRT engine is to use the `trtexec` tool provided by NVIDIA. The following steps show how to convert an ONNX model to a TensorRT engine using the `trtexec` tool.

## 1. Pull Docker Image
```bash
docker pull nvcr.io/nvidia/tensorrt:24.02-py3
```

## 2. Run Docker Container
```bash
docker run -it --gpus all -v /path/to/your/model:/model --rm nvcr.io/nvidia/tensorrt:24.02-py3 bash
```

## 3. Convert ONNX to TensorRT
```bash
trtexec --onnx=your_model.onnx --saveEngine=your_model.trt --fp32
```