# RT-DETRv2 on Gender Classification

## Install

- Follow the instructions on the [official website](https://pytorch.org/get-started/locally/)
```bash
pip install -r requirements.txt
pip install onnxruntime
```

- For TensorRT: [README_TRT_INSTALL.md](README_TRT_INSTALL.md)

## Convert model TensorRT: 
- Follow the instructions [README_TRT.md](README_TRT.md)

## Inference
- Pytorch
```bash
python references/deploy/rtdetrv2_torch.py -c <CONFIG> -r <PT_FILE> --image-file <IMAGE_FILE> --threshold <THRESHOLD> --d <DEVICE>
```

| Config | PT File |
| --- | --- |
| configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml | best_S.pth |
| configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml | best_M.pth |
| configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml | best_X.pth |


- Example:
    ```bash
    python references/deploy/rtdetrv2_torch.py -c configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml -r best_S.pth --image-file <IMAGE_FILE> --device cuda
    ```

- ONNX
```bash
python inference_onnx.py --onnx-file <ONNX_FILE> --image-file <IMAGE_FILE> -d <DEVICE> --threshold <THRESHOLD>
```

- Example:
    ```bash
    python inference_onnx.py --onnx-file best_S.onnx --image-file <IMAGE_FILE> -d cuda
    ```

- TensorRT
```bash
python inference_trt.py --trt <TRT_FILE> --image-file <IMAGE_FILE> --threshold <THRESHOLD>
```

- Example:
    ```bash
    python inference_trt.py --trt best_S.trt --image-file <IMAGE_FILE>
    ```

| Argument | Description |
| --- | --- |
| -c, --config | Path to the configuration file |
| -r, --resume | Path to the checkpoint file |
| -d, --device | Device to use for inference (cpu, cuda) |
| --onnx-file | Path to the ONNX file |
| --trt | Path to the TensorRT file |
| --image-file | Path to the image file |
| --threshold | Confidence threshold for detections |
