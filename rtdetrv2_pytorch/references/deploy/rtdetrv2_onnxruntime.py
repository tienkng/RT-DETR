"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time
import torch
import torchvision.transforms as T

import onnxruntime as ort
from PIL import Image, ImageDraw
from util import remove_high_overlap_bboxes

COLOR = {0: "green", 1: "blue", 2: "red"}


def draw(images, labels, boxes, scores, thrh=0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        remove_idx = remove_high_overlap_bboxes(box, lab, scrs, threshold=0.8)

        # Remove high overlap bboxes
        box = [b for i, b in enumerate(box) if i not in remove_idx]
        lab = [l for i, l in enumerate(lab) if i not in remove_idx]
        scrs = [s for i, s in enumerate(scrs) if i not in remove_idx]

        for j, b in enumerate(box):
            # Calculate thickness based on image size
            thickness = max(1, min(im.size) // 200)

            for t in range(thickness):
                draw.rectangle(
                    [b[0] - t, b[1] - t, b[2] + t, b[3] + t],
                    outline=COLOR[lab[j].item()],
                )
            draw.text(
                (b[0], b[1]),
                text=f"{round(scrs[j].item(), 2)}",
                fill=COLOR[lab[j].item()],
            )

        im.save(f"results_{i}.jpg")


def main(
    args,
):
    """main"""

    if args.device == "cuda":
        providers = [
            ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CPUExecutionProvider"]
    print("Model:", args.onnx_file)
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(args.onnx_file, sess_options, providers)

    # Warmup
    for _ in range(10):
        _ = sess.run(
            output_names=None,
            input_feed={
                "images": torch.randn(1, 3, 640, 640).data.numpy(),
                "orig_target_sizes": torch.tensor([[640, 640]]).data.numpy(),
            },
        )

    im_pil = Image.open(args.im_file).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None]

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )
    im_data = transforms(im_pil)[None]
    start = time.time()
    output = sess.run(
        output_names=None,
        input_feed={
            "images": im_data.data.numpy(),
            "orig_target_sizes": orig_size.data.numpy(),
        },
    )
    print("Inference time:", time.time() - start)
    labels, boxes, scores = output

    draw([im_pil], labels, boxes, scores, thrh=args.threshold)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx-file",
        type=str,
    )
    parser.add_argument(
        "--im-file",
        type=str,
    )
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("--threshold", type=float, default=0.68)
    args = parser.parse_args()
    main(args)
