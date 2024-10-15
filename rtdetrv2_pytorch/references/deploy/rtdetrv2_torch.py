"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time
import os

os.sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
import torch
import torch.nn as nn
import torchvision.transforms as T

from PIL import Image, ImageDraw

from src.core import YAMLConfig
from util import remove_high_overlap_bboxes

COLOR = {0: "green", 1: "blue", 2: "red"}


def draw(images, labels, boxes, scores, thrh=0.68):
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


@torch.no_grad()
def main(
    args,
):
    """main"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(
            self,
        ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)
    model.eval()

    print("Model:", args.resume)

    # Warmup
    for _ in range(10):
        _ = model(
            torch.randn(1, 3, 640, 640).to(args.device),
            torch.tensor([[640, 640]]).to(args.device),
        )

    im_pil = Image.open(args.im_file).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )
    im_data = transforms(im_pil)[None].to(args.device)

    start = time.time()
    output = model(im_data, orig_size)
    print("Inference time:", time.time() - start)
    labels, boxes, scores = output

    draw([im_pil], labels, boxes, scores, thrh=args.threshold)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--im-file",
        type=str,
    )
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("--threshold", type=float, default=0.68)
    args = parser.parse_args()

    main(args)
