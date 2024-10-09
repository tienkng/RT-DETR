"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os

os.sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
import torch
import torch.nn as nn
import torchvision.transforms as T

import numpy as np
from tqdm.auto import tqdm
from glob import glob
from PIL import Image, ImageDraw

from src.core import YAMLConfig
from util import remove_high_overlap_bboxes

COLOR = {0: "green", 1: "blue", 2: "red"}


def draw(images, labels, boxes, scores, img_name, thrh=0.68, output_folder="exp"):
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

        im.save(f"{output_folder}/{img_name}")


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

    os.makedirs(args.o, exist_ok=True)

    for img in tqdm(glob(f"{args.im_file}/*.jpg")):
        img_name = os.path.basename(img)
        im_pil = Image.open(img).convert("RGB")
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(args.device)

        transforms = T.Compose(
            [
                T.Resize((640, 640)),
                T.ToTensor(),
            ]
        )
        im_data = transforms(im_pil)[None].to(args.device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        draw([im_pil], labels, boxes, scores, output_folder=args.o, img_name=img_name)


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
    parser.add_argument(
        "-o",
        type=str,
    )
    parser.add_argument("-d", "--device", type=str, default="cpu")
    args = parser.parse_args(
        [
            "-c",
            # "configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml",  # S
            # "configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml",  # M
            "configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml",  # X
            "-r",
            # "./best_S.pth",
            # "./best_M.pth",
            "./best_X.pth",
            "-f",
            "test_outside",
            "-o",
            "exp/X",
            "--device",
            "cuda:0",
        ]
    )

    main(args)
