"""Inference: image -> flip-TTA probability map -> instances -> submission.csv.

Command line:
    python -m src.inference --weights weights/final_model.pth --test-dir <path to test_images> --out submission.csv
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .dataset import load_image, preprocess
from .model import ResNetUNet, load_weights
from .postprocessing import extract_instances
from .rle import encode_mask

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


@torch.no_grad()
def predict_probability(model, image_path, device, resolution: int = 1024, tta: bool = True) -> np.ndarray:
    """Probability map at the original image resolution; with `tta` the image and its h/v/hv flips are averaged."""
    img = load_image(image_path)
    h, w = img.shape
    x = torch.from_numpy(preprocess(img, resolution))[None, None].to(device)
    views = [None, [3], [2], [2, 3]] if tta else [None]
    outs = []
    for dims in views:
        xi = x if dims is None else torch.flip(x, dims)
        p = torch.sigmoid(model(xi))
        if dims is not None:
            p = torch.flip(p, dims)
        outs.append(p[0, 0].float().cpu().numpy())
    return cv2.resize(np.mean(outs, axis=0), (w, h), interpolation=cv2.INTER_LINEAR)


def run(weights, test_dir, out_csv, encoder="resnet50", threshold=0.5, closing_radius=10, min_pixels=400,
        resolution=1024, tta=True, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_weights(ResNetUNet(encoder=encoder, pretrained=False), weights, device)
    paths = sorted(p for p in Path(test_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    rows = []
    for path in tqdm(paths, desc="inference"):
        prob = predict_probability(model, path, device, resolution, tta)
        for k, mask in enumerate(extract_instances(prob, threshold, closing_radius, min_pixels)):
            rows.append({"filament_id": f"{path.stem}_{k}", "segmentation_rle": encode_mask(mask)})
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    df.to_csv(out_csv, index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="weights/final_model.pth")
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--encoder", default="resnet50")
    args = ap.parse_args()
    df = run(args.weights, args.test_dir, args.out, encoder=args.encoder)
    print(f"wrote {args.out}: {len(df)} filaments over {df.filament_id.str.split('_', n=1).str[0].nunique()} images")


if __name__ == "__main__":
    main()
