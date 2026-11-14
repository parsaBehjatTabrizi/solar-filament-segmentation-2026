"""Data utilities: image loading, COCO-style annotation parsing, the date-grouped split and the training dataset.

Training target = union of the polygons of ONE annotator record; with `multi_record=True` a record is drawn at random among the
image's records every time the sample is requested (the competition scores every annotator record separately).
"""
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

FILE_NAME_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})\d{6}([A-Za-z])h\.", re.I)   # YYYYMMDDHHMMSS + station letter + 'h'


def load_image(path) -> np.ndarray:
    """Grayscale float32 image (RGB images are averaged)."""
    img = np.array(Image.open(path))
    if img.ndim == 3:
        img = img[..., :3].mean(axis=2)
    return img.astype(np.float32)


def preprocess(img: np.ndarray, resolution: int = 1024) -> np.ndarray:
    """Resize (area interpolation) and normalise by the image's own 1st-99th percentiles to [0, 1]."""
    small = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
    lo, hi = np.percentile(small, [1, 99])
    return np.clip((small - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def load_annotations(json_path):
    """Return (images_info by record id, annotations by record id, record ids by image file name)."""
    with open(json_path) as f:
        coco = json.load(f)
    images_info = {im["id"]: im for im in coco["images"]}
    anns_by_record = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_record[a["image_id"]].append(a)
    file_to_records = defaultdict(list)
    for rid, info in images_info.items():
        file_to_records[Path(info["file_name"]).name].append(rid)
    return images_info, anns_by_record, {k: sorted(v) for k, v in file_to_records.items()}


def ann_to_mask(ann, height: int, width: int) -> np.ndarray:
    m = np.zeros((height, width), np.uint8)
    for poly in ann["segmentation"]:
        pts = np.asarray(poly, np.float64).reshape(-1, 2)
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m.astype(bool)


def record_masks(record_id, images_info, anns_by_record):
    """Full-resolution instance masks of one annotator record (empty masks dropped; raises if nothing is left)."""
    info = images_info[record_id]
    masks = [ann_to_mask(a, info["height"], info["width"]) for a in anns_by_record[record_id]]
    masks = [m for m in masks if m.any()]
    if not masks:
        raise ValueError(f"record {record_id} has no non-empty masks")
    return masks


def create_split(primary_ids, images_info, val_frac: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Date-grouped train/validation split: whole observation dates go to validation, so no date is in both parts."""
    rows = []
    for rid in primary_ids:
        fname = Path(images_info[rid]["file_name"]).name
        m = FILE_NAME_RE.match(fname)
        if not m:
            raise ValueError(f"unparseable file name: {fname}")
        rows.append({"img_id": rid, "file": fname, "date": f"{m[1]}-{m[2]}-{m[3]}", "station": m[4].upper()})
    df = pd.DataFrame(rows)
    dates = sorted(df["date"].unique())
    random.Random(seed).shuffle(dates)
    val_dates = set(dates[: int(round(len(dates) * val_frac))])
    df["split"] = df["date"].map(lambda d: "val" if d in val_dates else "train")
    return df


class FilamentDataset(Dataset):
    def __init__(self, primary_ids, images_info, anns_by_record, file_to_records, image_dir, resolution=1024,
                 augment=False, multi_record=False):
        self.ids, self.info, self.anns = list(primary_ids), images_info, anns_by_record
        self.f2r, self.image_dir = file_to_records, Path(image_dir)
        self.res, self.augment, self.multi = resolution, augment, multi_record

    def __len__(self):
        return len(self.ids)

    def _usable_records(self, fname):
        return [r for r in self.f2r[fname] if len(self.anns.get(r, [])) > 0]

    def __getitem__(self, idx):
        rid = self.ids[idx]
        fname = Path(self.info[rid]["file_name"]).name
        rec = random.choice(self._usable_records(fname) or [rid]) if (self.augment and self.multi) else rid
        img = load_image(self.image_dir / fname)
        h, w = img.shape
        gt = np.zeros((h, w), bool)
        for a in self.anns[rec]:
            gt |= ann_to_mask(a, h, w)
        img = preprocess(img, self.res)
        gt = cv2.resize(gt.astype(np.uint8), (self.res, self.res), interpolation=cv2.INTER_NEAREST).astype(bool)
        if self.augment:
            if random.random() < 0.5:
                img, gt = np.fliplr(img).copy(), np.fliplr(gt).copy()
            if random.random() < 0.5:
                img, gt = np.flipud(img).copy(), np.flipud(gt).copy()
        return {"image": torch.from_numpy(img)[None], "mask": torch.from_numpy(gt.astype(np.float32))[None], "img_id": rid}
