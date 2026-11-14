"""Run-length encoding of binary masks in the competition format (COCO compressed RLE, Fortran order)."""
import numpy as np
from pycocotools import mask as mask_util


def encode_mask(mask: np.ndarray) -> str:
    """Binary (H, W) mask -> compressed RLE 'counts' string (the value of the `segmentation_rle` column)."""
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("utf-8")


def decode_rle(counts: str, shape=(2048, 2048)) -> np.ndarray:
    """Compressed RLE 'counts' string -> boolean (H, W) mask."""
    return mask_util.decode({"size": list(shape), "counts": counts.encode("utf-8")}).astype(bool)
