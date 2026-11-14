"""Competition metric: pooled panoptic quality with many-to-many matching (re-implementation of the organizers' logic).

Every (ground-truth, prediction) pair with IoU strictly greater than 0.5 is a true positive and contributes its IoU to the numerator;
a prediction without such a pair is a false positive, a ground-truth instance without such a pair is a false negative.
PQ = sum(IoU of true positives) / (TP + 0.5 FP + 0.5 FN).  The organizers score the predictions of an image against EVERY annotator
record separately and pool TP / FP / FN / IoU over all records; apply `pooled_pq` to the per-record results to reproduce that.
Verified against the organizers' own scoring code (agreement to 6 decimals on validation predictions).
"""
import numpy as np
import pandas as pd
from pycocotools import mask as mask_util


def _bbox(m):
    rows = np.flatnonzero(m.any(axis=1))
    cols = np.flatnonzero(m.any(axis=0))
    if len(rows) == 0:
        return (0, 0, 0, 0)
    return (rows[0], rows[-1] + 1, cols[0], cols[-1] + 1)


def iou_and_dice_matrix(pred_masks, gt_masks):
    """Pairwise IoU and Dice, shape (n_pred, n_gt). Pairs whose bounding boxes do not intersect are skipped (same result, faster)."""
    n_p, n_g = len(pred_masks), len(gt_masks)
    iou = np.zeros((n_p, n_g))
    dice = np.zeros((n_p, n_g))
    pb = [_bbox(p) for p in pred_masks]
    gb = [_bbox(g) for g in gt_masks]
    pa = [int(p.sum()) for p in pred_masks]
    ga = [int(g.sum()) for g in gt_masks]
    for i, p in enumerate(pred_masks):
        py0, py1, px0, px1 = pb[i]
        for j, g in enumerate(gt_masks):
            gy0, gy1, gx0, gx1 = gb[j]
            y0, y1 = max(py0, gy0), min(py1, gy1)
            x0, x1 = max(px0, gx0), min(px1, gx1)
            if y0 >= y1 or x0 >= x1:
                continue
            inter = int(np.logical_and(p[y0:y1, x0:x1], g[y0:y1, x0:x1]).sum())
            if inter == 0:
                continue
            iou[i, j] = inter / (pa[i] + ga[j] - inter)
            dice[i, j] = 2 * inter / (pa[i] + ga[j])
    return iou, dice


def compute_pq(gt_masks, pred_masks):
    """Statistics for ONE annotator record: TP, FP, FN, IoUs / Dices of the true-positive pairs and the entry's own PQ."""
    if len(gt_masks) == 0 and len(pred_masks) == 0:
        return {"PQ": 1.0, "TP": 0, "FP": 0, "FN": 0, "tp_ious": [], "tp_dices": []}
    iou, dice = iou_and_dice_matrix(pred_masks, gt_masks)
    hit = iou > 0.5                                   # strictly greater
    tp_ious, tp_dices = iou[hit].tolist(), dice[hit].tolist()
    tp = len(tp_ious)
    fp = int((hit.sum(axis=1) == 0).sum()) if len(pred_masks) else 0
    fn = int((hit.sum(axis=0) == 0).sum()) if len(gt_masks) else 0
    denom = tp + 0.5 * fp + 0.5 * fn
    return {"PQ": (sum(tp_ious) / denom) if denom > 0 else 0.0, "TP": tp, "FP": fp, "FN": fn,
            "tp_ious": tp_ious, "tp_dices": tp_dices}


def pooled_pq(results):
    """Pool per-record results (list of `compute_pq` outputs) into the competition's single PQ number."""
    tp = sum(r["TP"] for r in results)
    fp = sum(r["FP"] for r in results)
    fn = sum(r["FN"] for r in results)
    ious = [x for r in results for x in r["tp_ious"]]
    denom = tp + 0.5 * fp + 0.5 * fn
    return {"PQ": (sum(ious) / denom) if denom else 0.0, "TP": tp, "FP": fp, "FN": fn,
            "mean_tp_iou": float(np.mean(ious)) if ious else 0.0}


def pq_between_csv(csv_a, csv_b):
    """Agreement between two submission files: csv_a plays the ground truth, csv_b the predictions (1.0 = identical predictions)."""
    a, b = pd.read_csv(csv_a), pd.read_csv(csv_b)
    a["img"] = a.filament_id.str.split("_", n=1).str[0]
    b["img"] = b.filament_id.str.split("_", n=1).str[0]
    ga, gb = dict(tuple(a.groupby("img"))), dict(tuple(b.groupby("img")))
    tp_iou, fp, fn = [], 0, 0
    for img in sorted(set(ga) | set(gb)):
        ra = [{"size": [2048, 2048], "counts": c.encode()} for c in ga[img].segmentation_rle] if img in ga else []
        rb = [{"size": [2048, 2048], "counts": c.encode()} for c in gb[img].segmentation_rle] if img in gb else []
        if not ra:
            fp += len(rb)
            continue
        if not rb:
            fn += len(ra)
            continue
        iou = np.asarray(mask_util.iou(rb, ra, [0] * len(ra)))
        hit = iou > 0.5
        tp_iou += iou[hit].tolist()
        fp += int((hit.sum(1) == 0).sum())
        fn += int((hit.sum(0) == 0).sum())
    denom = len(tp_iou) + 0.5 * fp + 0.5 * fn
    return (sum(tp_iou) / denom if denom else 0.0), len(tp_iou), fp, fn
