"""Unit tests for the metric, the post-processing and the RLE helpers (run with: pytest -q)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import compute_pq, pooled_pq, pq_between_csv                  # noqa: E402
from src.postprocessing import extract_instances                                # noqa: E402
from src.rle import decode_rle, encode_mask                                     # noqa: E402


def box(y0, y1, x0, x1, hw=64):
    m = np.zeros((hw, hw), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_metric_rules():
    g = box(10, 20, 10, 30)
    assert abs(compute_pq([g], [g.copy()])["PQ"] - 1.0) < 1e-9                  # perfect
    r = compute_pq([g], [box(10, 20, 10, 20)])                                  # IoU == 0.5 is NOT a match
    assert (r["TP"], r["FP"], r["FN"]) == (0, 1, 1)
    r = compute_pq([g], [box(10, 20, 10, 24)])                                  # IoU 0.7 is a match
    assert (r["TP"], r["FP"], r["FN"]) == (1, 0, 0)
    r = compute_pq([g], [box(10, 20, 10, 27), box(10, 20, 12, 30)])             # one GT, two predictions: both match
    assert (r["TP"], r["FP"], r["FN"]) == (2, 0, 0)
    r = compute_pq([box(10, 20, 10, 30), box(10, 20, 8, 32)], [box(10, 20, 9, 31)])   # two GT, one prediction: both match
    assert (r["TP"], r["FP"], r["FN"]) == (2, 0, 0)
    assert compute_pq([g], [])["FN"] == 1 and compute_pq([], [g])["FP"] == 1


def test_pooling_counts_every_record():
    g = box(10, 20, 10, 30)
    a, b = compute_pq([g], [g]), compute_pq([box(40, 50, 40, 50)], [g])       # one record matches, one does not
    p = pooled_pq([a, b])
    assert (p["TP"], p["FP"], p["FN"]) == (1, 1, 1) and abs(p["PQ"] - 1.0 / 2.0) < 1e-9


def test_postprocessing_closing_and_size_filter():
    prob = np.zeros((2048, 2048), np.float32)
    prob[100:130, 100:600] = 0.9
    prob[100:130, 615:900] = 0.9            # fragment 15 px away -> rejoined by the closing (radius 10)
    prob[500:510, 500:520] = 0.9            # 200 px -> removed
    prob[1500:1520, 300:900] = 0.9          # one clean instance
    inst = extract_instances(prob, 0.5, 10, 400)
    assert len(inst) == 2
    assert len(extract_instances(prob, 0.5, 0, 400)) == 3                       # without closing the two fragments stay separate


def test_rle_roundtrip_and_csv_agreement(tmp_path):
    m = np.zeros((2048, 2048), bool)
    m[100:130, 400:900] = True
    assert (decode_rle(encode_mask(m)) == m).all()
    df = pd.DataFrame([{"filament_id": "imgA_0", "segmentation_rle": encode_mask(m)}])
    df.to_csv(tmp_path / "a.csv", index=False)
    df.to_csv(tmp_path / "b.csv", index=False)
    assert pq_between_csv(tmp_path / "a.csv", tmp_path / "b.csv")[0] == 1.0
