import json

import numpy as np
import pytest
import torch

from src.data.ap10k import AP10KDataset
from src.eval.metrics.fitted import PartAnnotations, distinctiveness, point_hits
from src.utils.visualization import display_map


def test_point_visibility_and_window_hits():
    peaks = np.array([[[0, 0], [100, 100]]])
    hits, present = point_hits(peaks, [[(0, 5, 5), (1, 100, 100)]], 3)
    np.testing.assert_array_equal(present, [[True, True, False]])
    np.testing.assert_array_equal(hits[0], [[True, False, False], [False, True, False]])


def test_distinctiveness_repetitive_vs_separated_maps():
    repeated = torch.zeros(1, 3, 16, 16)
    repeated[:, :, 2, 2] = 1
    separate = torch.zeros_like(repeated)
    separate[0, 0, 2, 2] = separate[0, 1, 8, 8] = separate[0, 2, 14, 14] = 1
    assert distinctiveness(repeated)[0] == pytest.approx(0.0)
    assert distinctiveness(separate)[0] > 0.95


def test_spatial_softmax_is_shift_invariant_and_display_only():
    cosine = torch.linspace(0, 1, 16).reshape(1, 4, 4)
    heat = display_map(cosine, 400.0)
    torch.testing.assert_close(heat, display_map(cosine + 2, 400.0))
    assert torch.isfinite(heat).all() and float(heat.max()) == 1.0
    assert float(display_map(cosine, 600.0).sum()) < float(
        display_map(cosine, 100.0).sum()
    )
    with pytest.raises(ValueError):
        display_map(cosine, 0)


def test_ap10k_keypoints_and_optional_region_pack(tmp_path):
    from test_four_loss import ap10k_fixture, configuration

    ap10k_fixture(tmp_path)
    dataset = AP10KDataset(tmp_path, "test")
    cfg = configuration()
    cfg.dataset.name = "ap10k"
    cfg.dataset.input_size = [16, 16]
    annotation = PartAnnotations(dataset, cfg)
    maps = np.zeros((1, 2, 4, 4), dtype=np.float32)
    maps[:, :, 1, 1] = 1
    hits, present = annotation.hits(maps, [0])
    assert hits.shape == (1, 2, 1) and present[0, 0]
    # Mark both eyes occluded: neither should enter the visibility denominator.
    for row in dataset.annotations["annotations"]:
        row["keypoints"] = [4, 4, 1, 12, 4, 1]
    occluded = PartAnnotations(dataset, cfg)
    assert not occluded.hits(maps, [0])[1].any()
    pack = tmp_path / "regions.npz"
    np.savez_compressed(
        pack,
        image_ids=[3],
        group_names=["head", "legs"],
        masks=np.ones((1, 2, 16, 16), dtype=bool),
    )
    cfg.eval.ap10k_regions = str(pack)
    coarse = PartAnnotations(dataset, cfg)
    hits, present = coarse.hits(maps, [0])
    assert hits.shape == (1, 2, 2)
    assert hits[0, :, 0].all() and not hits[0, :, 1].any()
    assert present[0, 0] and not present[0, 1]


def test_ap10k_rejects_cross_split_duplicates(tmp_path):
    from test_four_loss import ap10k_fixture

    ap10k_fixture(tmp_path)
    path = tmp_path / "annotations" / "ap10k-test-split1.json"
    document = json.loads(path.read_text())
    document["images"][0]["id"] = 1
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="leakage"):
        AP10KDataset(tmp_path)
