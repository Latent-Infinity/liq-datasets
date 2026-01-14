import numpy as np

from liq.datasets.bundles import DatasetBundle, compute_hash


def test_compute_hash_stable() -> None:
    data = {"b": 2, "a": 1}
    first = compute_hash(data)
    second = compute_hash(data)
    assert first == second


def test_dataset_bundle_basic() -> None:
    bundle = DatasetBundle(X=np.zeros((1, 2, 3)), y={"y": np.array([1])})
    assert bundle.X.shape == (1, 2, 3)
    assert bundle.y["y"][0] == 1
