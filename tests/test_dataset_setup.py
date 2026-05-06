from anra_paths import DATASET_CANONICAL, ensure_dirs


def test_ensure_dirs_creates_canonical_dataset():
    ensure_dirs()
    assert DATASET_CANONICAL.exists(), f"{DATASET_CANONICAL} still missing after ensure_dirs()"

