# src/prepare_datasets.py
import os
import yaml
from pathlib import Path


def load_config(config_path: str = "configs/reid_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def verify_market1501(root: str) -> bool:
    root = Path(root)
    required = [
        "bounding_box_train",
        "bounding_box_test",
        "query",
    ]
    print(f"\n[Market-1501] Checking: {root}")
    ok = True
    for folder in required:
        p = root / folder
        if p.exists():
            count = len(list(p.glob("*.jpg")))
            print(f"  ✓ {folder}: {count} images")
        else:
            print(f"  ✗ MISSING: {folder}")
            ok = False
    return ok


def verify_duke(root: str) -> bool:
    root = Path(root)
    required = [
        "bounding_box_train",
        "bounding_box_test",
        "query",
    ]
    print(f"\n[DukeMTMC-reID] Checking: {root}")
    ok = True
    for folder in required:
        p = root / folder
        if p.exists():
            count = len(list(p.glob("*.jpg")))
            print(f"  ✓ {folder}: {count} images")
        else:
            print(f"  ✗ MISSING: {folder}")
            ok = False
    return ok


def check_expected_counts(root: str, dataset: str) -> None:
    # Known reference counts
    expected = {
        "Market1501": {
            "bounding_box_train": 12936,
            "bounding_box_test":  19732,
            "query":               3368,
        },
        "DukeMTMC": {
            "bounding_box_train": 16522,
            "bounding_box_test":  17661,
            "query":               2228,
        },
    }
    root = Path(root)
    refs = expected.get(dataset, {})
    print(f"\n[{dataset}] Count verification:")
    for folder, ref_count in refs.items():
        p = root / folder
        actual = len(list(p.glob("*.jpg"))) if p.exists() else 0
        status = "✓" if actual >= ref_count * 0.99 else "⚠ MISMATCH"
        print(f"  {status} {folder}: {actual} (expected ~{ref_count})")


if __name__ == "__main__":
    cfg = load_config()

    market_root = cfg["paths"]["market1501_root"]
    duke_root   = cfg["paths"]["duke_root"]

    m_ok = verify_market1501(market_root)
    check_expected_counts(market_root, "Market1501")

    d_ok = verify_duke(duke_root)
    check_expected_counts(duke_root, "DukeMTMC")

    print("\n─────────────────────────────")
    if m_ok and d_ok:
        print("✅ Both datasets verified successfully.")
    else:
        print("❌ One or more datasets failed verification. Check paths in configs/reid_config.yaml.")