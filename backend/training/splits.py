"""
Splits that cannot leak: grouped by patient, stratified by label, fixed by seed.

  locked_split   carves the untouched TEST set off first. Nothing (features, thresholds, calibration,
                 model selection) may look at it; train.py touches it only in the final evaluation.
  cv_folds       5-fold grouped CV inside the development part, for model comparison and calibration.
"""
import hashlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def locked_split(df: pd.DataFrame, test_fraction: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, Dict]:
    k = max(2, round(1 / test_fraction))
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    dev_idx, test_idx = next(splitter.split(df, df["esi"], groups=df["group"]))
    test_groups = sorted(set(df.iloc[test_idx]["group"].astype(str)))
    manifest = {
        "seed": seed, "test_fraction": test_fraction, "n_dev": int(len(dev_idx)), "n_test": int(len(test_idx)),
        "n_test_groups": len(test_groups),
        "test_groups_sha256": hashlib.sha256("\n".join(test_groups).encode()).hexdigest(),
    }
    return dev_idx, test_idx, manifest


def cv_folds(dev: pd.DataFrame, k: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(splitter.split(dev, dev["esi"], groups=dev["group"]))
