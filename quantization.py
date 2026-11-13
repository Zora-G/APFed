import numpy as np


def quantize(values, r=16):
    """Eq. (8): fixed-point integers, truncating toward zero without clipping."""
    return np.trunc(np.asarray(values, dtype=np.float64) * (2 ** r)).astype(np.int64)
