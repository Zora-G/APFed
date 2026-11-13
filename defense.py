"""APFed normalization, additive-mask SMed and scalar-blinded SCos."""

import math
import secrets

import numpy as np

from paillier import Encrypt, Decrypt, scalar_mul, sample_positive_invertible_scalar
from quantization import quantize


def normalize_gradient(gradient, leader_gradient):
    gradient = np.asarray(gradient, dtype=np.float64)
    norm = np.linalg.norm(gradient)
    if norm == 0:
        return gradient.copy()
    return gradient * (np.linalg.norm(leader_gradient) / norm)


def cosine(left, right):
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return 0.0
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


def SMed(gradients, crypto=0, pk=None, sk=None, r=16, mask=None):
    """Algorithm 4, with one common integer mask for every coordinate/client.

    Keep the middle pair as integers through unmasking so even cluster sizes
    retain half-integer medians exactly. Plain execution gives the same median.
    """
    values = quantize(gradients, r)
    if not crypto:
        return np.median(values, axis=0) / (2 ** r)
    if mask is None:
        mask = secrets.randbelow(2 ** 32)
    bound = max(abs(int(x)) for x in values.flat)
    if bound + abs(mask) >= pk.signed_bound:
        raise OverflowError('SMed mask would wrap the signed plaintext')
    encrypted_mask = Encrypt(pk, mask)
    decrypted = [[Decrypt(sk, Encrypt(pk, int(x)) * encrypted_mask % pk.nsquare)
                  for x in row] for row in values]
    middle = len(values) // 2
    median = []
    for column in zip(*decrypted):
        ordered = sorted(column)
        if len(values) % 2:
            median.append((ordered[middle] - mask) / (2 ** r))
        else:
            median.append((ordered[middle - 1] + ordered[middle] - 2 * mask)
                          / (2 * (2 ** r)))
    return np.asarray(median, dtype=np.float64)


def SCos(gradient, median, crypto=0, pk=None, sk=None, r=16, lambda_i=None):
    """One positive integer scales both whole vectors for cosine similarity."""
    left, right = quantize(gradient, r), quantize(median, r)
    if not crypto:
        # The common scalar cancels; evaluate the same quantized vectors.
        return cosine(left, right)
    bound = max(1, max(abs(int(x)) for x in left), max(abs(int(x)) for x in right))
    if lambda_i is None:
        lambda_i = sample_positive_invertible_scalar(pk, bound)
    if not isinstance(lambda_i, int) or lambda_i <= 0:
        raise ValueError('SCos requires a positive integer scalar')
    if lambda_i * bound >= pk.signed_bound:
        raise OverflowError('SCos scalar would wrap the signed plaintext')
    left = [Decrypt(sk, scalar_mul(Encrypt(pk, int(x)), lambda_i, pk)) for x in left]
    right = [Decrypt(sk, scalar_mul(Encrypt(pk, int(x)), lambda_i, pk)) for x in right]
    inner = sum(x * y for x, y in zip(left, right))
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
    return float(np.clip(inner / denominator, -1.0, 1.0)) if denominator else 0.0
