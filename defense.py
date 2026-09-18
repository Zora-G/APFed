import math
import numbers
import random

import numpy as np
import torch

from paillier import Encrypt, Decrypt, scalar_mul, sample_positive_invertible_scalar
from quantization import quantize, packed_scalar_limit, packed_scalar_transport


def simulation_signed_plaintext_bound(key_bits=256):
    """Conservative signed bound for a simulated key of the requested length."""
    key_bits = int(key_bits)
    if key_bits < 32:
        raise ValueError('key_bits must be at least 32')
    return (1 << (key_bits - 2)) // 3

def _array(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)

def _value_bound(x, r=1):
    return int(np.max(np.abs(_array(x))) * r) + 1

def _exact_int_dot(left, right):
    """Exact integer dot product with a proved-safe int64 fast path."""
    left = np.asarray(left, dtype=np.int64).reshape(-1)
    right = np.asarray(right, dtype=np.int64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError('integer dot operands must have the same shape')
    if left.size == 0:
        return 0
    left_min, left_max = int(np.min(left)), int(np.max(left))
    right_min, right_max = int(np.min(right)), int(np.max(right))
    left_abs = max(abs(left_min), abs(left_max))
    right_abs = max(abs(right_min), abs(right_max))
    absolute_sum_bound = left_abs * right_abs * int(left.size)
    if absolute_sum_bound <= np.iinfo(np.int64).max:
        return int(np.dot(left, right))
    return sum(int(x) * int(y) for x, y in zip(left, right))

def normalize_gradient(g_i, g_0):
    """Gradient normalization: ||g_0|| / ||g_i|| * g_i."""
    g_i = _array(g_i)
    n_i = np.linalg.norm(g_i)
    if n_i == 0:
        return g_i.copy()
    return np.linalg.norm(_array(g_0)) / n_i * g_i

def _validate_crypto_scalar(value, pk, value_bound, name):
    if not isinstance(value, numbers.Integral):
        raise ValueError(name + ' must be an integer')
    value = int(value)
    if value <= 0 or math.gcd(value, pk.n) != 1:
        raise ValueError(name + ' must be a positive element of Z_n^*')
    if value * int(value_bound) >= pk.signed_bound:
        raise OverflowError(name + ' would wrap the signed Paillier plaintext')
    return value

def SCos(g_i, g_med, crypto=0, pk=None, sk=None, r=16,
         lambda_i=None, return_lambda=False, packing=False, pack_bits=48,
         pad_bits=20, slots_per_ciphertext=3, return_transport=False,
         signed_plaintext_bound=None, clip_bound=None):
    """Cosine under one positive scalar for the complete vector pair."""
    g_i = _array(g_i).reshape(-1)
    g_med = _array(g_med).reshape(-1)
    if g_i.shape != g_med.shape:
        raise ValueError('g_i and g_med must have the same dimension')

    transport_ciphertexts = 0
    if crypto:
        if pk is None or sk is None:
            raise ValueError('crypto=1 requires pk and sk')
        q_i = quantize(g_i, r, clip_bound)
        q_med = quantize(g_med, r, clip_bound)
        value_bound = max(_value_bound(q_i), _value_bound(q_med))
        if packing:
            scalar_limit = min(
                packed_scalar_limit(q_i.tolist(), pack_bits, pad_bits,
                                    slots_per_ciphertext, pk.signed_bound),
                packed_scalar_limit(q_med.tolist(), pack_bits, pad_bits,
                                    slots_per_ciphertext, pk.signed_bound))
        else:
            scalar_limit = 1000000
        if lambda_i is None:
            lambda_i = sample_positive_invertible_scalar(
                pk, value_bound, limit=scalar_limit)
        lambda_i = _validate_crypto_scalar(lambda_i, pk, value_bound, 'lambda_i')
        if packing:
            d_i, count_i = packed_scalar_transport(
                q_i.tolist(), lambda_i, pack_bits, pad_bits,
                slots_per_ciphertext, crypto=1, pk=pk, sk=sk,
                signed_plaintext_bound=pk.signed_bound)
            d_med, count_med = packed_scalar_transport(
                q_med.tolist(), lambda_i, pack_bits, pad_bits,
                slots_per_ciphertext, crypto=1, pk=pk, sk=sk,
                signed_plaintext_bound=pk.signed_bound)
            transport_ciphertexts = count_i + count_med
        else:
            d_i = []
            d_med = []
            # One lambda_i is shared by both whole vectors, never per coordinate.
            for j in range(len(q_i)):
                d_i.append(Decrypt(
                    sk, scalar_mul(Encrypt(pk, int(q_i[j])), lambda_i, pk)))
                d_med.append(Decrypt(
                    sk, scalar_mul(Encrypt(pk, int(q_med[j])), lambda_i, pk)))
                transport_ciphertexts += 2
    else:
        q_i = quantize(g_i, r, clip_bound)
        q_med = quantize(g_med, r, clip_bound)
        if signed_plaintext_bound is None:
            signed_plaintext_bound = simulation_signed_plaintext_bound()
        if packing:
            scalar_limit = min(
                packed_scalar_limit(q_i, pack_bits, pad_bits,
                                    slots_per_ciphertext,
                                    signed_plaintext_bound),
                packed_scalar_limit(q_med, pack_bits, pad_bits,
                                    slots_per_ciphertext,
                                    signed_plaintext_bound))
        else:
            scalar_limit = 1000000
        if lambda_i is None:
            lambda_i = random.randint(1, scalar_limit)
        if not isinstance(lambda_i, numbers.Integral) or lambda_i <= 0:
            raise ValueError('lambda_i must be a positive integer')
        if packing:
            d_i, count_i = packed_scalar_transport(
                q_i, lambda_i, pack_bits, pad_bits,
                slots_per_ciphertext,
                signed_plaintext_bound=signed_plaintext_bound)
            d_med, count_med = packed_scalar_transport(
                q_med, lambda_i, pack_bits, pad_bits,
                slots_per_ciphertext,
                signed_plaintext_bound=signed_plaintext_bound)
            transport_ciphertexts = count_i + count_med
        else:
            d_i = [int(lambda_i) * int(x) for x in q_i]
            d_med = [int(lambda_i) * int(x) for x in q_med]

    if not crypto:
        # The common positive scalar cancels in the cosine.
        lambda_sq = int(lambda_i) * int(lambda_i)
        inner = lambda_sq * _exact_int_dot(q_i, q_med)
        norm_i_sq = lambda_sq * _exact_int_dot(q_i, q_i)
        norm_med_sq = lambda_sq * _exact_int_dot(q_med, q_med)
    else:
        inner = sum(x * y for x, y in zip(d_i, d_med))
        norm_i_sq = sum(x * x for x in d_i)
        norm_med_sq = sum(x * x for x in d_med)
    den = math.sqrt(norm_i_sq) * math.sqrt(norm_med_sq)
    cos_i = 0.0 if den == 0 else float(inner / den)
    if return_transport:
        return cos_i, {'scalar': int(lambda_i),
                       'packed_ciphertext_count': transport_ciphertexts}
    if return_lambda:
        return cos_i, int(lambda_i)
    return cos_i


class TrustedAnchorGuard:
    def __init__(self, anchor, clusters):
        self.anchor = torch.as_tensor(anchor).detach().cpu().double().reshape(-1).clone()
        if not torch.isfinite(self.anchor).all() or self.anchor.norm() <= 0:
            raise ValueError('A finite, nonzero clean anchor is required')
        members = [int(i) for c in clusters for i in c]
        if not members or len(set(members)) != len(members) or any(not c for c in clusters):
            raise ValueError('Clusters must be a nonempty disjoint partition')
        self.caps = [len(c)/len(members) for c in clusters]
        self.limit = float(self.anchor.norm())
        self.last = {}

    def filter(self, grads, args=None, crypto=None, pk=None, sk=None,
               cos_threshold=0., r=16, **kw):
        # Filter each update against the fixed clean direction.
        anchor = self.anchor.numpy()
        scores = [SCos(g, anchor, crypto=crypto or 0, pk=pk, sk=sk, r=r,
            packing=bool(getattr(args, 'packing', 0)),
            pack_bits=getattr(args, 'pack_bits', 48),pad_bits=getattr(args, 'pad_bits', 20),
            slots_per_ciphertext=getattr(args, 'slots_per_ciphertext', 3),
            clip_bound=getattr(args, 'quant_clip', 0) or None) for g in np.asarray(grads)]
        return ([i for i,s in enumerate(scores) if s > 0],
                [i for i,s in enumerate(scores) if s <= 0], scores)

    def cra(self, updates, previous, eta=1., mode=None, input_type='gradient', cluster_ids=None):
        if input_type != 'gradient' or not math.isfinite(eta) or eta < 0:
            raise ValueError('Finite nonnegative eta and gradient input required')
        ids = list(range(len(updates))) if cluster_ids is None else list(cluster_ids)
        if len(ids)!=len(updates) or len(set(ids))!=len(ids):raise ValueError('Invalid cluster IDs')
        w = torch.as_tensor(previous).detach().float()
        if not torch.isfinite(w).all():raise ValueError('Nonfinite global model')
        agg = torch.zeros_like(w)
        trust, weights = [], []
        for g,i in zip(updates,ids):
            if not 0 <= i < len(self.caps):raise ValueError('Unknown cluster')
            g = torch.as_tensor(g).detach().to(w).reshape(-1)
            if g.numel()!=self.anchor.numel() or not torch.isfinite(g).all():
                raise ValueError('Invalid update')
            gd=g.double().cpu();norm=float(gd.norm())
            score=max(0.,min(1.,float(gd@self.anchor)/(norm*self.limit))) if norm else 0.
            weight=self.caps[i]*score
            # Keep weights proportional to the original cluster sizes.
            agg += g * (min(1., self.limit/norm) if norm else 1.) * weight
            trust.append(score);weights.append(weight)
        result=w-float(eta)*agg
        if not torch.isfinite(result).all():raise FloatingPointError('Nonfinite guarded model')
        self.last=dict(anchor_norm=self.limit,aggregate_norm=float(agg.double().norm()),
                       cluster_ids=ids,weight_sum=sum(weights),all_rejected=not any(weights))
        return result,dict(mu_values=trust,normalized_weights=weights,
            baseline_index=None,baseline_cluster_id=None,baseline_similarity_to_previous_global=None,
            cra_zero_weight_fallback=False,input_type='gradient',mode='frozen_clean_anchor_bounded_clusters',
            weights_are_normalized=False)
