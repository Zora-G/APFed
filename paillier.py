import math
import secrets
from dataclasses import dataclass


@dataclass(frozen=True)
class PublicKey:
    n: int

    @property
    def g(self):
        return self.n + 1

    @property
    def nsquare(self):
        return self.n * self.n

    @property
    def signed_bound(self):
        return self.n // 3


@dataclass(frozen=True)
class PrivateKey:
    public_key: PublicKey
    lam: int
    mu: int


def _is_probable_prime(candidate, rounds=40):
    if candidate in (2, 3):
        return True
    if candidate < 2 or candidate % 2 == 0:
        return False
    odd = candidate - 1
    shifts = 0
    while odd % 2 == 0:
        shifts += 1
        odd //= 2
    for _ in range(rounds):
        base = secrets.randbelow(candidate - 3) + 2
        value = pow(base, odd, candidate)
        if value in (1, candidate - 1):
            continue
        for _ in range(shifts - 1):
            value = pow(value, 2, candidate)
            if value == candidate - 1:
                break
        else:
            return False
    return True


def _prime(bits):
    while True:
        candidate = secrets.randbits(bits) | (1 << (bits - 1)) | 1
        if _is_probable_prime(candidate):
            return candidate


def keygen(bits=1024):
    """Generate a Paillier modulus of exactly bits bits using Miller–Rabin."""
    while True:
        p, q = _prime(bits // 2), _prime(bits - bits // 2)
        n = p * q
        lam = math.lcm(p - 1, q - 1)
        if p != q and n.bit_length() == bits and math.gcd(lam, n) == 1:
            pk = PublicKey(n)
            return pk, PrivateKey(pk, lam, pow(lam, -1, n))


def _L(x, n):
    return (x - 1) // n

def _modinv(a, modulus):
    old_r, r = int(a), int(modulus)
    old_s, s = 1, 0
    while r:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
    if old_r != 1:
        raise ValueError('value is not invertible modulo modulus')
    return old_s % modulus

def _encode(value, pk):
    value = int(value)
    if abs(value) >= pk.signed_bound:
        raise OverflowError('signed plaintext exceeds Paillier safety bound')
    return value % pk.n

def Encrypt(pk, value, r_value=None):
    """Encrypt one signed integer."""
    m = _encode(value, pk)
    if r_value is None:
        r_value = secrets.randbelow(pk.n - 1) + 1
        while math.gcd(r_value, pk.n) != 1:
            r_value = secrets.randbelow(pk.n - 1) + 1
    if math.gcd(r_value, pk.n) != 1:
        raise ValueError('Paillier randomness must be in Z_n^*')
    return (pow(pk.g, m, pk.nsquare) * pow(r_value, pk.n, pk.nsquare)) % pk.nsquare

def Decrypt(sk, ciphertext):
    pk = sk.public_key
    x = pow(int(ciphertext), sk.lam, pk.nsquare)
    m = (_L(x, pk.n) * sk.mu) % pk.n
    if m > pk.n // 2:
        m -= pk.n
    return int(m)

def scalar_mul(ciphertext, scalar, pk):
    """Enc(x)^rho = Enc(rho*x).  Scalars here are non-negative."""
    scalar = int(scalar)
    if scalar < 0:
        # c^(-a) encrypts -a*x because ciphertexts form a multiplicative group.
        return pow(_modinv(int(ciphertext), pk.nsquare), -scalar, pk.nsquare)
    return pow(int(ciphertext), scalar, pk.nsquare)

def sample_positive_invertible_scalar(pk, value_bound, limit=1000000):
    """Sample a positive scalar without crossing the signed plaintext margin."""
    value_bound = max(1, int(math.ceil(abs(value_bound))))
    maximum = (pk.signed_bound - 1) // value_bound
    maximum = min(maximum, int(limit))
    if maximum < 1:
        raise OverflowError('plaintext bound leaves no safe blinding scalar')
    while True:
        rho = secrets.randbelow(maximum) + 1
        if math.gcd(rho, pk.n) == 1:
            return rho
