import numpy as np


def quantize(x, r, clip_bound=None):
    """Multiply by 2^r and round toward zero."""
    if int(r) != r or int(r) < 0:
        raise ValueError('r must be a non-negative integer')
    a = np.asarray(x, dtype=float)
    if clip_bound is not None and float(clip_bound) > 0.0:
        clip_bound = float(clip_bound)
        if not np.isfinite(clip_bound):
            raise ValueError('clip_bound must be finite')
        a = np.clip(a, -clip_bound, clip_bound)
    scaled = a * float(2 ** int(r))
    if not np.all(np.isfinite(scaled)):
        raise ValueError('quantization input must be finite')
    # Use the exact half-open int64 domain.  Comparing a float with int64.max is
    # unsafe because float(int64.max) rounds to 2**63.
    lower = -float(1 << 63)
    upper = float(1 << 63)
    if np.any(scaled < lower) or np.any(scaled >= upper):
        raise OverflowError('quantized value does not fit int64')
    q = np.trunc(scaled).astype(np.int64)
    if np.isscalar(x):
        return int(q)
    return q

def pack_values(values, bits=32, pad_bits=0):
    """Pack signed fixed-width integers into one non-negative Python integer."""
    bits = int(bits)
    if bits < 2:
        raise ValueError('bits must be at least 2')
    lower = -(1 << (bits - 1))
    upper = (1 << (bits - 1)) - 1
    pad_bits = int(pad_bits)
    if pad_bits < 0:
        raise ValueError('pad_bits must be non-negative')
    slot_bits = bits + pad_bits
    packed = 0
    for index, value in enumerate(values):
        value = int(value)
        if value < lower or value > upper:
            raise OverflowError('value does not fit its packing slot')
        packed |= (value & ((1 << bits) - 1)) << (index * slot_bits)
    return packed

def unpack_values(packed, num_values, bits=32, pad_bits=0):
    bits = int(bits)
    pad_bits = int(pad_bits)
    if bits < 2:
        raise ValueError('bits must be at least 2')
    if pad_bits < 0:
        raise ValueError('pad_bits must be non-negative')
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    slot_bits = bits + pad_bits
    values = []
    for index in range(int(num_values)):
        value = (int(packed) >> (index * slot_bits)) & mask
        if value & sign:
            value -= 1 << bits
        values.append(value)
    return values

def _chunks(values, slots_per_ciphertext):
    slots_per_ciphertext = int(slots_per_ciphertext)
    if slots_per_ciphertext < 1:
        raise ValueError('slots_per_ciphertext must be positive')
    values = [int(value) for value in values]
    return [values[i:i + slots_per_ciphertext]
            for i in range(0, len(values), slots_per_ciphertext)]

def _integer_values_array(values):
    """Return the one-dimensional integer sequence used by packing helpers."""
    if isinstance(values, np.ndarray):
        array = values
    else:
        try:
            array = np.asarray(values)
        except (TypeError, ValueError, OverflowError):
            array = np.asarray(list(values), dtype=object)
        if array.ndim == 0 and not np.isscalar(values):
            array = np.asarray(list(values), dtype=object)
    if array.ndim != 1:
        raise ValueError('packing values must be one-dimensional')
    if np.issubdtype(array.dtype, np.integer):
        return array
    # Convert scalar values to integers.
    coerced = [int(value) for value in array.tolist()]
    try:
        return np.asarray(coerced, dtype=np.int64)
    except (OverflowError, TypeError, ValueError):
        return np.asarray(coerced, dtype=object)

def packed_scalar_limit(values, bits=48, pad_bits=20,
                        slots_per_ciphertext=3,
                        signed_plaintext_bound=None, scalar_cap=1000000):
    """Largest positive scalar that cannot spill a packed signed slot."""
    bits = int(bits)
    pad_bits = int(pad_bits)
    if bits < 2 or pad_bits < 0:
        raise ValueError('invalid packing bit widths')
    slots_per_ciphertext = int(slots_per_ciphertext)
    if slots_per_ciphertext < 1:
        raise ValueError('slots_per_ciphertext must be positive')

    # Use array extrema when values fit in int64.
    array = _integer_values_array(values)
    if array.size == 0:
        raise ValueError('packing needs at least one value')
    if np.issubdtype(array.dtype, np.integer):
        minimum_value = int(np.min(array))
        maximum_value = int(np.max(array))
    else:
        minimum_value = min(int(value) for value in array)
        maximum_value = max(int(value) for value in array)

    lower = -(1 << (bits - 1))
    upper = (1 << (bits - 1)) - 1
    if minimum_value < lower or maximum_value > upper:
        raise OverflowError('value does not fit its packing slot')
    maximum = min(int(scalar_cap), 1 << pad_bits)
    max_abs = max(abs(minimum_value), abs(maximum_value))
    if max_abs > 0:
        maximum = min(maximum, ((1 << (bits - 1)) - 1) // max_abs)

    if signed_plaintext_bound is not None:
        signed_plaintext_bound = int(signed_plaintext_bound)
        if signed_plaintext_bound <= 0:
            raise ValueError('signed_plaintext_bound must be positive')
        # Check packed chunks when the width bound reaches the plaintext limit.
        mask = (1 << bits) - 1
        slot_bits = bits + pad_bits
        largest_packed = sum(
            mask << (index * slot_bits)
            for index in range(slots_per_ciphertext))
        if maximum * largest_packed >= signed_plaintext_bound:
            integer_values = [int(value) for value in array.tolist()]
            for offset in range(0, len(integer_values),
                                slots_per_ciphertext):
                packed = pack_values(
                    integer_values[offset:offset + slots_per_ciphertext],
                    bits, pad_bits)
                if packed > 0:
                    maximum = min(
                        maximum,
                        (signed_plaintext_bound - 1) // packed)
    if maximum < 1:
        raise OverflowError('no safe positive scalar for packed plaintext')
    return int(maximum)

def _simulation_packed_scalar_transport(values, scalar, bits, pad_bits,
                                        slots_per_ciphertext,
                                        block_values=196608):
    """Exact, bounded-memory simulation of pack/blind/unpack."""
    source = _integer_values_array(values)
    count = int(source.size)
    if count == 0:
        return [], 0

    slots = int(slots_per_ciphertext)
    block_values = max(slots, int(block_values))
    block_values -= block_values % slots
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    modulus = 1 << bits
    slot_bits = bits + pad_bits
    transported = []
    ciphertext_count = 0

    for offset in range(0, count, block_values):
        stop = min(count, offset + block_values)
        block = np.asarray(source[offset:stop], dtype=object).reshape(-1)
        original_length = int(block.size)
        padding = (-original_length) % slots
        if padding:
            block = np.concatenate(
                (block, np.zeros(padding, dtype=object)))
        rows = block.reshape(-1, slots)

        packed = np.zeros(rows.shape[0], dtype=object)
        for index in range(slots):
            encoded = np.bitwise_and(rows[:, index], mask)
            if index:
                encoded = np.left_shift(encoded, index * slot_bits)
            packed = np.bitwise_or(packed, encoded)
        masked = packed * scalar
        if any(value < 0 for value in masked):
            raise OverflowError(
                'masked packed plaintext left non-negative domain')

        unpacked_columns = []
        for index in range(slots):
            decoded = np.bitwise_and(
                np.right_shift(masked, index * slot_bits), mask)
            decoded = np.where(
                np.bitwise_and(decoded, sign) != 0,
                decoded - modulus, decoded)
            unpacked_columns.append(decoded)
        unpacked = np.stack(unpacked_columns, axis=1).reshape(-1)
        unpacked = unpacked[:original_length]

        expected = np.asarray(source[offset:stop], dtype=object) * scalar
        if not np.array_equal(unpacked, expected):
            raise OverflowError(
                'packed scalar transport changed a coordinate')
        transported.extend(unpacked.tolist())
        ciphertext_count += rows.shape[0]
    return transported, int(ciphertext_count)

def packed_scalar_transport(values, scalar, bits=48, pad_bits=20,
                            slots_per_ciphertext=3, crypto=0,
                            pk=None, sk=None, signed_plaintext_bound=None):
    """Pack, scalar-blind, decrypt/simulate, and unpack signed integers."""
    scalar = int(scalar)
    if scalar < 1:
        raise ValueError('scalar must be positive')
    raw_values = values
    limit = packed_scalar_limit(
        raw_values, bits, pad_bits, slots_per_ciphertext,
        signed_plaintext_bound=signed_plaintext_bound,
        scalar_cap=scalar)
    if scalar > limit:
        raise OverflowError('scalar would cause slot carry or plaintext wraparound')
    if crypto and (pk is None or sk is None):
        raise ValueError('crypto packed transport requires pk and sk')

    if not crypto:
        return _simulation_packed_scalar_transport(
            raw_values, scalar, int(bits), int(pad_bits),
            int(slots_per_ciphertext))

    # Encrypt, scale and unpack each chunk.
    values = [int(value) for value in raw_values]
    transported = []
    ciphertext_count = 0
    for chunk in _chunks(values, slots_per_ciphertext):
        ciphertext_count += 1
        packed = pack_values(chunk, bits, pad_bits)
        if crypto:
            from paillier import Encrypt, Decrypt, scalar_mul
            masked_packed = Decrypt(
                sk, scalar_mul(Encrypt(pk, packed), scalar, pk))
        else:
            masked_packed = scalar * packed
        if masked_packed < 0:
            raise OverflowError('masked packed plaintext left non-negative domain')
        transported.extend(
            unpack_values(masked_packed, len(chunk), bits, pad_bits))

    expected = [scalar * value for value in values]
    if transported != expected:
        raise OverflowError('packed scalar transport changed a coordinate')
    return transported, ciphertext_count
