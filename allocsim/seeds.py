"""Воспроизводимые сиды.

Сид каждой случайной составляющей примера выводится из набора параметров через SHA-256,
а не через встроенный hash() Python (он меняется от запуска к запуску). Поэтому один и тот же
набор параметров всегда даёт одни и те же случайные числа на любой машине.
"""
import hashlib
import json

import numpy as np


def seed_sequence(*parts) -> np.random.SeedSequence:
    """SeedSequence из произвольного набора частей (строки, числа, None)."""
    key = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    words = [int.from_bytes(digest[i:i + 4], "little") for i in range(0, 32, 4)]
    return np.random.SeedSequence(words)


def rng(*parts) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed_sequence(*parts)))
