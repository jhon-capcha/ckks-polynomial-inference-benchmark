"""
Pruebas de las primitivas CKKS instrumentadas (Hito 4B).

Verifican corrección numérica (contra NumPy), consumo de niveles y estabilidad
de escala. No provocan operaciones inválidas que causarían access violations;
las precondiciones se validan por estado.
"""

from __future__ import annotations

import numpy as np
import pytest

from ckks_benchmark.he.ciphertext_state import OperationRecorder
from ckks_benchmark.he.ckks_context import create_context
from ckks_benchmark.he.operations import (
    add_ciphertexts,
    add_plain,
    decrypt_vector,
    encrypt_vector,
    multiply_ciphertexts,
    multiply_plain,
    square_ciphertext,
)

PROFILE = "ckks_n16384_d5"
TOL = 1e-3


@pytest.fixture(scope="module")
def he():
    return create_context(PROFILE).he


def _const(he, value: float) -> np.ndarray:
    return np.full(he.get_nSlots(), value, dtype=np.float64)


# --- Cifrado / descifrado ---
def test_encrypt_decrypt_roundtrip(he):
    x = np.array([1.5, -2.0, 0.3, 0.0, -0.001])
    ct = he.encryptFrac(np.concatenate([x, np.zeros(he.get_nSlots() - len(x))]))
    dec = decrypt_vector(he, ct, len(x))
    assert np.allclose(dec, x, atol=TOL)


def test_encrypt_decrypt_negative_and_small(he):
    x = _const(he, -0.005)
    ct = encrypt_vector(he, x)
    dec = decrypt_vector(he, ct, 1)
    assert abs(dec[0] - (-0.005)) < TOL


# --- Suma ---
def test_add_plain(he):
    ct = encrypt_vector(he, _const(he, 1.0))
    ct = add_plain(he, ct, _const(he, 0.5))
    assert abs(decrypt_vector(he, ct, 1)[0] - 1.5) < TOL


def test_add_ciphertexts(he):
    ct1 = encrypt_vector(he, _const(he, 1.0))
    ct2 = encrypt_vector(he, _const(he, 2.5))
    ct = add_ciphertexts(he, ct1, ct2)
    assert abs(decrypt_vector(he, ct, 1)[0] - 3.5) < TOL


# --- Multiplicación por plaintext ---
def test_multiply_plain(he):
    ct = encrypt_vector(he, _const(he, 0.5))
    ct = multiply_plain(he, ct, _const(he, 3.0))
    assert abs(decrypt_vector(he, ct, 1)[0] - 1.5) < TOL


def test_multiply_plain_consumes_level(he):
    """multiply_plain con rescale consume un nivel."""
    ct = encrypt_vector(he, _const(he, 0.5))
    level_before = ct.mod_level
    ct = multiply_plain(he, ct, _const(he, 2.0))
    assert ct.mod_level == level_before + 1


# --- Multiplicación ciphertext-ciphertext ---
def test_multiply_ciphertexts(he):
    ct1 = encrypt_vector(he, _const(he, 0.5))
    ct2 = encrypt_vector(he, _const(he, 2.0))
    ct = multiply_ciphertexts(he, ct1, ct2)
    assert abs(decrypt_vector(he, ct, 1)[0] - 1.0) < TOL


def test_square(he):
    ct = encrypt_vector(he, _const(he, 0.7))
    ct = square_ciphertext(he, ct)
    assert abs(decrypt_vector(he, ct, 1)[0] - 0.49) < TOL


# --- Escala estable y niveles ---
def test_scale_stable_after_rescale(he):
    """La escala vuelve a ~40 bits tras rescale."""
    ct = encrypt_vector(he, _const(he, 0.5))
    ct = square_ciphertext(he, ct)
    assert abs(float(ct.scale_bits) - 40.0) < 1.0


def test_multiplicative_depth_within_budget(he):
    """Cadena de cuadrados hasta la profundidad del perfil (6)."""
    ct = encrypt_vector(he, _const(he, 1.02))
    expected = 1.02
    for _ in range(5):  # 5 squares: dentro de profundidad 6
        ct = square_ciphertext(he, ct)
        expected = expected**2
    dec = decrypt_vector(he, ct, 1)[0]
    assert np.isfinite(dec)
    assert abs(dec - expected) / expected < 0.01  # error relativo < 1%


# --- Instrumentación ---
def test_recorder_tracks_levels(he):
    rec = OperationRecorder()
    ct = encrypt_vector(he, _const(he, 0.5), rec)
    ct = square_ciphertext(he, ct, recorder=rec)
    ct = square_ciphertext(he, ct, recorder=rec)
    summary = rec.summary()
    assert summary["n_operations"] == 3
    assert summary["levels_consumed"] == 2  # dos squares = dos niveles


# --- Alineación de operandos a distinto nivel ---
def test_operation_aligns_different_levels(he):
    """Sumar/multiplicar ciphertexts a distinto nivel se alinea sin crash."""
    ct_fresh = encrypt_vector(he, _const(he, 1.0))
    ct_consumed = encrypt_vector(he, _const(he, 2.0))
    ct_consumed = square_ciphertext(he, ct_consumed)  # ahora en nivel 1
    # ct_fresh (nivel 0) + ct_consumed (nivel 1): debe alinear y no crashear.
    result = add_ciphertexts(he, ct_consumed, ct_fresh)
    dec = decrypt_vector(he, result, 1)[0]
    assert abs(dec - (4.0 + 1.0)) < TOL  # 2^2 + 1 = 5
