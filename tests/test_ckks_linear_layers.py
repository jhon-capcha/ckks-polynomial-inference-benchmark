"""
Pruebas de las capas lineales cifradas SIMD (Hito 4D).

Verifican la reducción de slots, el producto punto cifrado, fc2 contra NumPy,
y el bloque act3 -> fc2. Requieren claves de rotación.
"""

from __future__ import annotations

import numpy as np
import pytest

from ckks_benchmark.he.ciphertext_state import OperationRecorder
from ckks_benchmark.he.ckks_context import create_context
from ckks_benchmark.he.linear_layer import (
    decrypt_logits,
    encrypted_dot_product,
    encrypted_linear_simd,
)
from ckks_benchmark.he.polynomial_evaluator import evaluate_polynomial_ckks
from ckks_benchmark.he.simd_operations import (
    next_power_of_two,
    pad_to_slots,
    sum_slots_power_of_two,
)

TOL = 1e-3


@pytest.fixture(scope="module")
def he_rot():
    """Contexto con claves de rotación (compartido)."""
    return create_context("ckks_n16384_d5", generate_rotation_keys=True).he


# --- Utilidades SIMD ---
def test_next_power_of_two():
    assert next_power_of_two(120) == 128
    assert next_power_of_two(128) == 128
    assert next_power_of_two(4) == 4
    assert next_power_of_two(5) == 8


def test_sum_slots_small(he_rot):
    x = pad_to_slots(np.array([1.0, 2.0, 3.0, 4.0]), he_rot.get_nSlots())
    ct = he_rot.encryptFrac(x)
    result = sum_slots_power_of_two(he_rot, ct, reduction_size=4)
    assert abs(he_rot.decryptFrac(result)[0] - 10.0) < TOL


def test_sum_slots_120(he_rot):
    """Suma de 120 valores con padding a 128 (los ceros no contaminan)."""
    rng = np.random.default_rng(42)
    z = rng.standard_normal(120)
    x = pad_to_slots(z, he_rot.get_nSlots())
    ct = he_rot.encryptFrac(x)
    result = sum_slots_power_of_two(he_rot, ct, reduction_size=128)
    assert abs(he_rot.decryptFrac(result)[0] - float(np.sum(z))) < 1e-2


def test_sum_slots_rejects_non_power_of_two(he_rot):
    ct = he_rot.encryptFrac(pad_to_slots(np.array([1.0]), he_rot.get_nSlots()))
    with pytest.raises(ValueError):
        sum_slots_power_of_two(he_rot, ct, reduction_size=120)  # no potencia de 2


# --- Producto punto ---
def test_dot_product(he_rot):
    rng = np.random.default_rng(7)
    x = rng.standard_normal(120)
    w = rng.standard_normal(120)
    expected = float(np.dot(w, x) + 0.5)
    ct = he_rot.encryptFrac(pad_to_slots(x, he_rot.get_nSlots()))
    result = encrypted_dot_product(he_rot, ct, w, logical_size=120, bias=0.5)
    assert abs(he_rot.decryptFrac(result)[0] - expected) < 1e-2


# --- fc2 completa ---
def test_fc2_against_numpy(he_rot):
    rng = np.random.default_rng(11)
    x = rng.standard_normal(120) * 0.5
    W = rng.standard_normal((10, 120)) * 0.3
    b = rng.standard_normal(10) * 0.1
    y_clear = W @ x + b
    ct = he_rot.encryptFrac(pad_to_slots(x, he_rot.get_nSlots()))
    logit_cts = encrypted_linear_simd(he_rot, ct, W, b, logical_size=120)
    y_ckks = decrypt_logits(he_rot, logit_cts)
    assert np.allclose(y_clear, y_ckks, atol=1e-2)
    assert np.argmax(y_clear) == np.argmax(y_ckks)


def test_fc2_output_count(he_rot):
    """fc2 produce un ciphertext por logit (10)."""
    x = pad_to_slots(np.zeros(120), he_rot.get_nSlots())
    W = np.ones((10, 120)) * 0.1
    b = np.zeros(10)
    ct = he_rot.encryptFrac(x)
    logit_cts = encrypted_linear_simd(he_rot, ct, W, b, logical_size=120)
    assert len(logit_cts) == 10


# --- Bloque act3 -> fc2 ---
def test_final_block_degree3(he_rot):
    """Bloque act3(grado 3) -> fc2 completa y conserva predicción."""
    rng = np.random.default_rng(5)
    z = rng.standard_normal(120) * 2.0
    coeffs = np.array([0.1, 0.6, 0.05, -0.002])  # grado 3 tipo suave
    W = rng.standard_normal((10, 120)) * 0.3
    b = rng.standard_normal(10) * 0.1

    act3_clear = np.polyval(coeffs[::-1], z)
    logits_clear = W @ act3_clear + b

    ct = he_rot.encryptFrac(pad_to_slots(z, he_rot.get_nSlots()))
    act3_ct = evaluate_polynomial_ckks(he_rot, ct, coeffs, strategy="horner")
    logit_cts = encrypted_linear_simd(he_rot, act3_ct, W, b, logical_size=120)
    logits_ckks = decrypt_logits(he_rot, logit_cts)

    assert np.all(np.isfinite(logits_ckks))
    assert np.argmax(logits_clear) == np.argmax(logits_ckks)
    assert np.max(np.abs(logits_clear - logits_ckks)) < 1e-1


def test_final_block_degree5_feasible(he_rot):
    """Bloque act3(grado 5) -> fc2 completa dentro de la cadena (6 niveles)."""
    rng = np.random.default_rng(9)
    z = rng.standard_normal(120)
    coeffs = np.array([0.1, 0.6, 0.05, -0.002, 0.001, -0.0005])  # grado 5
    W = rng.standard_normal((10, 120)) * 0.3
    b = rng.standard_normal(10) * 0.1

    act3_clear = np.polyval(coeffs[::-1], z)
    logits_clear = W @ act3_clear + b

    rec = OperationRecorder()
    ct = he_rot.encryptFrac(pad_to_slots(z, he_rot.get_nSlots()))
    rec.record(ct, "encrypt")
    act3_ct = evaluate_polynomial_ckks(he_rot, ct, coeffs, strategy="horner", recorder=rec)
    logit_cts = encrypted_linear_simd(he_rot, act3_ct, W, b, logical_size=120, recorder=rec)
    logits_ckks = decrypt_logits(he_rot, logit_cts)

    assert np.all(np.isfinite(logits_ckks))  # completa sin scale out of bounds
    assert np.argmax(logits_clear) == np.argmax(logits_ckks)
