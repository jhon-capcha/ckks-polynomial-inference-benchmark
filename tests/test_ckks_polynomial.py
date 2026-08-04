"""
Pruebas de la evaluación homomórfica de polinomios (Hito 4C).

Verifican Horner (estrategia oficial) para grados 3 y 5, la equivalencia entre
evaluación clara y cifrada, el consumo de niveles, el gate de estrategias
experimentales, y el manejo de coeficientes nulos.
"""

from __future__ import annotations

import numpy as np
import pytest

from ckks_benchmark.he.ciphertext_state import OperationRecorder
from ckks_benchmark.he.ckks_context import create_context
from ckks_benchmark.he.operations import decrypt_vector, encrypt_vector
from ckks_benchmark.he.polynomial_evaluator import (
    OFFICIAL_STRATEGIES,
    evaluate_polynomial_ckks,
)

PROFILE_D5 = "ckks_n16384_d5"
PROFILE_D3 = "ckks_n16384_d3"


def _eval(profile, coeffs, x_val, strategy="horner", allow_experimental=False):
    ctx = create_context(profile)
    he = ctx.he
    x = encrypt_vector(he, np.full(he.get_nSlots(), x_val))
    rec = OperationRecorder()
    x = encrypt_vector(he, np.full(he.get_nSlots(), x_val), rec)
    result = evaluate_polynomial_ckks(
        he,
        x,
        np.asarray(coeffs),
        strategy=strategy,
        recorder=rec,
        allow_experimental=allow_experimental,
    )
    dec = decrypt_vector(he, result, 1)[0]
    return dec, rec.summary()


# --- Horner: corrección para grados 3 y 5 ---
def test_horner_degree3():
    coeffs = [0.5, 1.0, -0.25, 0.1]
    dec, _ = _eval(PROFILE_D3, coeffs, 0.7)
    expected = np.polyval(coeffs[::-1], 0.7)
    assert abs(dec - expected) < 1e-3


def test_horner_degree5():
    coeffs = [0.5, 1.0, -0.25, 0.1, 0.05, -0.02]
    dec, _ = _eval(PROFILE_D5, coeffs, 0.7)
    expected = np.polyval(coeffs[::-1], 0.7)
    assert abs(dec - expected) < 1e-3


def test_horner_clear_ckks_equivalence():
    """El polinomio cifrado debe coincidir con el claro (error CKKS pequeño)."""
    coeffs = [0.3, -0.5, 0.2, 0.15, -0.05]
    for x_val in (-2.0, 0.0, 1.5, 3.0):
        dec, _ = _eval(PROFILE_D5, coeffs, x_val)
        expected = np.polyval(coeffs[::-1], x_val)
        assert abs(dec - expected) < 1e-2, f"x={x_val}: {dec} vs {expected}"


# --- Consumo de niveles ---
def test_horner_degree3_consumes_3_levels():
    coeffs = [0.5, 1.0, -0.25, 0.1]
    _, summ = _eval(PROFILE_D3, coeffs, 0.5)
    assert summ["levels_consumed"] == 3


def test_horner_degree5_consumes_5_levels():
    coeffs = [0.5, 1.0, -0.25, 0.1, 0.05, -0.02]
    _, summ = _eval(PROFILE_D5, coeffs, 0.5)
    assert summ["levels_consumed"] == 5


# --- Coeficientes nulos ---
def test_horner_handles_zero_coefficients():
    """Un polinomio con coeficientes cero se evalúa correctamente."""
    coeffs = [1.0, 0.0, 0.5, 0.0]  # 1 + 0.5x^2
    dec, _ = _eval(PROFILE_D3, coeffs, 2.0)
    expected = np.polyval(coeffs[::-1], 2.0)  # 1 + 0.5*4 = 3.0
    assert abs(dec - expected) < 1e-3


# --- Gate de estrategias experimentales ---
def test_experimental_strategy_requires_flag():
    """power_basis (experimental) debe rechazarse sin allow_experimental."""
    coeffs = [0.5, 1.0, -0.25, 0.1]
    ctx = create_context(PROFILE_D3)
    he = ctx.he
    x = encrypt_vector(he, np.full(he.get_nSlots(), 0.7))
    with pytest.raises(ValueError, match="experimental"):
        evaluate_polynomial_ckks(he, x, np.asarray(coeffs), strategy="power_basis")


def test_horner_is_official():
    assert "horner" in OFFICIAL_STRATEGIES
    assert "power_basis" not in OFFICIAL_STRATEGIES


def test_unknown_strategy_rejected():
    coeffs = [0.5, 1.0]
    ctx = create_context(PROFILE_D3)
    he = ctx.he
    x = encrypt_vector(he, np.full(he.get_nSlots(), 0.7))
    with pytest.raises(ValueError):
        evaluate_polynomial_ckks(he, x, np.asarray(coeffs), strategy="nonexistent")


# --- Grado 7: documentado como no factible con Horner ---
def test_degree7_horner_infeasible():
    """Grado 7 con Horner agota la cadena (scale out of bounds).

    Documenta el veredicto: no factible bajo n16384_c8. El fallo es a nivel
    SEAL (ValueError), capturado aquí para confirmar el comportamiento esperado.
    """
    coeffs = [0.5, 1.0, -0.25, 0.1, 0.05, -0.02, 0.01, 0.005]  # grado 7
    ctx = create_context(PROFILE_D5)  # profundidad 6, insuficiente para 7
    he = ctx.he
    x = encrypt_vector(he, np.full(he.get_nSlots(), 0.7))
    with pytest.raises(Exception):  # scale out of bounds (nivel SEAL)
        result = evaluate_polynomial_ckks(he, x, np.asarray(coeffs), strategy="horner")
        decrypt_vector(he, result, 1)
