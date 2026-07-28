"""
Pruebas unitarias del módulo de evaluación funcional (Hito 3B).

Cubren: MAE/RMSE/error máximo con valores conocidos, máscaras de intervalo,
arreglos vacíos, detección de no-finitud, reproducibilidad, e integración
sobre una configuración completa.
"""

from __future__ import annotations

import numpy as np
import pytest

from ckks_benchmark.approximation.chebyshev import fit as chebyshev_fit
from ckks_benchmark.approximation.evaluation import (
    evaluate_configuration,
    max_error_with_location,
    mean_absolute_error,
    relu_reference,
    root_mean_squared_error,
)
from ckks_benchmark.approximation.taylor import fit as taylor_fit

INTERVAL = (-6.6096, 7.2350)


# --- MAE / RMSE con valores conocidos ---
def test_mae_known():
    expected = np.array([0.0, 1.0, 2.0])
    observed = np.array([0.0, 2.0, 0.0])  # errores: 0, 1, 2
    assert mean_absolute_error(expected, observed) == pytest.approx(1.0)


def test_rmse_known():
    expected = np.array([0.0, 0.0])
    observed = np.array([3.0, 4.0])  # errores 3, 4 -> sqrt((9+16)/2)=sqrt(12.5)
    assert root_mean_squared_error(expected, observed) == pytest.approx(np.sqrt(12.5))


def test_mae_zero_when_equal():
    x = np.array([1.0, -2.0, 3.5])
    assert mean_absolute_error(x, x) == 0.0


# --- Error máximo y su ubicación ---
def test_max_error_location():
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    expected = np.array([0.0, 0.0, 0.0, 0.0])
    observed = np.array([0.1, 0.0, 0.0, 5.0])  # máximo error en x=2.0
    emax, x_at = max_error_with_location(expected, observed, x)
    assert emax == pytest.approx(5.0)
    assert x_at == pytest.approx(2.0)


# --- Arreglos vacíos y formas incompatibles ---
def test_empty_array_raises():
    with pytest.raises(ValueError):
        mean_absolute_error(np.array([]), np.array([]))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mean_absolute_error(np.array([1.0, 2.0]), np.array([1.0]))


# --- ReLU de referencia ---
def test_relu_reference():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 3.0])
    expected = np.array([0.0, 0.0, 0.0, 0.5, 3.0])
    assert np.allclose(relu_reference(x), expected)


# --- Detección de no-finitud: config válida vs inválida ---
def test_valid_configuration():
    p = chebyshev_fit(5, INTERVAL, "act1", "I1")
    samples = np.linspace(INTERVAL[0], INTERVAL[1], 1000)
    m = evaluate_configuration(p, samples)
    assert m.valid is True
    assert m.invalid_reason is None
    assert np.isfinite(m.mae_uniform)
    assert np.isfinite(m.mae_empirical)


def test_empty_samples_invalid():
    p = chebyshev_fit(5, INTERVAL, "act1", "I1")
    m = evaluate_configuration(p, np.array([]))
    assert m.valid is False
    assert m.invalid_reason is not None


# --- Origen: la ventana captura muestras ---
def test_origin_window_counts():
    p = chebyshev_fit(5, INTERVAL, "act1", "I1")
    # Muestras: la mitad en la ventana origen, la mitad fuera.
    samples = np.concatenate(
        [
            np.linspace(-0.4, 0.4, 500),  # dentro de [-0.5, 0.5]
            np.linspace(3.0, 6.0, 500),  # fuera
        ]
    )
    m = evaluate_configuration(p, samples)
    assert m.origin_empirical_sample_count == 500


# --- Taylor con error alto sigue siendo válido (no inviable numéricamente) ---
def test_high_error_taylor_still_valid():
    # Taylor grado 9 en un intervalo amplio: error enorme pero finito.
    p = taylor_fit(9, (-24.8339, 12.2336), "act2", "I2")
    samples = np.linspace(-24.0, 12.0, 5000)
    m = evaluate_configuration(p, samples)
    assert m.valid is True  # finito -> válido
    assert m.mae_empirical > 1.0  # error alto, como esperado


# --- Reproducibilidad ---
def test_reproducible_metrics():
    p = chebyshev_fit(5, INTERVAL, "act1", "I1")
    samples = np.linspace(INTERVAL[0], INTERVAL[1], 1000)
    m1 = evaluate_configuration(p, samples)
    m2 = evaluate_configuration(p, samples)
    assert m1.mae_uniform == m2.mae_uniform
    assert m1.mae_empirical == m2.mae_empirical
