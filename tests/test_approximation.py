"""
Pruebas unitarias de las aproximaciones polinómicas (Hito 3A).

Cubren:
    - Interfaz común (base): validación de grados, intervalos, coeficientes.
    - Los tres métodos: generan d+1 coeficientes finitos en base monomial.
    - Reproducibilidad: misma entrada -> mismos coeficientes.
    - Chebyshev: aproxima ReLU con error razonable.
    - Least squares: residuo finito, matriz de rango completo.
    - Taylor: coeficientes conocidos para beta=1, grado 5.
    - Evaluación: Horner coincide con evaluación directa.
"""

from __future__ import annotations

import numpy as np
import pytest

from ckks_benchmark.approximation import chebyshev, least_squares, taylor
from ckks_benchmark.approximation.base import (
    ALLOWED_DEGREES,
    PolynomialApproximation,
    evaluate_monomial,
    relu_reference,
    validate_degree,
    validate_interval,
)

INTERVAL = (-6.6096, 7.2350)  # act1 I1
ALL_METHODS = (taylor.fit, chebyshev.fit, least_squares.fit)


# --- Validación de la interfaz común ---
def test_validate_degree_accepts_allowed():
    for d in ALLOWED_DEGREES:
        validate_degree(d)  # no debe lanzar


@pytest.mark.parametrize("bad_degree", [0, 1, 2, 4, 6, 8, 10, -1])
def test_validate_degree_rejects_invalid(bad_degree):
    with pytest.raises(ValueError):
        validate_degree(bad_degree)


def test_validate_interval_rejects_reversed():
    with pytest.raises(ValueError):
        validate_interval((5.0, 1.0))  # a >= b


def test_validate_interval_rejects_infinite():
    with pytest.raises(ValueError):
        validate_interval((0.0, np.inf))


# --- Propiedades comunes a los tres métodos ---
@pytest.mark.parametrize("fit_fn", ALL_METHODS)
@pytest.mark.parametrize("degree", ALLOWED_DEGREES)
def test_method_produces_correct_length(fit_fn, degree):
    poly = fit_fn(degree=degree, interval=INTERVAL)
    assert poly.coefficients.shape[0] == degree + 1


@pytest.mark.parametrize("fit_fn", ALL_METHODS)
@pytest.mark.parametrize("degree", ALLOWED_DEGREES)
def test_method_coefficients_finite(fit_fn, degree):
    poly = fit_fn(degree=degree, interval=INTERVAL)
    assert np.all(np.isfinite(poly.coefficients))


@pytest.mark.parametrize("fit_fn", ALL_METHODS)
def test_method_uses_monomial_basis(fit_fn):
    poly = fit_fn(degree=5, interval=INTERVAL)
    assert poly.coefficient_basis == "monomial"


@pytest.mark.parametrize("fit_fn", ALL_METHODS)
def test_method_reproducible(fit_fn):
    p1 = fit_fn(degree=5, interval=INTERVAL)
    p2 = fit_fn(degree=5, interval=INTERVAL)
    assert np.allclose(p1.coefficients, p2.coefficients, rtol=0, atol=1e-15)


# --- Taylor: coeficientes conocidos (beta=1, grado 5) ---
def test_taylor_known_coefficients():
    poly = taylor.fit(degree=5, interval=INTERVAL)
    # Serie de Softplus_1 en 0: [ln2, 1/2, 1/8, 0, -1/192, 0]
    expected = np.array([np.log(2), 0.5, 0.125, 0.0, -1.0 / 192.0, 0.0])
    assert np.allclose(poly.coefficients, expected, atol=1e-6)


def test_taylor_effective_degree_below_nominal():
    poly = taylor.fit(degree=5, interval=INTERVAL)
    assert poly.effective_degree == 4  # a5 = 0


# --- Chebyshev: aproxima ReLU razonablemente ---
def test_chebyshev_approximates_relu():
    poly = chebyshev.fit(degree=9, interval=INTERVAL)
    x = np.linspace(INTERVAL[0], INTERVAL[1], 5000)
    mae = np.mean(np.abs(relu_reference(x) - poly.evaluate(x)))
    assert mae < 0.1  # grado 9 debe aproximar bien


# --- Least squares: diagnósticos válidos ---
def test_least_squares_full_rank():
    poly = least_squares.fit(degree=5, interval=INTERVAL)
    assert poly.metadata["matrix_rank"] == 6  # degree + 1


def test_least_squares_finite_residual():
    poly = least_squares.fit(degree=5, interval=INTERVAL)
    assert np.isfinite(poly.metadata["residual"])


# --- Evaluación: Horner coincide con suma directa ---
def test_horner_matches_direct_evaluation():
    coeffs = np.array([1.0, -2.0, 0.5, 3.0])
    x = np.array([-1.0, 0.0, 1.0, 2.5])
    horner = evaluate_monomial(coeffs, x)
    direct = coeffs[0] + coeffs[1] * x + coeffs[2] * x**2 + coeffs[3] * x**3
    assert np.allclose(horner, direct)


# --- La estructura rechaza polinomios mal formados ---
def test_polynomial_rejects_wrong_coefficient_length():
    with pytest.raises(ValueError):
        PolynomialApproximation(
            method="test",
            activation="act1",
            interval_name="I1",
            interval=INTERVAL,
            degree=5,
            coefficients=np.array([1.0, 2.0]),  # solo 2, se esperan 6
        )
