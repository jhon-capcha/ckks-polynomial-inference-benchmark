"""
Aproximación de ReLU por mínimos cuadrados discretos (Hito 3).

Ajuste L2 discreto NO ponderado sobre una malla uniforme del intervalo. Esto lo
distingue de Chebyshev (proyección L2 ponderada con 1/sqrt(1-t^2)):

    Least squares  -> minimiza sum (ReLU(x_i) - p(x_i))^2 sobre malla uniforme
    Chebyshev      -> proyección con peso que enfatiza los extremos

Se usa numpy.linalg.lstsq (descomposición estable, no la ecuación normal
(X^T X)^{-1} que es numéricamente peligrosa).
"""

from __future__ import annotations

import numpy as np

from ckks_benchmark.approximation.base import (
    PolynomialApproximation,
    relu_reference,
    validate_degree,
    validate_interval,
)

# Número de puntos de la malla de ajuste (fijo para todas las configuraciones).
LSQ_FIT_POINTS = 1000


def _least_squares_coefficients(
    degree: int, interval: tuple[float, float], n_points: int
) -> tuple[np.ndarray, dict]:
    """Coeficientes monomiales [a0..ad] por mínimos cuadrados sobre malla uniforme.

    Devuelve (coeficientes, diagnósticos) donde diagnósticos incluye el residuo
    y el número de condición de la matriz de diseño.
    """
    a, b = interval
    x = np.linspace(a, b, n_points, dtype=np.float64)
    y = relu_reference(x)

    # Matriz de diseño de Vandermonde: columnas [1, x, x^2, ..., x^degree].
    design = np.vander(x, N=degree + 1, increasing=True)

    # Solución estable por lstsq (usa SVD internamente).
    coeffs, residuals, rank, singular_values = np.linalg.lstsq(design, y, rcond=None)

    cond_number = (
        float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else float("inf")
    )
    residual = float(residuals[0]) if residuals.size else float("nan")

    diagnostics = {
        "fit_points": n_points,
        "matrix_rank": int(rank),
        "condition_number": cond_number,
        "residual": residual,
    }
    return coeffs.astype(np.float64), diagnostics


def fit(
    degree: int,
    interval: tuple[float, float],
    activation: str = "",
    interval_name: str = "",
    n_points: int = LSQ_FIT_POINTS,
) -> PolynomialApproximation:
    """Construye la aproximación por mínimos cuadrados de ReLU.

    Args:
        degree: Grado nominal (uno de ALLOWED_DEGREES).
        interval: Intervalo (a, b) de aproximación.
        activation: Nombre de la activación.
        interval_name: Nombre de la política de intervalo.
        n_points: Puntos de la malla de ajuste (fijo en LSQ_FIT_POINTS).

    Returns:
        PolynomialApproximation con coeficientes en base monomial.
    """
    validate_degree(degree)
    validate_interval(interval)

    coefficients, diagnostics = _least_squares_coefficients(degree, interval, n_points)

    metadata = {
        "construction": "discrete_least_squares",
        "weight": "uniform",
        "solver": "numpy.linalg.lstsq",
        **diagnostics,
        "note": "Ajuste L2 uniforme. Distinto de Chebyshev (proyección ponderada).",
    }

    return PolynomialApproximation(
        method="least_squares",
        activation=activation,
        interval_name=interval_name,
        interval=interval,
        degree=degree,
        coefficients=coefficients,
        coefficient_basis="monomial",
        metadata=metadata,
    )
