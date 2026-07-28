"""
Aproximación de ReLU por serie de Taylor de Softplus (Hito 3).

Taylor es el baseline deliberadamente desfavorable del proyecto. ReLU no es
diferenciable en 0 y no tiene serie de Taylor clásica; se construye sobre una
versión suavizada (Softplus) y se expande en x0=0.

Decisiones congeladas (ver docs/metodologia/definicion_taylor.md):
    - Función suavizada: Softplus_beta(x) = (1/beta) * ln(1 + exp(beta*x))
    - Punto de expansión: x0 = 0
    - beta = 1 (seleccionado empíricamente: menor error y coeficientes más
      estables sobre los intervalos evaluados; ver piloto Hito 3A)
    - Derivación simbólica con SymPy (evita errores algebraicos manuales)

El error de Taylor se mide contra ReLU (no contra Softplus). Se conservan los
coeficientes nulos para mantener el grado nominal y la trazabilidad.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from ckks_benchmark.approximation.base import (
    PolynomialApproximation,
    validate_degree,
    validate_interval,
)

# Parámetro de suavización congelado (piloto Hito 3A).
TAYLOR_BETA = 1.0
TAYLOR_EXPANSION_POINT = 0.0


def _taylor_softplus_coefficients(beta: float, degree: int) -> np.ndarray:
    """Coeficientes monomiales [a0..ad] de la serie de Taylor de Softplus_beta en 0.

    Usa SymPy para la derivación simbólica. Conserva ceros explícitos para
    mantener longitud degree+1.
    """
    x = sp.symbols("x", real=True)
    softplus = sp.log(1 + sp.exp(beta * x)) / beta
    series = sp.series(softplus, x, TAYLOR_EXPANSION_POINT, degree + 1).removeO()
    poly = sp.Poly(series, x)

    # all_coeffs() devuelve [ad..a0]; invertir a [a0..ad].
    coeffs_low_to_high = list(reversed(poly.all_coeffs()))
    while len(coeffs_low_to_high) < degree + 1:
        coeffs_low_to_high.append(sp.Integer(0))

    return np.array([float(c) for c in coeffs_low_to_high[: degree + 1]], dtype=np.float64)


def fit(
    degree: int,
    interval: tuple[float, float],
    activation: str = "",
    interval_name: str = "",
    beta: float = TAYLOR_BETA,
) -> PolynomialApproximation:
    """Construye la aproximación de Taylor-Softplus para un grado e intervalo.

    Args:
        degree: Grado nominal (uno de ALLOWED_DEGREES).
        interval: Intervalo (a, b) donde se usará el polinomio. No afecta los
            coeficientes (Taylor es local en x0=0), pero se registra en la
            metadata y se usa para la evaluación del error.
        activation: Nombre de la activación ("act1", etc.), para el identificador.
        interval_name: Nombre de la política de intervalo ("I1", "I2").
        beta: Parámetro de suavización de Softplus (congelado en 1.0).

    Returns:
        PolynomialApproximation con coeficientes en base monomial.
    """
    validate_degree(degree)
    validate_interval(interval)

    coefficients = _taylor_softplus_coefficients(beta, degree)

    metadata = {
        "smoothing_function": "softplus",
        "beta": beta,
        "expansion_point": TAYLOR_EXPANSION_POINT,
        "construction": "sympy_symbolic_series",
        "note": (
            "Baseline local desfavorable. El error se mide contra ReLU, no "
            "contra Softplus. Coeficientes nulos conservados."
        ),
    }

    return PolynomialApproximation(
        method="taylor",
        activation=activation,
        interval_name=interval_name,
        interval=interval,
        degree=degree,
        coefficients=coefficients,
        coefficient_basis="monomial",
        metadata=metadata,
    )
