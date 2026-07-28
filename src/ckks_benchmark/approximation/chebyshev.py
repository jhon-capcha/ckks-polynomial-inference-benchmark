"""
Aproximación de ReLU por proyección truncada en serie de Chebyshev (Hito 3).

Chebyshev se construye por PROYECCIÓN L2 ponderada con el peso característico
w(t) = 1/sqrt(1 - t^2), NO por interpolación en nodos ni por ajuste de mínimos
cuadrados uniforme. Esto lo distingue matemáticamente de least_squares:

    Chebyshev      -> proyección L2 con peso 1/sqrt(1-t^2) (pondera extremos)
    Least squares  -> ajuste L2 discreto uniforme (sin peso)

Los coeficientes de Chebyshev se calculan por cuadratura de Gauss-Chebyshev
sobre la función ReLU reescalada al dominio [-1, 1], y luego el polinomio se
convierte a base monomial en el dominio original [a, b].
"""

from __future__ import annotations

import numpy as np

from ckks_benchmark.approximation.base import (
    PolynomialApproximation,
    relu_reference,
    validate_degree,
    validate_interval,
)


def _chebyshev_projection_coefficients(degree: int, n_quadrature: int = 2048) -> np.ndarray:
    """Coeficientes c_k de la proyección de ReLU en [-1,1] sobre T_0..T_degree.

    Usa cuadratura de Gauss-Chebyshev: los nodos t_j = cos((j+0.5)*pi/N) y el
    peso implícito 1/sqrt(1-t^2) hacen que la integral
        c_k = (2/pi) * integral_{-1}^{1} f(t) T_k(t) / sqrt(1-t^2) dt
    se reduzca a un promedio simple sobre los nodos:
        c_k = (2/N) * sum_j f(t_j) cos(k * theta_j),  theta_j=(j+0.5)*pi/N
    con c_0 llevando un factor 1/2.

    Aquí f es ReLU evaluada en [-1,1] (se reescala al dominio real fuera).
    """
    j = np.arange(n_quadrature)
    theta = (j + 0.5) * np.pi / n_quadrature
    nodes = np.cos(theta)  # nodos en [-1, 1]

    # ReLU en [-1,1] (el reescalado al dominio real se aplica al construir).
    # Nota: aquí f se define sobre [-1,1] directamente; el mapeo se hace en fit.
    return theta, nodes  # se usa en fit para evaluar f correctamente


def _project_relu_on_interval(
    degree: int, interval: tuple[float, float], n_quadrature: int = 2048
) -> np.ndarray:
    """Coeficientes monomiales [a0..ad] de la proyección de Chebyshev de ReLU en [a,b]."""
    a, b = interval
    j = np.arange(n_quadrature)
    theta = (j + 0.5) * np.pi / n_quadrature
    t_nodes = np.cos(theta)  # nodos Gauss-Chebyshev en [-1, 1]

    # Mapear nodos de [-1,1] al dominio real [a,b] y evaluar ReLU.
    x_nodes = 0.5 * (b - a) * t_nodes + 0.5 * (a + b)
    f_values = relu_reference(x_nodes)

    # Coeficientes de Chebyshev por cuadratura.
    cheb_coeffs = np.zeros(degree + 1, dtype=np.float64)
    for k in range(degree + 1):
        cheb_coeffs[k] = (2.0 / n_quadrature) * np.sum(f_values * np.cos(k * theta))
    cheb_coeffs[0] *= 0.5  # c_0 lleva factor 1/2

    # Convertir de serie Chebyshev (en t) a monomial (en t), luego a monomial (en x).
    # numpy.polynomial.chebyshev.cheb2poly da coeficientes monomiales en t.
    from numpy.polynomial import chebyshev as C

    mono_in_t = C.cheb2poly(cheb_coeffs)  # [b0..bd] en base t, dominio [-1,1]

    # Componer con el cambio de variable t = (2x - (a+b)) / (b - a).
    # t = alpha*x + beta, con:
    alpha = 2.0 / (b - a)
    beta = -(a + b) / (b - a)

    # Evaluar el polinomio en t como polinomio en x mediante composición.
    # p(x) = sum_i mono_in_t[i] * (alpha*x + beta)^i
    from numpy.polynomial import polynomial as P

    # Polinomio (alpha*x + beta) en base monomial de x: [beta, alpha]
    linear = np.array([beta, alpha], dtype=np.float64)
    result = np.zeros(1, dtype=np.float64)
    power = np.array([1.0], dtype=np.float64)  # (linear)^0
    for coeff in mono_in_t:
        result = P.polyadd(result, coeff * power)
        power = P.polymul(power, linear)

    # Asegurar longitud degree+1.
    mono_in_x = np.zeros(degree + 1, dtype=np.float64)
    mono_in_x[: len(result)] = result[: degree + 1]
    return mono_in_x


def fit(
    degree: int,
    interval: tuple[float, float],
    activation: str = "",
    interval_name: str = "",
    n_quadrature: int = 2048,
) -> PolynomialApproximation:
    """Construye la aproximación de Chebyshev (proyección ponderada) de ReLU.

    Args:
        degree: Grado nominal (uno de ALLOWED_DEGREES).
        interval: Intervalo (a, b) de aproximación.
        activation: Nombre de la activación.
        interval_name: Nombre de la política de intervalo.
        n_quadrature: Número de nodos Gauss-Chebyshev para la cuadratura.

    Returns:
        PolynomialApproximation con coeficientes en base monomial.
    """
    validate_degree(degree)
    validate_interval(interval)

    coefficients = _project_relu_on_interval(degree, interval, n_quadrature)

    metadata = {
        "construction": "chebyshev_projection",
        "weight": "1/sqrt(1-t^2)",
        "quadrature": "gauss_chebyshev",
        "n_quadrature": n_quadrature,
        "note": "Proyección L2 ponderada. Distinto de LSQ (ajuste uniforme sin peso).",
    }

    return PolynomialApproximation(
        method="chebyshev",
        activation=activation,
        interval_name=interval_name,
        interval=interval,
        degree=degree,
        coefficients=coefficients,
        coefficient_basis="monomial",
        metadata=metadata,
    )
