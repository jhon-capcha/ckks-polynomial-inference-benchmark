"""
Interfaz común para las aproximaciones polinómicas (Hito 3).

Define la representación única de un polinomio de aproximación y las utilidades
compartidas por los tres métodos (Taylor, Chebyshev, mínimos cuadrados):

- PolynomialApproximation: estructura inmutable que almacena coeficientes en
  base monomial [a0, a1, ..., ad] junto con su metadata (método, activación,
  intervalo, grado).
- Validación de entradas (grados permitidos, intervalos válidos, coeficientes
  finitos).
- Evaluación en texto plano por Horner, para medir el error funcional.

Todos los métodos exportan coeficientes en la MISMA base monomial:
    p(x) = a0 + a1*x + a2*x^2 + ... + ad*x^d
almacenados como [a0, a1, ..., ad] (índice = potencia). Esto permite comparar
y evaluar los tres métodos con la misma estrategia en los hitos posteriores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Grados permitidos por el diseño experimental del proyecto.
ALLOWED_DEGREES = (3, 5, 7, 9)

# Tolerancia para considerar un coeficiente como efectivamente nulo.
ZERO_COEFFICIENT_TOL = 1e-12


@dataclass(frozen=True)
class PolynomialApproximation:
    """Representación inmutable de un polinomio de aproximación de ReLU.

    Los coeficientes se almacenan en base monomial, ordenados de menor a mayor
    potencia: coefficients[k] es el coeficiente de x^k. La longitud es degree+1
    (incluye ceros explícitos para términos ausentes).

    Attributes:
        method: Nombre del método ("taylor", "chebyshev", "least_squares").
        activation: Activación a la que se aplica ("act1", "act2", "act3").
        interval_name: Nombre de la política de intervalo ("I1", "I2").
        interval: Límites (a, b) del intervalo de aproximación, con a < b.
        degree: Grado nominal solicitado (uno de ALLOWED_DEGREES).
        coefficients: Array float64 de longitud degree+1, base monomial [a0..ad].
        coefficient_basis: Base de los coeficientes (siempre "monomial").
        metadata: Información adicional específica del método.
    """

    method: str
    activation: str
    interval_name: str
    interval: tuple[float, float]
    degree: int
    coefficients: np.ndarray
    coefficient_basis: str = "monomial"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validaciones estructurales al construir.
        validate_degree(self.degree)
        validate_interval(self.interval)
        validate_coefficients(self.coefficients, self.degree)
        if self.coefficient_basis != "monomial":
            raise ValueError(
                f"coefficient_basis debe ser 'monomial', no '{self.coefficient_basis}'."
            )

    @property
    def effective_degree(self) -> int:
        """Mayor potencia con coeficiente no despreciable.

        Puede ser menor que el grado nominal si los coeficientes de orden alto
        se anulan (relevante para la profundidad multiplicativa en CKKS).
        """
        nonzero = np.nonzero(np.abs(self.coefficients) > ZERO_COEFFICIENT_TOL)[0]
        return int(nonzero[-1]) if nonzero.size else 0

    @property
    def max_abs_coefficient(self) -> float:
        """Mayor magnitud entre los coeficientes (diagnóstico de escala CKKS)."""
        return float(np.max(np.abs(self.coefficients)))

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evalúa el polinomio en x usando el esquema de Horner."""
        return evaluate_monomial(self.coefficients, x)

    def identifier(self) -> str:
        """Identificador único: p.ej. 'act1_taylor_d5_I1'."""
        return f"{self.activation}_{self.method}_d{self.degree}_{self.interval_name}"


def validate_degree(degree: int) -> None:
    """Verifica que el grado esté entre los permitidos por el diseño."""
    if degree not in ALLOWED_DEGREES:
        raise ValueError(f"Grado {degree} no permitido. Debe ser uno de {ALLOWED_DEGREES}.")


def validate_interval(interval: tuple[float, float]) -> None:
    """Verifica que el intervalo (a, b) sea válido (finito, con a < b)."""
    if len(interval) != 2:
        raise ValueError(f"El intervalo debe tener dos elementos, no {len(interval)}.")
    a, b = interval
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError(f"Los límites del intervalo deben ser finitos: ({a}, {b}).")
    if a >= b:
        raise ValueError(f"El intervalo requiere a < b, pero se recibió ({a}, {b}).")


def validate_coefficients(coefficients: np.ndarray, degree: int) -> None:
    """Verifica que los coeficientes sean finitos y tengan la longitud correcta."""
    if not isinstance(coefficients, np.ndarray):
        raise TypeError("Los coeficientes deben ser un np.ndarray.")
    if coefficients.ndim != 1:
        raise ValueError(
            f"Los coeficientes deben ser 1-D, pero tienen {coefficients.ndim} dimensiones."
        )
    if coefficients.shape[0] != degree + 1:
        raise ValueError(
            f"Se esperaban {degree + 1} coeficientes (grado {degree}), "
            f"pero se recibieron {coefficients.shape[0]}."
        )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("Los coeficientes contienen NaN o infinito.")


def evaluate_monomial(coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evalúa un polinomio en base monomial [a0..ad] por el esquema de Horner.

    Horner es numéricamente más estable y eficiente que la suma directa de
    potencias: p(x) = a0 + x*(a1 + x*(a2 + ... )).

    Args:
        coefficients: Array [a0, a1, ..., ad] (índice = potencia).
        x: Puntos donde evaluar (array float64).

    Returns:
        p(x) evaluado en cada punto de x.
    """
    coefficients = np.asarray(coefficients, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    for coeff in reversed(coefficients):
        result = result * x + coeff
    return result


def relu_reference(x: np.ndarray) -> np.ndarray:
    """Función objetivo de referencia en float64: ReLU(x) = max(0, x)."""
    return np.maximum(np.asarray(x, dtype=np.float64), 0.0)
