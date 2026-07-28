"""
Activación polinómica para PyTorch (Hito 3C).

Sustituye una ReLU por un polinomio de aproximación (base monomial, evaluación
por Horner). Los coeficientes se registran como buffer (no entrenables).

VERSIÓN MÍNIMA CONFIGURABLE para el piloto 3C-P1 de sensibilidad al dtype:
    - evaluation_dtype = None   -> evalúa en el dtype de la entrada (Modo A si float32)
    - evaluation_dtype = float64 -> evalúa el polinomio en float64 y devuelve al
                                    dtype original (Modo B: precisión ampliada del
                                    polinomio dentro de una CNN float32)

Modo C (CNN completa en float64) se logra externamente convirtiendo todo el
modelo y las entradas a float64; esta clase no lo fuerza.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class PolynomialActivation(nn.Module):
    """Evalúa un polinomio en base monomial [a0..ad] como activación."""

    def __init__(
        self,
        coefficients,
        evaluation_dtype: torch.dtype | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        coeffs = torch.as_tensor(coefficients, dtype=torch.float64)
        if coeffs.ndim != 1:
            raise ValueError(f"Los coeficientes deben ser 1-D, no {coeffs.ndim}-D.")
        if not torch.all(torch.isfinite(coeffs)):
            raise ValueError("Los coeficientes contienen NaN o infinito.")

        # Buffer: se mueve con el modelo, aparece en state_dict, no es entrenable.
        self.register_buffer("coefficients", coeffs)
        self.evaluation_dtype = evaluation_dtype
        self.metadata = metadata or {}

    @property
    def nominal_degree(self) -> int:
        return self.coefficients.shape[0] - 1

    @property
    def effective_degree(self) -> int:
        nonzero = torch.nonzero(torch.abs(self.coefficients) > 1e-12).flatten()
        return int(nonzero[-1].item()) if nonzero.numel() else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype

        # Modo B: evaluar en dtype ampliado (float64) si se especifica.
        eval_dtype = self.evaluation_dtype if self.evaluation_dtype is not None else x.dtype
        x_eval = x.to(dtype=eval_dtype)
        coeffs = self.coefficients.to(device=x.device, dtype=eval_dtype)

        # Horner: p(x) = a0 + x*(a1 + x*(a2 + ...)).
        result = torch.zeros_like(x_eval)
        for coeff in reversed(coeffs):
            result = result * x_eval + coeff

        # Devolver al dtype original de la entrada.
        return result.to(dtype=original_dtype)
