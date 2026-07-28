"""
Fábrica de CNN polinómica (Hito 3C).

Construye una ReducedLeNet con sus tres ReLU sustituidas por aproximaciones
polinómicas (una terna coherente act1/act2/act3 del mismo método, grado e
intervalo lógico). Los pesos del backbone (conv1, conv2, fc1, fc2) NO se
modifican; solo se reemplazan las activaciones.

La fábrica valida estructura (terna coherente, coeficientes finitos, pesos
congelados) pero NO ejecuta datos: la no-finitud en cascada (p. ej. Taylor
grado 9) se detecta durante la inferencia en el benchmark, no aquí.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from ckks_benchmark.approximation.base import PolynomialApproximation
from ckks_benchmark.model.polynomial_activation import PolynomialActivation
from ckks_benchmark.model.preactivations import load_trained_model

EXPECTED_ACTIVATIONS = ("act1", "act2", "act3")
BACKBONE_LAYERS = ("conv1", "conv2", "fc1", "fc2")


@dataclass(frozen=True)
class PolynomialModelConfiguration:
    """Identifica una configuración de CNN polinómica (una de las 24 ternas)."""

    method: str
    degree: int
    interval_name: str
    approximations: Mapping[str, PolynomialApproximation]

    @property
    def identifier(self) -> str:
        return f"{self.method}_d{self.degree}_{self.interval_name}"


def validate_polynomial_triplet(
    approximations: Mapping[str, PolynomialApproximation],
) -> tuple[str, int, str]:
    """Valida que la terna sea coherente y devuelve (method, degree, interval_name).

    Exige exactamente act1/act2/act3, con el mismo método, grado e intervalo
    lógico. Los límites numéricos pueden diferir entre activaciones (cada una
    tiene su propio intervalo); lo que debe coincidir es el nombre lógico.
    """
    if set(approximations) != set(EXPECTED_ACTIVATIONS):
        raise ValueError(
            f"La terna debe contener exactamente {EXPECTED_ACTIVATIONS}, "
            f"pero se recibió {sorted(approximations)}."
        )

    signatures = {
        (approx.method, approx.degree, approx.interval_name) for approx in approximations.values()
    }
    if len(signatures) != 1:
        raise ValueError(
            f"Terna incoherente: se esperaba un único (método, grado, intervalo), "
            f"pero se encontraron {signatures}."
        )

    # Coeficientes finitos en las tres.
    for act_name, approx in approximations.items():
        import numpy as np

        if not np.all(np.isfinite(approx.coefficients)):
            raise ValueError(f"Coeficientes no finitos en {act_name}.")

    method, degree, interval_name = next(iter(signatures))
    return method, degree, interval_name


def replace_relu_activations(
    model: nn.Module,
    approximations: Mapping[str, PolynomialApproximation],
    *,
    evaluation_dtype: torch.dtype | None = None,
) -> nn.Module:
    """Sustituye act1/act2/act3 del modelo por activaciones polinómicas (in place)."""
    for act_name in EXPECTED_ACTIVATIONS:
        approx = approximations[act_name]
        setattr(
            model,
            act_name,
            PolynomialActivation(
                approx.coefficients,
                evaluation_dtype=evaluation_dtype,
                metadata={
                    "activation": act_name,
                    "method": approx.method,
                    "degree": approx.degree,
                    "interval_name": approx.interval_name,
                },
            ),
        )
    return model


def build_polynomial_model(
    checkpoint_path: str | Path,
    approximations: Mapping[str, PolynomialApproximation],
    *,
    evaluation_dtype: torch.dtype | None = None,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Construye una CNN polinómica desde el checkpoint ReLU y una terna.

    Flujo: carga ReLU -> valida terna -> copia independiente -> sustituye
    activaciones -> eval() -> verifica pesos congelados. No ejecuta datos.
    """
    validate_polynomial_triplet(approximations)

    # Modelo de referencia (ReLU) desde el checkpoint.
    reference_model, _ckpt = load_trained_model(Path(checkpoint_path), torch.device(device))

    # Copia independiente para no modificar la referencia.
    poly_model = copy.deepcopy(reference_model)
    replace_relu_activations(poly_model, approximations, evaluation_dtype=evaluation_dtype)
    poly_model.to(device)
    poly_model.eval()

    # Verificación de pesos congelados (bit a bit).
    verify_backbone_weights_unchanged(reference_model, poly_model)

    return poly_model


def verify_backbone_weights_unchanged(
    reference_model: nn.Module,
    polynomial_model: nn.Module,
) -> None:
    """Verifica que los parámetros del backbone sean exactamente iguales.

    Compara todos los parámetros con nombre: nombres, formas, dtype, valores
    exactos (torch.equal) y requires_grad. Lanza RuntimeError ante cualquier
    discrepancia.
    """
    ref_params = dict(reference_model.named_parameters())
    poly_params = dict(polynomial_model.named_parameters())

    if set(ref_params) != set(poly_params):
        raise RuntimeError(
            f"Los conjuntos de parámetros difieren: "
            f"ref={sorted(ref_params)} vs poly={sorted(poly_params)}."
        )

    for name, ref_param in ref_params.items():
        cand = poly_params[name]
        if ref_param.shape != cand.shape:
            raise RuntimeError(f"Forma cambiada en '{name}'.")
        if ref_param.dtype != cand.dtype:
            raise RuntimeError(f"dtype cambiado en '{name}'.")
        if not torch.equal(ref_param, cand):
            raise RuntimeError(f"Valores cambiados en '{name}' tras la sustitución.")
        if ref_param.requires_grad != cand.requires_grad:
            raise RuntimeError(f"requires_grad cambiado en '{name}'.")


def count_trainable_parameters(model: nn.Module) -> int:
    """Cuenta parámetros entrenables (los buffers de coeficientes no cuentan)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
