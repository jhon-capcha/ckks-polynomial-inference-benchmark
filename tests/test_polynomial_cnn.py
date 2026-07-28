"""
Pruebas de integración estructural de la CNN polinómica (Hito 3C-A).

Cubren PolynomialActivation (equivalencia con NumPy, dtype, sin entrenables) y
la fábrica polynomial_model (validación de terna, pesos congelados, parámetros,
inferencia mínima, invariancia del modelo base).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from ckks_benchmark.approximation.base import evaluate_monomial
from ckks_benchmark.approximation.registry import (
    generate_all,
    load_and_validate_intervals,
)
from ckks_benchmark.model.polynomial_activation import PolynomialActivation
from ckks_benchmark.model.polynomial_model import (
    build_polynomial_model,
    count_trainable_parameters,
    validate_polynomial_triplet,
    verify_backbone_weights_unchanged,
)
from ckks_benchmark.model.preactivations import load_trained_model

CHECKPOINT = "models/reduced_lenet_relu_best.pt"
INTERVALS = Path("results/published/preactivation_intervals.json")


@pytest.fixture(scope="module")
def polys():
    return generate_all(load_and_validate_intervals(INTERVALS))


def _triplet(polys, method, degree, interval):
    return {a: polys[f"{a}_{method}_d{degree}_{interval}"] for a in ("act1", "act2", "act3")}


# --- PolynomialActivation ---
def test_activation_matches_numpy():
    coeffs = np.array([0.5, 1.0, -0.25, 0.1])
    act = PolynomialActivation(coeffs, evaluation_dtype=torch.float64)
    x = np.linspace(-3, 3, 100)
    torch_out = act(torch.tensor(x, dtype=torch.float64)).numpy()
    numpy_out = evaluate_monomial(coeffs, x)
    assert np.allclose(torch_out, numpy_out, rtol=1e-12, atol=1e-12)


def test_activation_preserves_shape():
    act = PolynomialActivation(np.array([0.0, 1.0, 0.1]))
    for shape in [(4, 6, 28, 28), (4, 16, 10, 10), (4, 120)]:
        x = torch.randn(*shape, dtype=torch.float32)
        assert act(x).shape == x.shape


def test_activation_float32_and_float64():
    act = PolynomialActivation(np.array([0.5, 1.0, 0.1]))
    x32 = torch.randn(50, dtype=torch.float32)
    x64 = torch.randn(50, dtype=torch.float64)
    assert act(x32).dtype == torch.float32
    assert act(x64).dtype == torch.float64


def test_activation_no_trainable_params():
    act = PolynomialActivation(np.array([0.0, 1.0, 0.1]))
    assert count_trainable_parameters(act) == 0


def test_activation_effective_degree():
    # a3 = 0 -> grado efectivo 2 aunque nominal sea 3.
    act = PolynomialActivation(np.array([0.0, 0.5, 0.1, 0.0]))
    assert act.nominal_degree == 3
    assert act.effective_degree == 2


def test_activation_rejects_non_finite():
    with pytest.raises(ValueError):
        PolynomialActivation(np.array([0.0, np.inf, 0.1]))


# --- Validación de terna ---
def test_triplet_valid(polys):
    method, degree, interval = validate_polynomial_triplet(_triplet(polys, "chebyshev", 5, "I1"))
    assert (method, degree, interval) == ("chebyshev", 5, "I1")


def test_triplet_rejects_missing_activation(polys):
    triplet = _triplet(polys, "chebyshev", 5, "I1")
    del triplet["act2"]
    with pytest.raises(ValueError):
        validate_polynomial_triplet(triplet)


def test_triplet_rejects_mixed_methods(polys):
    triplet = _triplet(polys, "chebyshev", 5, "I1")
    triplet["act2"] = polys["act2_least_squares_d5_I1"]  # método distinto
    with pytest.raises(ValueError):
        validate_polynomial_triplet(triplet)


def test_triplet_rejects_mixed_degrees(polys):
    triplet = _triplet(polys, "chebyshev", 5, "I1")
    triplet["act3"] = polys["act3_chebyshev_d7_I1"]  # grado distinto
    with pytest.raises(ValueError):
        validate_polynomial_triplet(triplet)


def test_triplet_rejects_mixed_intervals(polys):
    triplet = _triplet(polys, "chebyshev", 5, "I1")
    triplet["act1"] = polys["act1_chebyshev_d5_I2"]  # intervalo distinto
    with pytest.raises(ValueError):
        validate_polynomial_triplet(triplet)


def test_triplet_allows_different_numeric_bounds(polys):
    # act1-I1, act2-I1, act3-I1 tienen límites numéricos distintos pero mismo I1.
    triplet = _triplet(polys, "chebyshev", 5, "I1")
    b1 = triplet["act1"].interval
    b2 = triplet["act2"].interval
    assert b1 != b2  # límites distintos
    validate_polynomial_triplet(triplet)  # pero la terna es válida


# --- Fábrica de modelos ---
def test_build_freezes_weights(polys):
    triplet = _triplet(polys, "chebyshev", 5, "I1")
    model = build_polynomial_model(CHECKPOINT, triplet)
    # Si build no lanzó, los pesos están congelados. Verificación redundante:
    reference, _ = load_trained_model(Path(CHECKPOINT))
    verify_backbone_weights_unchanged(reference, model)  # no debe lanzar


def test_build_preserves_parameter_count(polys):
    triplet = _triplet(polys, "chebyshev", 5, "I1")
    model = build_polynomial_model(CHECKPOINT, triplet)
    reference, _ = load_trained_model(Path(CHECKPOINT))
    assert count_trainable_parameters(model) == count_trainable_parameters(reference)


def test_build_replaces_three_relu(polys):
    triplet = _triplet(polys, "chebyshev", 5, "I1")
    model = build_polynomial_model(CHECKPOINT, triplet)
    for act_name in ("act1", "act2", "act3"):
        assert isinstance(getattr(model, act_name), PolynomialActivation)


def test_build_model_in_eval(polys):
    triplet = _triplet(polys, "chebyshev", 5, "I1")
    model = build_polynomial_model(CHECKPOINT, triplet)
    assert model.training is False


def test_build_does_not_modify_reference(polys):
    """El modelo base de referencia no debe cambiar tras construir la polinómica."""
    reference, _ = load_trained_model(Path(CHECKPOINT))
    w_before = reference.conv1.weight.clone()
    triplet = _triplet(polys, "chebyshev", 5, "I1")
    _ = build_polynomial_model(CHECKPOINT, triplet)
    assert torch.equal(reference.conv1.weight, w_before)


# --- Inferencia mínima ---
def test_inference_produces_valid_logits(polys):
    triplet = _triplet(polys, "chebyshev", 5, "I1")
    model = build_polynomial_model(CHECKPOINT, triplet)
    x = torch.randn(8, 1, 28, 28, dtype=torch.float32)
    with torch.inference_mode():
        logits = model(x)
    assert logits.shape == (8, 10)
    assert torch.all(torch.isfinite(logits))  # Chebyshev d5 debe ser finito


def test_extreme_config_can_be_built(polys):
    """Taylor grado 9 (extremo) debe poder construirse aunque falle numéricamente."""
    triplet = _triplet(polys, "taylor", 9, "I2")
    model = build_polynomial_model(CHECKPOINT, triplet)  # no debe lanzar al construir
    for act_name in ("act1", "act2", "act3"):
        assert isinstance(getattr(model, act_name), PolynomialActivation)
