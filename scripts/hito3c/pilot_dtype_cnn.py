"""
Piloto 3C-P2: efecto del dtype en la CNN completa (logits, márgenes, predicciones).

Compara Modo A (polinomio float32) vs Modo B (polinomio float64) a nivel de red,
sobre tres configuraciones representativas, midiendo si las diferencias numéricas
alteran logits, márgenes o predicciones. NO usa test.

    least_squares_d5_I1  (estable)
    taylor_d5_I1         (viable)
    taylor_d9_I2         (extremo)
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from ckks_benchmark.approximation.registry import (
    generate_all,
    load_and_validate_intervals,
)
from ckks_benchmark.model.polynomial_activation import PolynomialActivation
from ckks_benchmark.model.preactivations import load_trained_model

INTERVALS_PATH = Path("results/published/preactivation_intervals.json")
CHECKPOINT = Path("models/reduced_lenet_relu_best.pt")
SUBSET_SIZE = 1024

SCENARIOS = [
    ("least_squares", 5, "I1"),
    ("taylor", 5, "I1"),
    ("taylor", 9, "I2"),
]


def build_polynomial_cnn(base_model, polys, method, degree, interval, eval_dtype):
    """Clona la CNN base y sustituye act1/act2/act3 por polinomios (terna)."""
    model = copy.deepcopy(base_model)
    for act_name in ("act1", "act2", "act3"):
        poly = polys[f"{act_name}_{method}_d{degree}_{interval}"]
        setattr(
            model,
            act_name,
            PolynomialActivation(poly.coefficients, evaluation_dtype=eval_dtype),
        )
    model.eval()
    return model


def get_validation_subset(size: int):
    """Subconjunto fijo y reproducible de validación (sin shuffle)."""
    from ckks_benchmark.model.train import TrainingConfig, create_dataloaders

    config = TrainingConfig()
    _train, val_loader, _test = create_dataloaders(config)

    images, labels = [], []
    for x, y in val_loader:
        images.append(x)
        labels.append(y)
        if sum(t.shape[0] for t in images) >= size:
            break
    x_all = torch.cat(images)[:size]
    y_all = torch.cat(labels)[:size]
    return x_all, y_all


def logits_of(model, x) -> torch.Tensor:
    with torch.inference_mode():
        return model(x)


def margins(logits: torch.Tensor) -> torch.Tensor:
    """Margen = top1 - top2 por muestra."""
    top2 = torch.topk(logits, 2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def main() -> int:
    base_model, _ckpt = load_trained_model(CHECKPOINT)
    intervals = load_and_validate_intervals(INTERVALS_PATH)
    polys = generate_all(intervals)

    x, y = get_validation_subset(SUBSET_SIZE)
    print("=" * 96)
    print(f"PILOTO 3C-P2 — dtype en CNN completa ({SUBSET_SIZE} imágenes de validación)")
    print("=" * 96)

    for method, degree, interval in SCENARIOS:
        model_a = build_polynomial_cnn(base_model, polys, method, degree, interval, None)
        model_b = build_polynomial_cnn(base_model, polys, method, degree, interval, torch.float64)

        logits_a = logits_of(model_a, x)
        logits_b = logits_of(model_b, x)

        pred_a = torch.argmax(logits_a, dim=1)
        pred_b = torch.argmax(logits_b, dim=1)

        # Diferencias de logits.
        diff = (logits_a.double() - logits_b.double()).abs()
        logit_mae = float(diff.mean())
        logit_rmse = float(torch.sqrt((diff**2).mean()))
        logit_max = float(diff.max())

        # Predicciones.
        pred_change = int((pred_a != pred_b).sum())
        pred_frac = pred_change / len(y)

        # Accuracy / margen.
        acc_a = float((pred_a == y).float().mean())
        acc_b = float((pred_b == y).float().mean())
        margin_diff = (margins(logits_a).double() - margins(logits_b).double()).abs()
        margin_mean = float(margin_diff.mean())
        margin_p95 = float(torch.quantile(margin_diff, 0.95))

        # No-finitud.
        nf_a = int((~torch.isfinite(logits_a)).sum())
        nf_b = int((~torch.isfinite(logits_b)).sum())

        print(f"\n### {method} grado {degree} {interval} ###")
        print(f"  logit MAE(A vs B):     {logit_mae:.4e}")
        print(f"  logit RMSE(A vs B):    {logit_rmse:.4e}")
        print(f"  logit max_err(A vs B): {logit_max:.4e}")
        print(f"  pred_change:           {pred_change} ({pred_frac * 100:.3f}%)")
        print(f"  accuracy A / B:        {acc_a:.4f} / {acc_b:.4f}  (Δ={abs(acc_a - acc_b):.4e})")
        print(f"  margin |Δ| mean / p95: {margin_mean:.4e} / {margin_p95:.4e}")
        print(f"  non_finite logits A/B: {nf_a} / {nf_b}")

    print("\n" + "=" * 96)
    print("Decisión: si pred_change=0 y Δaccuracy=0 en las 3 configs, Modo A (float32) basta.")
    print("Si alguna cambia predicciones o accuracy, Modo B (float64) es principal.")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
