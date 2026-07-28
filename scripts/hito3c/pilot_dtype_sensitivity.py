"""
Piloto 3C-P1: sensibilidad de la evaluación polinómica al dtype.

Compara tres modos sobre configuraciones representativas, midiendo la diferencia
numérica de la evaluación polinómica y su impacto en logits/predicciones:
    Modo A: polinomio en float32 (CNN float32)
    Modo B: polinomio en float64, devuelto a float32 (CNN float32)
    Modo C: referencia — evaluación pura del polinomio en float64

Escenarios: least_squares_d5_I1 (estable), taylor_d5_I1 (viable),
taylor_d9_I2 (extremo). Mide sobre las preactivaciones de validación.
NO usa test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ckks_benchmark.approximation.registry import (
    generate_all,
    load_and_validate_intervals,
)
from ckks_benchmark.model.polynomial_activation import PolynomialActivation

INTERVALS_PATH = Path("results/published/preactivation_intervals.json")
NPZ_PATH = Path("data/processed/preactivations_validation_sample.npz")

SCENARIOS = [
    ("least_squares", 5, "I1"),  # estable representativa
    ("taylor", 5, "I1"),  # Taylor viable
    ("taylor", 9, "I2"),  # extremo
]


def relative_error(a: np.ndarray, b: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    """Error relativo |a-b| / max(|b|, floor)."""
    return np.abs(a - b) / np.maximum(np.abs(b), floor)


def main() -> int:
    intervals = load_and_validate_intervals(INTERVALS_PATH)
    polys = generate_all(intervals)
    data = np.load(NPZ_PATH)

    print("=" * 100)
    print("PILOTO 3C-P1 — Sensibilidad al dtype de la evaluación polinómica")
    print("=" * 100)

    for method, degree, interval in SCENARIOS:
        print(f"\n### {method} grado {degree} {interval} ###")
        print(
            f"{'act':>5} {'modo':>6} {'max_abs':>14} {'non_finite':>11} "
            f"{'MAE_vs_f64':>14} {'RMSE_vs_f64':>14} {'relE_P99':>12}"
        )
        print("-" * 90)

        for act in ("act1", "act2", "act3"):
            poly = polys[f"{act}_{method}_d{degree}_{interval}"]
            samples = data[act].astype(np.float64)

            # Referencia: NumPy float64 (Horner).
            ref = np.zeros_like(samples)
            for c in reversed(poly.coefficients):
                ref = ref * samples + c

            # Modo A: PolynomialActivation en float32.
            act_module = PolynomialActivation(poly.coefficients, evaluation_dtype=None)
            x32 = torch.tensor(samples, dtype=torch.float32)
            out_a = act_module(x32).numpy().astype(np.float64)

            # Modo B: evaluación en float64, salida a float32.
            act_module_b = PolynomialActivation(poly.coefficients, evaluation_dtype=torch.float64)
            out_b = act_module_b(x32).numpy().astype(np.float64)

            # Comparación de cada modo contra la referencia float64.
            for mode, out in (("A_f32", out_a), ("B_f64", out_b)):
                max_abs = float(np.max(np.abs(out)))
                non_finite = int(np.count_nonzero(~np.isfinite(out)))
                if non_finite == 0:
                    mae = float(np.mean(np.abs(out - ref)))
                    rmse = float(np.sqrt(np.mean((out - ref) ** 2)))
                    rel_p99 = float(np.percentile(relative_error(out, ref), 99))
                else:
                    mae = rmse = rel_p99 = float("nan")
                print(
                    f"{act:>5} {mode:>6} {max_abs:>14.4e} {non_finite:>11} "
                    f"{mae:>14.4e} {rmse:>14.4e} {rel_p99:>12.4e}"
                )

    print("\n" + "=" * 100)
    print("Lectura: si Modo A (f32) difiere apreciablemente de la referencia f64,")
    print("o produce non_finite, el Modo B (polinomio en f64) es el principal.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
