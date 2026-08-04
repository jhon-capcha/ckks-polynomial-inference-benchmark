"""
Evaluación CKKS de los polinomios oficiales del Hito 3A con Horner (Hito 4C).

Carga la shortlist congelada (Chebyshev/LSQ, grados 3 y 5, act1/act2/act3),
evalúa cada polinomio bajo CKKS con Horner sobre muestras reales de
preactivación, y mide el error CKKS (MAE, RMSE, P95, P99) contra la evaluación
clara del mismo polinomio.

Separa: error de aproximación (Hito 3B, ReLU vs p) del error CKKS
(p claro vs p cifrado). Aquí SOLO se mide el error CKKS.

Estrategia oficial: horner. Grado máximo oficial: 5.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ckks_benchmark.he.ciphertext_state import OperationRecorder
from ckks_benchmark.he.ckks_context import create_context
from ckks_benchmark.he.operations import decrypt_vector, encrypt_vector
from ckks_benchmark.he.parameters import profile_for_degree
from ckks_benchmark.he.polynomial_evaluator import evaluate_polynomial_ckks

POLYS_PATH = Path("results/approximations/coefficients/polynomials.json")
FROZEN_PATH = Path("results/published/hito3c_frozen_selection.json")
SAMPLE_PATH = Path("data/processed/preactivations_validation_sample.npz")
OUT_CSV = Path("results/tables/hito4_polynomial_ckks_error.csv")

# Solo grados oficiales (3 y 5); grado 7 se documenta aparte como no factible.
OFFICIAL_DEGREES = {3, 5}
N_SAMPLES = 200  # muestras de preactivación por configuración


def load_official_polynomials() -> list[dict]:
    """Carga los polinomios de la shortlist con grados 3 y 5."""
    data = json.loads(POLYS_PATH.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    selected_ids = {s["configuration_id"] for s in frozen["selected_for_ckks"]}

    # Los IDs de la shortlist son por configuración (e.g. chebyshev_d5_I1);
    # cada uno corresponde a 3 polinomios (act1/act2/act3).
    result = []
    for poly in data["polynomials"]:
        # Config id: method_dDEGREE_interval (sin activación).
        config_id = f"{poly['method']}_d{poly['degree']}_{poly['interval_name']}"
        if config_id in selected_ids and poly["degree"] in OFFICIAL_DEGREES:
            result.append(poly)
    return result


def load_samples() -> dict:
    """Carga muestras reales de preactivación por activación."""
    data = np.load(SAMPLE_PATH)
    return {key: data[key] for key in data.files}


def evaluate_one(poly: dict, samples: np.ndarray) -> dict:
    """Evalúa un polinomio bajo CKKS con Horner y mide error vs claro."""
    coeffs = np.array(poly["coefficients"], dtype=np.float64)
    degree = poly["degree"]
    profile = profile_for_degree(degree)

    # Evaluación clara (referencia): p(x) con NumPy.
    clear = np.polyval(coeffs[::-1], samples)

    # Evaluación CKKS: cifrar el vector de muestras, evaluar, descifrar.
    ctx = create_context(profile.profile_id)
    he = ctx.he
    slots = he.get_nSlots()

    # Empaquetar las muestras en un ciphertext (rellenar a slots).
    n = len(samples)
    packed = np.zeros(slots, dtype=np.float64)
    packed[:n] = samples

    rec = OperationRecorder()
    x_ct = encrypt_vector(he, packed, rec)
    result_ct = evaluate_polynomial_ckks(he, x_ct, coeffs, strategy="horner", recorder=rec)
    ckks = decrypt_vector(he, result_ct, n)

    # Error CKKS: claro vs cifrado.
    err = np.abs(clear - ckks)
    finite = np.isfinite(ckks)
    n_nonfinite = int(np.sum(~finite))
    err_finite = err[finite]

    summ = rec.summary()
    return {
        "configuration_id": f"{poly['method']}_d{degree}_{poly['interval_name']}",
        "polynomial_id": poly["id"],
        "activation": poly["activation"],
        "method": poly["method"],
        "degree": degree,
        "interval_name": poly["interval_name"],
        "strategy": "horner",
        "n_samples": n,
        "mae_ckks": float(np.mean(err_finite)) if err_finite.size else float("nan"),
        "rmse_ckks": float(np.sqrt(np.mean(err_finite**2))) if err_finite.size else float("nan"),
        "max_error_ckks": float(np.max(err_finite)) if err_finite.size else float("nan"),
        "p95_error_ckks": float(np.percentile(err_finite, 95)) if err_finite.size else float("nan"),
        "p99_error_ckks": float(np.percentile(err_finite, 99)) if err_finite.size else float("nan"),
        "non_finite_count": n_nonfinite,
        "levels_consumed": summ["levels_consumed"],
        "final_mod_level": summ["final_mod_level"],
        "final_scale_bits": summ["final_scale_bits"],
    }


def main() -> int:
    print("=" * 72)
    print("EVALUACIÓN CKKS DE POLINOMIOS OFICIALES — Hito 4C (Horner)")
    print("=" * 72)

    polys = load_official_polynomials()
    samples_by_act = load_samples()
    print(f"Polinomios oficiales (grados 3/5): {len(polys)}")
    print(f"Activaciones con muestras: {list(samples_by_act.keys())}")
    print("-" * 72)

    results = []
    for poly in sorted(polys, key=lambda p: (p["degree"], p["method"], p["activation"])):
        act = poly["activation"]
        # Muestra reproducible de preactivaciones de esta activación.
        all_samples = samples_by_act[act]
        rng = np.random.default_rng(20260803)
        idx = rng.choice(len(all_samples), size=min(N_SAMPLES, len(all_samples)), replace=False)
        samples = all_samples[idx].astype(np.float64)

        res = evaluate_one(poly, samples)
        results.append(res)
        print(
            f"  {res['polynomial_id']:<28} MAE={res['mae_ckks']:.2e} "
            f"P99={res['p99_error_ckks']:.2e} niveles={res['levels_consumed']} "
            f"nf={res['non_finite_count']}"
        )

    # Export.
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("-" * 72)
    # Resumen: error CKKS máximo observado.
    max_mae = max(results, key=lambda r: r["mae_ckks"])
    print(f"MAE CKKS máximo: {max_mae['mae_ckks']:.2e} ({max_mae['polynomial_id']})")
    print(f"Todas las evaluaciones finitas: {all(r['non_finite_count'] == 0 for r in results)}")
    print(f"CSV: {OUT_CSV}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
