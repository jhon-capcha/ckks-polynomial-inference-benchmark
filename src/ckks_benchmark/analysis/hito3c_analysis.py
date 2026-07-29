"""
Análisis conjunto Hito 3B (error funcional) + Hito 3C-B (clasificación CNN).

Parte 1 de la fase 3C-C: une las tablas, agrega las tres activaciones por
configuración, calcula indicadores derivados (amplificación en cascada, deltas
relativos), asigna estados descriptivos, y calcula correlaciones con jerarquía
predefinida (principal = Chebyshev+LSQ válidas y elegibles).

NO selecciona configuraciones (eso es la Parte 2). Universo: las 24.

max_abs_coefficient se obtiene del registry de polinomios (polynomials.json),
no del CSV del 3B, que solo exportó magnitudes de salida.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ckks_benchmark.approximation.registry import (
    generate_all,
    load_and_validate_intervals,
)

FUNCTIONAL_CSV = Path("results/tables/hito3b_functional_metrics.csv")
CNN_CSV = Path("results/tables/hito3c_cnn_validation_metrics.csv")
INTERVALS_PATH = Path("results/published/preactivation_intervals.json")
OUT_CSV = Path("results/tables/hito3c_combined_analysis.csv")
OUT_JSON = Path("results/published/hito3c_analysis_summary.json")

EPS = 1e-12
VIABLE_ACCURACY_FLOOR = 0.50  # operativo: separa colapsado de funcional
ELIGIBLE_ACCURACY_MIN = 0.90  # operativo: umbral de shortlist CKKS
ACTIVATIONS = ("act1", "act2", "act3")


@dataclass(frozen=True)
class CombinedConfigurationAnalysis:
    configuration_id: str
    method: str
    degree: int
    interval_name: str

    # Estado
    valid: bool
    practically_viable: bool
    eligible_for_ckks: bool

    # Clasificación CNN (validación)
    validation_accuracy: float
    validation_macro_f1: float
    validation_loss: float
    delta_accuracy_vs_relu: float
    delta_f1_vs_relu: float
    delta_accuracy_relative: float
    delta_f1_relative: float
    prediction_change_fraction: float

    # Error funcional agregado (de las 3 activaciones)
    functional_mae_mean: float
    functional_mae_max: float
    functional_rmse_mean: float
    functional_rmse_max: float
    functional_p99_mean: float
    functional_max_error_max: float

    # Error funcional por activación
    act1_functional_mae: float
    act2_functional_mae: float
    act3_functional_mae: float

    # Magnitudes internas (del monitor)
    act1_max_abs: float
    act2_max_abs: float
    act3_max_abs: float
    logits_max_abs: float

    # Amplificación en cascada
    amplification_act1_act2: float
    amplification_act2_act3: float
    amplification_act3_logits: float

    # Complejidad
    max_abs_coefficient: float
    max_effective_degree: int

    first_non_finite_layer: str | None


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}.")
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} está vacío.")
    return rows


def _f(row: dict, key: str) -> float:
    """Lee un float de una fila CSV, tolerando vacíos/NaN."""
    val = row.get(key, "")
    if val == "" or val is None:
        return float("nan")
    return float(val)


def build_combined_analysis() -> list[CombinedConfigurationAnalysis]:
    functional_rows = _load_csv(FUNCTIONAL_CSV)
    cnn_rows = _load_csv(CNN_CSV)

    if len(functional_rows) != 72:
        raise ValueError(f"Se esperaban 72 filas funcionales, hay {len(functional_rows)}.")
    if len(cnn_rows) != 24:
        raise ValueError(f"Se esperaban 24 filas CNN, hay {len(cnn_rows)}.")

    functional_by_id = {r["configuration_id"]: r for r in functional_rows}

    # Coeficientes desde el registry (para max_abs_coefficient real).
    polys = generate_all(load_and_validate_intervals(INTERVALS_PATH))
    max_coef_by_id = {pid: float(np.max(np.abs(p.coefficients))) for pid, p in polys.items()}

    results = []
    for cnn in cnn_rows:
        method = cnn["method"]
        degree = int(cnn["degree"])
        interval = cnn["interval_name"]
        config_id = cnn["configuration_id"]

        # Unir con las 3 filas funcionales.
        act_rows = {}
        act_coefs = []
        for act in ACTIVATIONS:
            fkey = f"{act}_{method}_d{degree}_{interval}"
            if fkey not in functional_by_id:
                raise KeyError(f"Falta la fila funcional {fkey} para {config_id}.")
            act_rows[act] = functional_by_id[fkey]
            act_coefs.append(max_coef_by_id[fkey])

        maes = [_f(act_rows[a], "mae_empirical") for a in ACTIVATIONS]
        rmses = [_f(act_rows[a], "rmse_empirical") for a in ACTIVATIONS]
        p99s = [_f(act_rows[a], "p99_abs_error_empirical") for a in ACTIVATIONS]
        max_errs = [_f(act_rows[a], "max_error_empirical") for a in ACTIVATIONS]
        eff_degrees = [int(_f(act_rows[a], "effective_degree")) for a in ACTIVATIONS]

        valid = cnn["valid"].strip().lower() == "true"
        acc = _f(cnn, "validation_accuracy")
        viable = valid and (not np.isnan(acc)) and acc >= VIABLE_ACCURACY_FLOOR
        eligible = viable and acc >= ELIGIBLE_ACCURACY_MIN

        a1 = _f(cnn, "act1_max_abs")
        a2 = _f(cnn, "act2_max_abs")
        a3 = _f(cnn, "act3_max_abs")
        lg = _f(cnn, "logits_max_abs")

        d_acc = _f(cnn, "delta_accuracy_vs_relu")
        d_f1 = _f(cnn, "delta_f1_vs_relu")
        f1 = _f(cnn, "validation_macro_f1")
        relu_acc = acc + d_acc if not np.isnan(d_acc) else float("nan")
        relu_f1 = f1 + d_f1 if not np.isnan(d_f1) else float("nan")
        d_acc_rel = (
            (d_acc / relu_acc * 100) if (not np.isnan(relu_acc) and relu_acc != 0) else float("nan")
        )
        d_f1_rel = (
            (d_f1 / relu_f1 * 100) if (not np.isnan(relu_f1) and relu_f1 != 0) else float("nan")
        )

        results.append(
            CombinedConfigurationAnalysis(
                configuration_id=config_id,
                method=method,
                degree=degree,
                interval_name=interval,
                valid=valid,
                practically_viable=viable,
                eligible_for_ckks=eligible,
                validation_accuracy=acc,
                validation_macro_f1=f1,
                validation_loss=_f(cnn, "validation_loss"),
                delta_accuracy_vs_relu=d_acc,
                delta_f1_vs_relu=d_f1,
                delta_accuracy_relative=d_acc_rel,
                delta_f1_relative=d_f1_rel,
                prediction_change_fraction=_f(cnn, "prediction_change_fraction"),
                functional_mae_mean=float(np.mean(maes)),
                functional_mae_max=float(np.max(maes)),
                functional_rmse_mean=float(np.mean(rmses)),
                functional_rmse_max=float(np.max(rmses)),
                functional_p99_mean=float(np.mean(p99s)),
                functional_max_error_max=float(np.max(max_errs)),
                act1_functional_mae=maes[0],
                act2_functional_mae=maes[1],
                act3_functional_mae=maes[2],
                act1_max_abs=a1,
                act2_max_abs=a2,
                act3_max_abs=a3,
                logits_max_abs=lg,
                amplification_act1_act2=a2 / (a1 + EPS),
                amplification_act2_act3=a3 / (a2 + EPS),
                amplification_act3_logits=lg / (a3 + EPS),
                max_abs_coefficient=float(np.max(act_coefs)),
                max_effective_degree=int(np.max(eff_degrees)),
                first_non_finite_layer=(cnn.get("first_non_finite_layer") or None) or None,
            )
        )

    if len(results) != 24:
        raise RuntimeError(f"Se esperaban 24 análisis, hay {len(results)}.")
    return results


def _correlations(analyses, subset_mask) -> dict:
    """Pearson y Spearman entre MAE funcional y Δaccuracy sobre un subconjunto."""
    subset = [a for a in analyses if subset_mask(a)]
    if len(subset) < 3:
        return {"n": len(subset), "note": "muestra insuficiente"}

    mae = np.array([a.functional_mae_mean for a in subset])
    d_acc = np.array([a.delta_accuracy_vs_relu for a in subset])
    finite = np.isfinite(mae) & np.isfinite(d_acc)
    mae, d_acc = mae[finite], d_acc[finite]
    if len(mae) < 3:
        return {"n": int(len(mae)), "note": "insuficientes finitos"}

    pearson = float(np.corrcoef(mae, d_acc)[0, 1])
    rmae = np.argsort(np.argsort(mae))
    racc = np.argsort(np.argsort(d_acc))
    spearman = float(np.corrcoef(rmae, racc)[0, 1])
    return {"n": int(len(mae)), "pearson": pearson, "spearman": spearman}


def compute_correlation_hierarchy(analyses) -> dict:
    return {
        "primary_analysis": {
            "description": "Chebyshev + LSQ válidas, viables y elegibles",
            **_correlations(
                analyses,
                lambda a: a.eligible_for_ckks and a.method in ("chebyshev", "least_squares"),
            ),
        },
        "sensitivity_all_valid": {
            "description": "Todas las válidas",
            **_correlations(analyses, lambda a: a.valid),
        },
        "sensitivity_valid_no_taylor": {
            "description": "Válidas sin Taylor",
            **_correlations(analyses, lambda a: a.valid and a.method != "taylor"),
        },
        "sensitivity_cheby_lsq_valid": {
            "description": "Chebyshev y LSQ válidas",
            **_correlations(
                analyses, lambda a: a.valid and a.method in ("chebyshev", "least_squares")
            ),
        },
    }


def export_combined(analyses, correlations) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(analyses[0]).keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in analyses:
            writer.writerow(asdict(a))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_configurations": len(analyses),
        "n_valid": sum(1 for a in analyses if a.valid),
        "n_practically_viable": sum(1 for a in analyses if a.practically_viable),
        "n_eligible_for_ckks": sum(1 for a in analyses if a.eligible_for_ckks),
        "viable_accuracy_floor": VIABLE_ACCURACY_FLOOR,
        "eligible_accuracy_min": ELIGIBLE_ACCURACY_MIN,
        "correlations": correlations,
        "configurations": [asdict(a) for a in analyses],
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    analyses = build_combined_analysis()
    correlations = compute_correlation_hierarchy(analyses)
    export_combined(analyses, correlations)

    n_valid = sum(1 for a in analyses if a.valid)
    n_viable = sum(1 for a in analyses if a.practically_viable)
    n_eligible = sum(1 for a in analyses if a.eligible_for_ckks)

    print("=" * 72)
    print("ANÁLISIS CONJUNTO — Hito 3C-C Parte 1")
    print("=" * 72)
    print(f"Configuraciones:        {len(analyses)}")
    print(f"Válidas:                {n_valid}")
    print(f"Prácticamente viables:  {n_viable}")
    print(f"Elegibles para CKKS:    {n_eligible}")
    print("-" * 72)
    prim = correlations["primary_analysis"]
    print(f"Correlación principal (Cheby+LSQ elegibles, n={prim.get('n')}):")
    if "pearson" in prim:
        print(f"  Pearson (MAE↔Δacc):  {prim['pearson']:+.4f}")
        print(f"  Spearman:            {prim['spearman']:+.4f}")
    for key in (
        "sensitivity_all_valid",
        "sensitivity_valid_no_taylor",
        "sensitivity_cheby_lsq_valid",
    ):
        s = correlations[key]
        if "pearson" in s:
            print(f"  [{key}] n={s['n']} Pearson={s['pearson']:+.4f} Spearman={s['spearman']:+.4f}")
    print("-" * 72)
    print(f"CSV:  {OUT_CSV}")
    print(f"JSON: {OUT_JSON}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
