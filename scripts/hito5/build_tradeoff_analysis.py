"""
Análisis de trade-off precisión–profundidad–latencia–huella (Hito 5D).

Integra (NO genera datos primarios): une por configuration_id las tres fuentes:
  - precisión: hito4_ckks_validation_summary.csv
  - latencia:  hito5_latency_by_config.csv
  - recursos:  hito5_resource_consumption.csv

Calcula fronteras de Pareto:
  - 2D: maximizar ckks_accuracy, minimizar median_online_ms
  - 3D: + minimizar levels_consumed
Claves y tamaños quedan como dimensiones descriptivas (no en la frontera).

Nota terminológica: los tamaños son huella de almacenamiento/comunicación
(bytes serializados), no memoria RAM residente.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

PRECISION_CSV = Path("results/tables/hito4_ckks_validation_summary.csv")
LATENCY_CSV = Path("results/tables/hito5_latency_by_config.csv")
RESOURCE_CSV = Path("results/tables/hito5_resource_consumption.csv")
OUT_CSV = Path("results/tables/hito5_precision_depth_latency.csv")
OUT_JSON = Path("results/published/hito5_tradeoff_analysis.json")

CONFIGURATIONS = [
    "chebyshev_d3_I1",
    "least_squares_d3_I1",
    "chebyshev_d5_I1",
    "least_squares_d5_I1",
    "chebyshev_d5_I2",
    "least_squares_d5_I2",
]

# Niveles consumidos por grado (act3 + fc2): grado 3 = 4, grado 5 = 6.
LEVELS_BY_DEGREE = {3: 4, 5: 6}
DEPTH_BY_DEGREE = {3: 3, 5: 5}  # profundidad del polinomio act3


def _index_by_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return {r["configuration_id"]: r for r in csv.DictReader(f)}


def compute_pareto(rows: list[dict], objectives: list[tuple[str, str]]) -> list[bool]:
    """Marca configuraciones Pareto-óptimas.

    objectives: lista de (columna, 'max'|'min'). Una config A domina a B si es
    igual o mejor en todos los objetivos y estrictamente mejor en al menos uno.
    """
    n = len(rows)
    optimal = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # ¿j domina a i?
            better_or_equal_all = True
            strictly_better_one = False
            for col, direction in objectives:
                vi, vj = float(rows[i][col]), float(rows[j][col])
                if direction == "max":
                    if vj < vi:
                        better_or_equal_all = False
                        break
                    if vj > vi:
                        strictly_better_one = True
                else:  # min
                    if vj > vi:
                        better_or_equal_all = False
                        break
                    if vj < vi:
                        strictly_better_one = True
            if better_or_equal_all and strictly_better_one:
                optimal[i] = False
                break
    return optimal


def main() -> int:
    print("=" * 72)
    print("TRADE-OFF PRECISIÓN–PROFUNDIDAD–LATENCIA–HUELLA — Hito 5D")
    print("=" * 72)

    precision = _index_by_config(PRECISION_CSV)
    latency = _index_by_config(LATENCY_CSV)
    resources = _index_by_config(RESOURCE_CSV)

    # Validación de unión: cada config debe estar en las tres fuentes.
    for cid in CONFIGURATIONS:
        assert cid in precision, f"{cid} falta en precisión"
        assert cid in latency, f"{cid} falta en latencia"
        assert cid in resources, f"{cid} falta en recursos"
    print(f"Unión validada: {len(CONFIGURATIONS)} configuraciones en las 3 fuentes.")

    rows = []
    for cid in CONFIGURATIONS:
        p, l, r = precision[cid], latency[cid], resources[cid]
        degree = int(r["degree"])
        rows.append(
            {
                "configuration_id": cid,
                "method": r["method"],
                "degree": degree,
                "interval_name": r["interval_name"],
                "parameter_profile": r["profile"],
                # Precisión (Hito 4).
                "relu_accuracy": float(p["relu_accuracy"]),
                "polynomial_clear_accuracy": float(p["polynomial_clear_accuracy"]),
                "ckks_accuracy": float(p["ckks_accuracy"]),
                "delta_approximation_accuracy": float(p["delta_approximation_accuracy"]),
                "delta_ckks_accuracy": float(p["delta_ckks_accuracy"]),
                "delta_total_accuracy": float(p["delta_total_accuracy"]),
                # Profundidad.
                "levels_consumed": LEVELS_BY_DEGREE[degree],
                "multiplicative_depth": DEPTH_BY_DEGREE[degree],
                # Latencia (Hito 5B).
                "median_online_ms": float(l["online_total_median_ms"]),
                "p95_online_ms": float(l["online_total_p95_ms"]),
                "act3_median_ms": float(l["act3_median_ms"]),
                "fc2_median_ms": float(l["fc2_median_ms"]),
                # Huella de almacenamiento (Hito 5C) — descriptiva.
                "rotation_keys_mb": float(r["rotation_key_bytes"]) / 1e6,
                "relin_key_mb": float(r["relin_key_bytes"]) / 1e6,
                "public_key_mb": float(r["public_key_bytes"]) / 1e6,
                "input_ciphertext_mb": float(r["input_ciphertext_bytes"]) / 1e6,
                "output_ciphertexts_mb": float(r["output_ciphertexts_bytes"]) / 1e6,
            }
        )

    # Pareto 2D: max accuracy, min latencia.
    pareto_2d = compute_pareto(rows, [("ckks_accuracy", "max"), ("median_online_ms", "min")])
    # Pareto 3D: + min niveles.
    pareto_3d = compute_pareto(
        rows, [("ckks_accuracy", "max"), ("median_online_ms", "min"), ("levels_consumed", "min")]
    )
    for row, p2, p3 in zip(rows, pareto_2d, pareto_3d):
        row["pareto_accuracy_latency"] = int(p2)
        row["pareto_accuracy_latency_levels"] = int(p3)

    # Export CSV.
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Diferencias relativas grado 3 -> 5 (representativas).
    d3 = next(r for r in rows if r["degree"] == 3)
    d5 = next(
        r
        for r in rows
        if r["degree"] == 5 and r["interval_name"] == "I1" and "chebyshev" in r["configuration_id"]
    )
    rel = {
        "latency_increase_pct": 100
        * (d5["median_online_ms"] - d3["median_online_ms"])
        / d3["median_online_ms"],
        "rotation_keys_increase_pct": 100
        * (d5["rotation_keys_mb"] - d3["rotation_keys_mb"])
        / d3["rotation_keys_mb"],
        "input_ct_increase_pct": 100
        * (d5["input_ciphertext_mb"] - d3["input_ciphertext_mb"])
        / d3["input_ciphertext_mb"],
        "accuracy_gain": d5["ckks_accuracy"] - d3["ckks_accuracy"],
    }

    # JSON consolidado.
    OUT_JSON.write_text(
        json.dumps(
            {
                "phase": "Hito 5D",
                "configurations": rows,
                "grade_3_to_5_relative": rel,
                "pareto_2d_optimal": [
                    r["configuration_id"] for r in rows if r["pareto_accuracy_latency"]
                ],
                "pareto_3d_optimal": [
                    r["configuration_id"] for r in rows if r["pareto_accuracy_latency_levels"]
                ],
                "note": "Tamaños = huella de almacenamiento/comunicación (bytes serializados), no RAM residente.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Resumen.
    print("\nTabla de trade-off:")
    print(f"{'config':<24} {'acc':>6} {'lat(ms)':>8} {'niveles':>8} {'rot(MB)':>8} {'Pareto2D':>9}")
    for r in rows:
        print(
            f"{r['configuration_id']:<24} {r['ckks_accuracy']:>6.3f} "
            f"{r['median_online_ms']:>8.1f} {r['levels_consumed']:>8} "
            f"{r['rotation_keys_mb']:>8.1f} {'SÍ' if r['pareto_accuracy_latency'] else 'no':>9}"
        )

    print("\n" + "-" * 72)
    print(
        f"Incremento grado 3->5: latencia {rel['latency_increase_pct']:+.1f}%, "
        f"claves rotación {rel['rotation_keys_increase_pct']:+.1f}%, "
        f"ct entrada {rel['input_ct_increase_pct']:+.1f}%, "
        f"ganancia accuracy {rel['accuracy_gain']:+.3f}"
    )
    print(
        f"\nPareto 2D (accuracy-latencia): "
        f"{[r['configuration_id'] for r in rows if r['pareto_accuracy_latency']]}"
    )
    print(
        f"Pareto 3D (+niveles): "
        f"{[r['configuration_id'] for r in rows if r['pareto_accuracy_latency_levels']]}"
    )
    print(f"\nCSV: {OUT_CSV}")
    print(f"JSON: {OUT_JSON}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
