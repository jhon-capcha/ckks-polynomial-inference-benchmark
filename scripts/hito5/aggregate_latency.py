"""
Agregación estadística de la latencia (Hito 5B).

Consume el crudo (1800 filas, desglose encrypt/act3/fc2/decrypt) y produce:
  - hito5_latency_by_stage.csv: estadística por configuración y etapa
  - hito5_latency_by_config.csv: latencia total y fracción por etapa
  - verificación del residual temporal (control de instrumentación)

Estadística: mediana y P95 principales; media/desv/CV complementarias.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from ckks_benchmark.benchmark.timing import summarize_latency

RAW = Path("results/tables/hito5_latency_raw.csv")
OUT_BY_STAGE = Path("results/tables/hito5_latency_by_stage.csv")
OUT_BY_CONFIG = Path("results/tables/hito5_latency_by_config.csv")

STAGES = ["encrypt", "act3", "fc2", "decrypt"]


def load_raw() -> list[dict]:
    with RAW.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    rows = load_raw()
    print(f"Filas crudas: {len(rows)}")

    # Verificar residual temporal (control de instrumentación).
    residuals = [abs(float(r["timing_residual_ratio"])) for r in rows]
    print(f"Residual temporal: max={max(residuals):.2e}, media={np.mean(residuals):.2e}")
    print("  (residual ~0 confirma que no hay trabajo no instrumentado entre etapas)")

    # Agrupar por configuración.
    by_config = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cid = r["configuration_id"]
        for stage in STAGES:
            by_config[cid][stage].append(float(r[f"{stage}_seconds"]))
        by_config[cid]["online_total"].append(float(r["online_total_seconds"]))

    configs = sorted(by_config.keys())

    # --- Tabla por etapa ---
    stage_rows = []
    for cid in configs:
        degree = int(cid.split("_d")[1][0])
        method = cid.split("_d")[0]
        interval = cid.split("_")[-1]
        for stage in STAGES + ["online_total"]:
            summ = summarize_latency(by_config[cid][stage])
            stage_rows.append(
                {
                    "configuration_id": cid,
                    "method": method,
                    "degree": degree,
                    "interval_name": interval,
                    "stage": stage,
                    "sample_count": summ["sample_count"],
                    "median_ms": summ["median_seconds"] * 1000,
                    "p95_ms": summ["p95_seconds"] * 1000,
                    "mean_ms": summ["mean_seconds"] * 1000,
                    "std_ms": summ["std_seconds"] * 1000,
                    "p99_ms": summ["p99_seconds"] * 1000,
                    "min_ms": summ["min_seconds"] * 1000,
                    "max_ms": summ["max_seconds"] * 1000,
                    "coefficient_of_variation": summ["coefficient_of_variation"],
                }
            )

    OUT_BY_STAGE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_BY_STAGE.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stage_rows[0].keys()))
        w.writeheader()
        w.writerows(stage_rows)

    # --- Tabla por configuración: total + fracción por etapa ---
    config_rows = []
    for cid in configs:
        degree = int(cid.split("_d")[1][0])
        total_median = np.median(by_config[cid]["online_total"]) * 1000
        row = {
            "configuration_id": cid,
            "method": cid.split("_d")[0],
            "degree": degree,
            "interval_name": cid.split("_")[-1],
            "online_total_median_ms": total_median,
            "online_total_p95_ms": np.percentile(by_config[cid]["online_total"], 95) * 1000,
        }
        for stage in STAGES:
            stage_median = np.median(by_config[cid][stage]) * 1000
            row[f"{stage}_median_ms"] = stage_median
            row[f"{stage}_share_pct"] = 100 * stage_median / total_median
        config_rows.append(row)

    with OUT_BY_CONFIG.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(config_rows[0].keys()))
        w.writeheader()
        w.writerows(config_rows)

    # --- Resumen en consola ---
    print("\n" + "=" * 72)
    print("DESGLOSE POR ETAPA (mediana ms | fracción del total)")
    print("=" * 72)
    print(f"{'config':<24} {'encrypt':>14} {'act3':>14} {'fc2':>14} {'decrypt':>12} {'total':>8}")
    for row in config_rows:
        print(
            f"{row['configuration_id']:<24} "
            f"{row['encrypt_median_ms']:>6.1f}({row['encrypt_share_pct']:>4.1f}%) "
            f"{row['act3_median_ms']:>6.1f}({row['act3_share_pct']:>4.1f}%) "
            f"{row['fc2_median_ms']:>6.1f}({row['fc2_share_pct']:>4.1f}%) "
            f"{row['decrypt_median_ms']:>5.1f}({row['decrypt_share_pct']:>3.1f}%) "
            f"{row['online_total_median_ms']:>6.1f}"
        )

    # Análisis grado 3 vs 5.
    print("\n" + "-" * 72)
    print("INCREMENTO grado 3 -> 5 (por etapa, mediana ms):")
    d3 = next(r for r in config_rows if r["degree"] == 3 and "chebyshev" in r["configuration_id"])
    d5 = next(
        r
        for r in config_rows
        if r["degree"] == 5
        and "chebyshev" in r["configuration_id"]
        and "I1" in r["configuration_id"]
    )
    for stage in STAGES + ["online_total"]:
        key = f"{stage}_median_ms" if stage != "online_total" else "online_total_median_ms"
        delta = d5[key] - d3[key]
        print(f"  {stage:<14} d3={d3[key]:>6.1f}  d5={d5[key]:>6.1f}  Δ={delta:>+6.1f}ms")

    print(f"\nTablas: {OUT_BY_STAGE}, {OUT_BY_CONFIG}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
