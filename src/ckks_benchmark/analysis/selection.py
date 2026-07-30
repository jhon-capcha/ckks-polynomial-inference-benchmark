"""
Motor de selección de shortlist para CKKS (fase 3C-C, Parte 2).

Aplica reglas categóricas GENERALES (no IDs predefinidos) sobre las
configuraciones elegibles del análisis conjunto, para producir una shortlist
diversa y justificable. Congela la selección en JSON con hashes, ANTES de
usar el conjunto de test.

Categorías:
    A - mejor Chebyshev / mejor LSQ por accuracy (máxima precisión)
    B - mejor Chebyshev / mejor LSQ grado 5 (compromiso)
    C - mejor Chebyshev / mejor LSQ grado 3 (bajo grado, menor profundidad)
    D - pareja I1/I2 comparable (comparación de intervalo), si aporta
    Baseline diagnóstico: mejor Taylor finito (selected_for_ckks=false)

Desempates: mayor accuracy -> mayor F1 -> menor grado -> menor MAE funcional
-> menor amplificación interna.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from ckks_benchmark.analysis.hito3c_analysis import (
    ELIGIBLE_ACCURACY_MIN,
    CombinedConfigurationAnalysis,
    build_combined_analysis,
)

FROZEN_SELECTION = Path("results/published/hito3c_frozen_selection.json")
SELECTION_MANIFEST = Path("results/published/hito3c_selection_manifest.json")
COMBINED_CSV = Path("results/tables/hito3c_combined_analysis.csv")
FUNCTIONAL_CSV = Path("results/tables/hito3b_functional_metrics.csv")
CNN_CSV = Path("results/tables/hito3c_cnn_validation_metrics.csv")

TIE_ACCURACY_TOLERANCE = 0.001


def _tie_break_key(a: CombinedConfigurationAnalysis) -> tuple:
    """Clave de ordenamiento: mejor primero.

    Mayor accuracy, mayor F1, menor grado, menor MAE funcional, menor
    amplificación act2->act3 (proxy de estabilidad de cascada).
    """
    amp = a.amplification_act2_act3
    amp = amp if np.isfinite(amp) else float("inf")
    return (
        -a.validation_accuracy,
        -a.validation_macro_f1,
        a.degree,
        a.functional_mae_mean,
        amp,
    )


def _best(candidates: list[CombinedConfigurationAnalysis]) -> CombinedConfigurationAnalysis | None:
    return min(candidates, key=_tie_break_key) if candidates else None


def select_shortlist(
    analyses: list[CombinedConfigurationAnalysis],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Aplica las reglas categóricas. Devuelve (selected, diagnostics, excluded)."""
    eligible = [a for a in analyses if a.eligible_for_ckks]

    selected: dict[str, dict] = {}  # config_id -> registro (dedup)

    def add(config: CombinedConfigurationAnalysis, category: str, reason: str) -> None:
        if config is None or config.configuration_id in selected:
            return
        selected[config.configuration_id] = {
            "configuration_id": config.configuration_id,
            "method": config.method,
            "degree": config.degree,
            "interval_name": config.interval_name,
            "validation_accuracy": config.validation_accuracy,
            "validation_macro_f1": config.validation_macro_f1,
            "delta_accuracy_vs_relu": config.delta_accuracy_vs_relu,
            "max_effective_degree": config.max_effective_degree,
            "max_abs_coefficient": config.max_abs_coefficient,
            "selection_category": category,
            "selection_reason": reason,
        }

    # Categoría A: mejor por método (máxima accuracy).
    for method in ("chebyshev", "least_squares"):
        best = _best([a for a in eligible if a.method == method])
        add(
            best,
            "maximum_accuracy",
            f"Best validation accuracy among eligible {method} configurations",
        )

    # Categoría B: mejor grado 5 por método.
    for method in ("chebyshev", "least_squares"):
        best = _best([a for a in eligible if a.method == method and a.degree == 5])
        add(
            best,
            "degree5_tradeoff",
            f"Best eligible {method} degree-5 configuration (intermediate complexity)",
        )

    # Categoría C: mejor grado 3 por método.
    for method in ("chebyshev", "least_squares"):
        best = _best([a for a in eligible if a.method == method and a.degree == 3])
        add(
            best,
            "degree3_low_depth",
            f"Best eligible {method} degree-3 configuration (lower multiplicative depth)",
        )

    # Categoría D: pareja I1/I2 comparable (mismo método y grado, ambos elegibles).
    for method in ("chebyshev", "least_squares"):
        for degree in (3, 5, 7, 9):
            group = [a for a in eligible if a.method == method and a.degree == degree]
            intervals = {a.interval_name for a in group}
            if intervals == {"I1", "I2"}:
                # Añade ambos si aún no están, para comparar intervalo.
                for a in group:
                    add(
                        a,
                        "interval_comparison",
                        f"Interval comparison pair {method} d{degree} (I1 vs I2)",
                    )
                break  # una pareja por método basta

    # Baseline diagnóstico: mejor Taylor finito (aunque no viable).
    taylor_finite = [a for a in analyses if a.method == "taylor" and a.valid]
    best_taylor = _best(taylor_finite) if taylor_finite else None
    diagnostics = []
    if best_taylor is not None:
        diagnostics.append(
            {
                "configuration_id": best_taylor.configuration_id,
                "method": "taylor",
                "degree": best_taylor.degree,
                "interval_name": best_taylor.interval_name,
                "validation_accuracy": best_taylor.validation_accuracy,
                "purpose": "diagnostic_baseline",
                "selected_for_ckks": False,
                "reason": "Finite but classification performance is not practically viable",
            }
        )

    # Exclusiones: todo lo no seleccionado ni diagnóstico, con causa.
    selected_ids = set(selected)
    diag_ids = {d["configuration_id"] for d in diagnostics}
    excluded = []
    for a in analyses:
        if a.configuration_id in selected_ids or a.configuration_id in diag_ids:
            continue
        if not a.valid:
            reason = f"invalid_non_finite (collapse at {a.first_non_finite_layer})"
        elif not a.practically_viable:
            reason = "not_practically_viable (accuracy near chance)"
        elif not a.eligible_for_ckks:
            reason = f"accuracy_below_threshold (<{ELIGIBLE_ACCURACY_MIN})"
        else:
            reason = "eligible_but_not_selected (dominated by chosen candidates)"
        excluded.append(
            {
                "configuration_id": a.configuration_id,
                "excluded_reason": reason,
            }
        )

    return list(selected.values()), diagnostics, excluded


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def freeze_selection() -> dict:
    analyses = build_combined_analysis()
    selected, diagnostics, excluded = select_shortlist(analyses)

    created = datetime.now(UTC).isoformat()
    commit = _git_commit()

    selection = {
        "schema_version": "1.0",
        "phase": "Hito 3C-C",
        "selection_status": "frozen",
        "test_used": False,
        "selection_split": "validation",
        "selection_rules": {
            "minimum_validation_accuracy": ELIGIBLE_ACCURACY_MIN,
            "require_finite_outputs": True,
            "require_practical_viability": True,
            "tie_accuracy_tolerance": TIE_ACCURACY_TOLERANCE,
            "categories": [
                "maximum_accuracy (best per method)",
                "degree5_tradeoff",
                "degree3_low_depth",
                "interval_comparison",
            ],
        },
        "selected_for_ckks": selected,
        "diagnostic_baselines": diagnostics,
        "excluded": excluded,
        "counts": {
            "selected": len(selected),
            "diagnostics": len(diagnostics),
            "excluded": len(excluded),
            "total": len(analyses),
        },
        "inputs": {
            "combined_analysis_csv": str(COMBINED_CSV).replace("\\", "/"),
            "functional_metrics_csv": str(FUNCTIONAL_CSV).replace("\\", "/"),
            "cnn_metrics_csv": str(CNN_CSV).replace("\\", "/"),
        },
        "hashes": {
            "combined_analysis_csv": _sha256_of_file(COMBINED_CSV),
            "functional_metrics_csv": _sha256_of_file(FUNCTIONAL_CSV),
            "cnn_metrics_csv": _sha256_of_file(CNN_CSV),
        },
        "git_commit": commit,
        "created_at_utc": created,
    }

    FROZEN_SELECTION.parent.mkdir(parents=True, exist_ok=True)
    FROZEN_SELECTION.write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Manifiesto (metadatos de trazabilidad, sin los registros completos).
    manifest = {
        "created_at_utc": created,
        "git_commit": commit,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "selection_split": "validation",
        "test_used": False,
        "minimum_validation_accuracy": ELIGIBLE_ACCURACY_MIN,
        "n_selected": len(selected),
        "n_diagnostics": len(diagnostics),
        "n_excluded": len(excluded),
        "selected_ids": [s["configuration_id"] for s in selected],
        "hashes": selection["hashes"],
    }
    SELECTION_MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return selection


def main() -> int:
    selection = freeze_selection()

    print("=" * 72)
    print("SELECCIÓN Y CONGELAMIENTO — Hito 3C-C Parte 2")
    print("=" * 72)
    print(f"Configuraciones totales:  {selection['counts']['total']}")
    print(f"Seleccionadas para CKKS:  {selection['counts']['selected']}")
    print(f"Baselines diagnósticos:   {selection['counts']['diagnostics']}")
    print(f"Excluidas:                {selection['counts']['excluded']}")
    print(f"Test utilizado:           {'NO' if not selection['test_used'] else 'SÍ'}")
    print("-" * 72)
    print("Shortlist seleccionada:")
    for s in selection["selected_for_ckks"]:
        print(
            f"  {s['configuration_id']:<24} acc={s['validation_accuracy']:.4f} "
            f"[{s['selection_category']}]"
        )
    print("-" * 72)
    for d in selection["diagnostic_baselines"]:
        print(f"  [diagnóstico] {d['configuration_id']} acc={d['validation_accuracy']:.4f}")
    print("-" * 72)
    print(f"Selección congelada: {FROZEN_SELECTION}")
    print(f"Manifiesto:          {SELECTION_MANIFEST}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
