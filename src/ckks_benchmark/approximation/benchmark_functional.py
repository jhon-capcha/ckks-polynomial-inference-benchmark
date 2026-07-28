"""
Ejecutor funcional del Hito 3B: evalúa las 72 configuraciones polinómicas.

Carga los polinomios del Hito 3A y las preactivaciones de validación, evalúa
cada configuración (error uniforme, empírico, dentro/fuera del intervalo, origen,
diagnósticos), y exporta:
    results/tables/hito3b_functional_metrics.csv        (tabla maestra)
    results/published/hito3b_functional_metrics.json    (estructura + resúmenes)
    results/published/hito3b_manifest.json              (trazabilidad)

Ninguna configuración desaparece: las inválidas quedan con valid=false y razón.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ckks_benchmark.approximation.base import PolynomialApproximation
from ckks_benchmark.approximation.evaluation import (
    ORIGIN_HALF_WIDTH,
    UNIFORM_GRID_POINTS,
    ApproximationMetrics,
    evaluate_configuration,
)

ACTIVATIONS = ("act1", "act2", "act3")
METHODS = ("taylor", "chebyshev", "least_squares")


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


def load_polynomials(path: Path) -> list[PolynomialApproximation]:
    """Reconstruye los PolynomialApproximation desde polynomials.json."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró {path}. Ejecuta el Hito 3A (registry).")

    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data["polynomials"]

    polynomials = []
    for e in entries:
        poly = PolynomialApproximation(
            method=e["method"],
            activation=e["activation"],
            interval_name=e["interval_name"],
            interval=(e["interval"][0], e["interval"][1]),
            degree=e["degree"],
            coefficients=np.array(e["coefficients"], dtype=np.float64),
            coefficient_basis=e["coefficient_basis"],
            metadata=e.get("metadata", {}),
        )
        polynomials.append(poly)

    if len(polynomials) != 72:
        raise ValueError(f"Se esperaban 72 polinomios, se cargaron {len(polynomials)}.")
    return polynomials


def load_samples(npz_path: Path) -> dict[str, np.ndarray]:
    """Carga las preactivaciones de validación, validando integridad."""
    if not npz_path.exists():
        raise FileNotFoundError(f"No se encontró {npz_path}. Ejecuta sample_preactivations.")

    data = np.load(npz_path)
    samples = {}
    for act in ACTIVATIONS:
        if act not in data:
            raise KeyError(f"Falta '{act}' en el .npz de muestras.")
        arr = data[act].astype(np.float64)
        if arr.size == 0:
            raise ValueError(f"{act}: muestra vacía.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{act}: contiene NaN o infinito.")
        samples[act] = arr
    return samples


def run_benchmark(
    polynomials: list[PolynomialApproximation],
    samples: dict[str, np.ndarray],
) -> list[ApproximationMetrics]:
    """Evalúa las 72 configuraciones, mostrando progreso."""
    results = []
    total = len(polynomials)
    for i, poly in enumerate(polynomials, start=1):
        metrics = evaluate_configuration(poly, samples[poly.activation])
        results.append(metrics)
        status = "OK" if metrics.valid else "INVÁLIDO"
        viable = "" if metrics.practically_viable else " [no viable]"
        print(f"[{i:02d}/{total}] {metrics.configuration_id:<30} {status}{viable}")
    return results


def export_csv(results: list[ApproximationMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(results[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def _summary_by_key(results: list[ApproximationMetrics], key: str) -> dict[str, dict[str, Any]]:
    """Resumen de MAE empírico agrupado por un atributo (method/activation/etc.)."""
    groups: dict[str, list[ApproximationMetrics]] = {}
    for r in results:
        if r.valid:
            groups.setdefault(str(getattr(r, key)), []).append(r)

    summary = {}
    for k, items in groups.items():
        maes = [r.mae_empirical for r in items]
        best = min(items, key=lambda r: r.mae_empirical)
        summary[k] = {
            "count": len(items),
            "mae_empirical_mean": float(np.mean(maes)),
            "mae_empirical_median": float(np.median(maes)),
            "mae_empirical_min": float(np.min(maes)),
            "mae_empirical_max": float(np.max(maes)),
            "best_configuration": best.configuration_id,
        }
    return summary


def export_json(
    results: list[ApproximationMetrics],
    path: Path,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    invalid = [r.configuration_id for r in results if not r.valid]
    not_viable = [r.configuration_id for r in results if r.valid and not r.practically_viable]
    payload = {
        "metadata": metadata,
        "summary_by_method": _summary_by_key(results, "method"),
        "summary_by_activation": _summary_by_key(results, "activation"),
        "summary_by_degree": _summary_by_key(results, "degree"),
        "summary_by_interval": _summary_by_key(results, "interval_name"),
        "invalid_configurations": invalid,
        "not_practically_viable": not_viable,
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_manifest(path: Path, metadata: dict[str, Any], elapsed: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {**metadata, "execution_seconds": round(elapsed, 3)}
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación funcional del Hito 3B.")
    parser.add_argument(
        "--polynomials",
        type=str,
        default="results/approximations/coefficients/polynomials.json",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="data/processed/preactivations_validation_sample.npz",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="results/tables/hito3b_functional_metrics.csv",
    )
    parser.add_argument(
        "--json",
        type=str,
        default="results/published/hito3b_functional_metrics.json",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="results/published/hito3b_manifest.json",
    )
    args = parser.parse_args()

    poly_path = Path(args.polynomials)
    npz_path = Path(args.samples)

    polynomials = load_polynomials(poly_path)
    samples = load_samples(npz_path)

    metadata = {
        "generated_at": datetime.now().isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "polynomials_source": str(poly_path).replace("\\", "/"),
        "polynomials_sha256": _sha256_of_file(poly_path),
        "samples_source": str(npz_path).replace("\\", "/"),
        "samples_sha256": _sha256_of_file(npz_path),
        "uniform_grid_points": UNIFORM_GRID_POINTS,
        "origin_half_width": ORIGIN_HALF_WIDTH,
        "n_configurations": len(polynomials),
    }

    print("=" * 72)
    print("EVALUACIÓN FUNCIONAL — Hito 3B")
    print("=" * 72)
    start = time.perf_counter()
    results = run_benchmark(polynomials, samples)
    elapsed = time.perf_counter() - start

    export_csv(results, Path(args.csv))
    export_json(results, Path(args.json), metadata)
    export_manifest(Path(args.manifest), metadata, elapsed)

    n_valid = sum(1 for r in results if r.valid)
    n_invalid = len(results) - n_valid
    n_not_viable = sum(1 for r in results if r.valid and not r.practically_viable)

    print("-" * 72)
    print(f"Configuraciones procesadas: {len(results)}")
    print(f"Válidas:                    {n_valid}")
    print(f"Inválidas:                  {n_invalid}")
    print(f"Válidas no viables (práct): {n_not_viable}")
    print(f"Tiempo:                     {elapsed:.2f}s")
    print("-" * 72)
    print(f"CSV:       {args.csv}")
    print(f"JSON:      {args.json}")
    print(f"Manifiesto: {args.manifest}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
