"""
Evaluación final sobre TEST de la shortlist congelada (Hito 3C-D).

Este es el ÚNICO punto del proyecto donde se usa el conjunto de test. Requisitos:
    1. La selección debe estar congelada (test_used=false) y sus hashes de
       entrada deben coincidir (garantía anti-leakage).
    2. NO se re-selecciona nada: solo se evalúan ReLU + las configs elegidas.
    3. Opcionalmente se corren las 24 como reporte descriptivo, con la selección
       ya congelada.

Produce el Δ_aproximación final: la pérdida de accuracy atribuible exclusivamente
a la sustitución de ReLU por polinomios, sobre datos nunca vistos.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from ckks_benchmark.approximation.registry import (
    generate_all,
    load_and_validate_intervals,
)
from ckks_benchmark.model.benchmark_polynomial_cnn import (
    accuracy_and_f1,
    compute_metrics,
    group_into_triplets,
    run_inference,
)
from ckks_benchmark.model.polynomial_model import build_polynomial_model
from ckks_benchmark.model.preactivations import load_trained_model

FROZEN_SELECTION = Path("results/published/hito3c_frozen_selection.json")
INTERVALS_PATH = Path("results/published/preactivation_intervals.json")

INPUT_HASH_FILES = {
    "combined_analysis_csv": "results/tables/hito3c_combined_analysis.csv",
    "functional_metrics_csv": "results/tables/hito3b_functional_metrics.csv",
    "cnn_metrics_csv": "results/tables/hito3c_cnn_validation_metrics.csv",
}


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
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


def verify_frozen_selection() -> dict:
    """Puerta anti-leakage: confirma que la selección es válida para test."""
    if not FROZEN_SELECTION.exists():
        raise FileNotFoundError("No existe la selección congelada. Ejecuta 3C-C primero.")

    sel = json.loads(FROZEN_SELECTION.read_text(encoding="utf-8"))

    if sel.get("test_used") is not False:
        raise RuntimeError("La selección indica test_used != false. Abortando.")
    if sel.get("selection_status") != "frozen":
        raise RuntimeError("La selección no está congelada. Abortando.")

    # Verificar hashes de entradas.
    for key, path in INPUT_HASH_FILES.items():
        current = _sha256(path)
        frozen = sel["hashes"][key]
        if current != frozen:
            raise RuntimeError(
                f"Hash de {key} cambió desde el congelamiento. "
                f"La selección ya no corresponde a las entradas. Abortando."
            )

    return sel


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluación final sobre test (Hito 3C-D).")
    parser.add_argument("--checkpoint", default="models/reduced_lenet_relu_best.pt")
    parser.add_argument(
        "--full-report",
        action="store_true",
        help="Evaluar las 24 configuraciones sobre test (reporte descriptivo).",
    )
    parser.add_argument("--csv", default="results/tables/hito3c_cnn_test_metrics.csv")
    parser.add_argument("--json", default="results/published/hito3c_test_results.json")
    args = parser.parse_args()

    # PUERTA ANTI-LEAKAGE.
    print("=" * 72)
    print("EVALUACIÓN FINAL SOBRE TEST — Hito 3C-D")
    print("=" * 72)
    print("Verificando integridad de la selección congelada...")
    sel = verify_frozen_selection()
    selected_ids = {s["configuration_id"] for s in sel["selected_for_ckks"]}
    diagnostic_ids = {d["configuration_id"] for d in sel["diagnostic_baselines"]}
    print(
        f"  test_used=false, hashes OK. {len(selected_ids)} seleccionadas + "
        f"{len(diagnostic_ids)} diagnóstico."
    )
    print("-" * 72)

    from ckks_benchmark.model.train import TrainingConfig, create_dataloaders

    intervals = load_and_validate_intervals(INTERVALS_PATH)
    polys = generate_all(intervals)
    triplets = group_into_triplets(polys)

    config = TrainingConfig()
    _train, _val, test_loader = create_dataloaders(config)

    # Línea base ReLU sobre TEST.
    print("[BASE] ReducedLeNet + ReLU sobre TEST")
    base_model, _ckpt = load_trained_model(Path(args.checkpoint))
    relu_result = run_inference(base_model, test_loader, stop_on_non_finite=False)
    relu_acc, relu_f1 = accuracy_and_f1(relu_result.predictions, relu_result.labels)
    print(f"       accuracy={relu_acc:.4f} f1={relu_f1:.4f} loss={relu_result.loss:.4f}")
    print("-" * 72)

    # Configuraciones a evaluar: shortlist + diagnóstico (o las 24 si --full-report).
    to_evaluate = set(selected_ids) | set(diagnostic_ids)
    if args.full_report:
        to_evaluate = set(triplets.keys())

    results = []
    for config_id in sorted(to_evaluate):
        method, degree_s, interval = config_id.rsplit("_", 2)
        degree = int(degree_s[1:])
        model = build_polynomial_model(args.checkpoint, triplets[config_id])
        poly_result = run_inference(model, test_loader, stop_on_non_finite=True)
        metrics = compute_metrics(config_id, method, degree, interval, poly_result, relu_result)
        results.append(metrics)

        role = (
            "seleccionada"
            if config_id in selected_ids
            else "diagnóstico"
            if config_id in diagnostic_ids
            else "reporte"
        )
        acc_str = f"{metrics.validation_accuracy:.4f}" if metrics.valid else "  -   "
        status = "OK" if metrics.valid else f"INVÁLIDO ({metrics.first_non_finite_layer})"
        d_acc = f"Δ={metrics.delta_accuracy_vs_relu:+.4f}" if metrics.valid else ""
        print(f"  {config_id:<24} acc={acc_str} {d_acc:<12} {status} [{role}]")

    # Export.
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    metadata = {
        "phase": "Hito 3C-D",
        "split": "test",
        "test_used": True,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "relu_baseline_test": {"accuracy": relu_acc, "macro_f1": relu_f1, "loss": relu_result.loss},
        "frozen_selection_created_at": sel["created_at_utc"],
        "frozen_selection_git_commit": sel["git_commit"],
        "selected_ids": sorted(selected_ids),
        "full_report": args.full_report,
        "n_evaluated": len(results),
    }
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "results": [asdict(r) for r in results],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Resumen: shortlist con su Δ final.
    print("-" * 72)
    print("RESULTADO FINAL — Δ_aproximación sobre test (shortlist):")
    for r in results:
        if r.configuration_id in selected_ids and r.valid:
            print(
                f"  {r.configuration_id:<24} test_acc={r.validation_accuracy:.4f} "
                f"Δacc={r.delta_accuracy_vs_relu:+.4f} Δf1={r.delta_f1_vs_relu:+.4f}"
            )
    print("-" * 72)
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
