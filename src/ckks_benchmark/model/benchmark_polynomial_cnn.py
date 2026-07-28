"""
Benchmark de la CNN polinómica sobre validación (Hito 3C-B).

Agrupa los 72 polinomios en 24 ternas (método × grado × intervalo), construye
cada CNN, y la evalúa sobre validación midiendo accuracy, F1 macro, loss,
Δ vs ReLU, cambios de predicción e instrumentación ligera por capa.

Disciplina anti-leakage: SOLO validación aquí. El test se reserva para 3C-D.

Manejo de no-finitud (esperado en Taylor grado alto): al detectar el primer
logit no finito, la configuración se marca inválida, se registra la capa donde
empezó el colapso, y el benchmark continúa con la siguiente. Ninguna
configuración desaparece.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ckks_benchmark.approximation.base import PolynomialApproximation
from ckks_benchmark.approximation.registry import (
    generate_all,
    load_and_validate_intervals,
)
from ckks_benchmark.model.activation_monitor import ActivationMonitor
from ckks_benchmark.model.polynomial_model import build_polynomial_model
from ckks_benchmark.model.preactivations import load_trained_model

METHODS = ("chebyshev", "least_squares", "taylor")
DEGREES = (3, 5, 7, 9)
INTERVALS = ("I1", "I2")
NUM_CLASSES = 10


# --------------------------------------------------------------------------- #
# Agrupación de las 72 aproximaciones en 24 ternas
# --------------------------------------------------------------------------- #
def group_into_triplets(
    polys: dict[str, PolynomialApproximation],
) -> dict[str, dict[str, PolynomialApproximation]]:
    """Agrupa los 72 polinomios en 24 ternas act1/act2/act3."""
    triplets: dict[str, dict[str, PolynomialApproximation]] = {}
    for method in METHODS:
        for degree in DEGREES:
            for interval in INTERVALS:
                config_id = f"{method}_d{degree}_{interval}"
                triplet = {}
                for act in ("act1", "act2", "act3"):
                    key = f"{act}_{method}_d{degree}_{interval}"
                    if key not in polys:
                        raise KeyError(f"Falta el polinomio {key}.")
                    triplet[act] = polys[key]
                triplets[config_id] = triplet

    if len(triplets) != 24:
        raise RuntimeError(f"Se esperaban 24 ternas, se formaron {len(triplets)}.")
    return triplets


# --------------------------------------------------------------------------- #
# Evaluación de un modelo sobre un split
# --------------------------------------------------------------------------- #
@dataclass
class InferenceResult:
    predictions: np.ndarray
    labels: np.ndarray
    logits_first_batch: np.ndarray | None
    loss: float
    non_finite_logits: int
    first_non_finite_layer: str | None
    layer_stats: dict[str, dict[str, float]]
    processed_images: int


def run_inference(
    model: nn.Module,
    loader,
    stop_on_non_finite: bool = True,
) -> InferenceResult:
    """Ejecuta inferencia acumulando predicciones, loss e instrumentación.

    Si stop_on_non_finite y aparece un logit no finito, detiene la inferencia
    de esta configuración (ya es inválida) tras registrar la causa.
    """
    monitor = ActivationMonitor(model)
    monitor.register()

    all_preds, all_labels = [], []
    total_loss = 0.0
    total_n = 0
    non_finite = 0
    logits_first = None
    processed = 0
    ce = nn.CrossEntropyLoss(reduction="sum")

    with torch.inference_mode():
        for batch_idx, (x, y) in enumerate(loader):
            logits = model(x)
            if batch_idx == 0:
                logits_first = logits.detach().cpu().numpy()

            batch_non_finite = int((~torch.isfinite(logits)).sum().item())
            if batch_non_finite > 0:
                non_finite += batch_non_finite
                if stop_on_non_finite:
                    processed += x.shape[0]
                    break

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            total_loss += float(ce(logits, y).item())
            total_n += x.shape[0]
            processed += x.shape[0]

    monitor.remove()

    predictions = np.concatenate(all_preds) if all_preds else np.array([])
    labels = np.concatenate(all_labels) if all_labels else np.array([])
    loss = total_loss / total_n if total_n > 0 else float("nan")

    return InferenceResult(
        predictions=predictions,
        labels=labels,
        logits_first_batch=logits_first,
        loss=loss,
        non_finite_logits=non_finite,
        first_non_finite_layer=monitor.first_non_finite_layer,
        layer_stats=monitor.summary(),
        processed_images=processed,
    )


def accuracy_and_f1(preds: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Accuracy y F1 macro sin sklearn (matriz de confusión con bincount)."""
    if preds.size == 0:
        return float("nan"), float("nan")
    accuracy = float(np.mean(preds == labels))

    # Matriz de confusión.
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    f1s = []
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return accuracy, float(np.mean(f1s))


# --------------------------------------------------------------------------- #
# Métricas por configuración
# --------------------------------------------------------------------------- #
@dataclass
class ConfigurationMetrics:
    configuration_id: str
    method: str
    degree: int
    interval_name: str
    valid: bool
    invalid_reason: str | None
    validation_loss: float
    validation_accuracy: float
    validation_macro_f1: float
    delta_loss_vs_relu: float
    delta_accuracy_vs_relu: float
    delta_f1_vs_relu: float
    prediction_change_count: int
    prediction_change_fraction: float
    relu_correct_poly_wrong: int
    relu_wrong_poly_correct: int
    logit_mae: float
    logit_rmse: float
    logit_max_error: float
    act1_max_abs: float
    act2_max_abs: float
    act3_max_abs: float
    logits_max_abs: float
    act1_non_finite: int
    act2_non_finite: int
    act3_non_finite: int
    logits_non_finite: int
    first_non_finite_layer: str | None
    processed_images: int


def compute_metrics(
    config_id: str,
    method: str,
    degree: int,
    interval: str,
    poly_result: InferenceResult,
    relu_result: InferenceResult,
) -> ConfigurationMetrics:
    ls = poly_result.layer_stats
    valid = poly_result.non_finite_logits == 0

    if valid:
        acc, f1 = accuracy_and_f1(poly_result.predictions, relu_result.labels)
        loss = poly_result.loss
        relu_acc, relu_f1 = accuracy_and_f1(relu_result.predictions, relu_result.labels)

        # Cambios de predicción (solo si ambas cubren las mismas imágenes).
        pc = int(np.sum(poly_result.predictions != relu_result.predictions))
        pf = pc / relu_result.predictions.size
        rc_pw = int(
            np.sum(
                (relu_result.predictions == relu_result.labels)
                & (poly_result.predictions != relu_result.labels)
            )
        )
        rw_pc = int(
            np.sum(
                (relu_result.predictions != relu_result.labels)
                & (poly_result.predictions == relu_result.labels)
            )
        )

        # Diferencia de logits (primer batch, común).
        lp = poly_result.logits_first_batch
        lr = relu_result.logits_first_batch
        diff = np.abs(lp - lr)
        logit_mae = float(np.mean(diff))
        logit_rmse = float(np.sqrt(np.mean(diff**2)))
        logit_max = float(np.max(diff))

        d_loss = loss - relu_result.loss
        d_acc = relu_acc - acc
        d_f1 = relu_f1 - f1
        reason = None
    else:
        acc = f1 = loss = float("nan")
        d_loss = d_acc = d_f1 = float("nan")
        pc = rc_pw = rw_pc = 0
        pf = float("nan")
        logit_mae = logit_rmse = logit_max = float("nan")
        reason = "non-finite logits detected"

    return ConfigurationMetrics(
        configuration_id=config_id,
        method=method,
        degree=degree,
        interval_name=interval,
        valid=valid,
        invalid_reason=reason,
        validation_loss=loss,
        validation_accuracy=acc,
        validation_macro_f1=f1,
        delta_loss_vs_relu=d_loss,
        delta_accuracy_vs_relu=d_acc,
        delta_f1_vs_relu=d_f1,
        prediction_change_count=pc,
        prediction_change_fraction=pf,
        relu_correct_poly_wrong=rc_pw,
        relu_wrong_poly_correct=rw_pc,
        logit_mae=logit_mae,
        logit_rmse=logit_rmse,
        logit_max_error=logit_max,
        act1_max_abs=ls["act1"]["max_abs"],
        act2_max_abs=ls["act2"]["max_abs"],
        act3_max_abs=ls["act3"]["max_abs"],
        logits_max_abs=ls["logits"]["max_abs"],
        act1_non_finite=ls["act1"]["non_finite_count"],
        act2_non_finite=ls["act2"]["non_finite_count"],
        act3_non_finite=ls["act3"]["non_finite_count"],
        logits_non_finite=ls["logits"]["non_finite_count"],
        first_non_finite_layer=poly_result.first_non_finite_layer,
        processed_images=poly_result.processed_images,
    )


# --------------------------------------------------------------------------- #
# Utilidades de export
# --------------------------------------------------------------------------- #
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark CNN polinómica (Hito 3C-B).")
    parser.add_argument("--checkpoint", default="models/reduced_lenet_relu_best.pt")
    parser.add_argument(
        "--polynomials",
        default="results/approximations/coefficients/polynomials.json",
    )
    parser.add_argument("--csv", default="results/tables/hito3c_cnn_validation_metrics.csv")
    parser.add_argument("--json", default="results/published/hito3c_validation_results.json")
    parser.add_argument("--manifest", default="results/published/hito3c_validation_manifest.json")
    args = parser.parse_args()

    from ckks_benchmark.model.train import TrainingConfig, create_dataloaders

    intervals = load_and_validate_intervals(Path("results/published/preactivation_intervals.json"))
    polys = generate_all(intervals)
    triplets = group_into_triplets(polys)

    config = TrainingConfig()
    _train, val_loader, _test = create_dataloaders(config)

    # Línea base ReLU sobre validación.
    print("=" * 72)
    print("BENCHMARK CNN POLINÓMICA — Hito 3C-B (validación)")
    print("=" * 72)
    print("[BASE] ReducedLeNet + ReLU")
    base_model, _ckpt = load_trained_model(Path(args.checkpoint))
    relu_result = run_inference(base_model, val_loader, stop_on_non_finite=False)
    relu_acc, relu_f1 = accuracy_and_f1(relu_result.predictions, relu_result.labels)
    print(f"       accuracy={relu_acc:.4f} f1={relu_f1:.4f} loss={relu_result.loss:.4f}")

    # 24 configuraciones.
    results = []
    start = time.perf_counter()
    for i, (config_id, triplet) in enumerate(sorted(triplets.items()), start=1):
        method, degree_s, interval = config_id.rsplit("_", 2)
        degree = int(degree_s[1:])
        model = build_polynomial_model(args.checkpoint, triplet)
        poly_result = run_inference(model, val_loader, stop_on_non_finite=True)
        metrics = compute_metrics(config_id, method, degree, interval, poly_result, relu_result)
        results.append(metrics)

        status = "OK" if metrics.valid else f"INVÁLIDO ({metrics.first_non_finite_layer})"
        acc_str = f"{metrics.validation_accuracy:.4f}" if metrics.valid else "  -   "
        print(f"[{i:02d}/24] {config_id:<24} acc={acc_str} {status}")
    elapsed = time.perf_counter() - start

    # Export CSV.
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # Metadata.
    metadata = {
        "generated_at": datetime.now().isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "checkpoint": args.checkpoint,
        "polynomials_sha256": _sha256_of_file(Path(args.polynomials)),
        "split": "validation",
        "relu_baseline": {"accuracy": relu_acc, "macro_f1": relu_f1, "loss": relu_result.loss},
        "n_configurations": len(results),
        "execution_seconds": round(elapsed, 2),
    }

    # Export JSON.
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    valid_ids = [r.configuration_id for r in results if r.valid]
    invalid_ids = [r.configuration_id for r in results if not r.valid]
    json_path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "valid_configurations": valid_ids,
                "invalid_configurations": invalid_ids,
                "results": [asdict(r) for r in results],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Manifiesto.
    Path(args.manifest).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    n_valid = len(valid_ids)
    print("-" * 72)
    print(f"Configuraciones: {len(results)} | válidas: {n_valid} | inválidas: {len(invalid_ids)}")
    print(f"Tiempo: {elapsed:.1f}s")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
