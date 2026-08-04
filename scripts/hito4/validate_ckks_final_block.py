"""
Validación del bloque final cifrado sobre imágenes MNIST reales (Hito 4F).

Para cada configuración oficial de act3 (Chebyshev/LSQ, grados 3 y 5, I1/I2) y
cada imagen de una muestra estratificada (100 imágenes, 10 por clase, seed
congelada), compara tres rutas:
  - ReLU clara (CNN original): error total de referencia
  - Polinómica clara (act1/act2/act3 polinómicas): referencia del bloque CKKS
  - Polinómica con bloque act3->fc2 cifrado: la ruta CKKS

Mide Delta_CKKS = polinómica clara - bloque CKKS, aislado del Delta_aproximación.

test_used=true (evaluación), test_used_for_selection=false (la selección se
congeló en el Hito 3C; no se reajusta nada con estos resultados).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from ckks_benchmark.approximation.registry import (
    generate_all,
    load_and_validate_intervals,
)
from ckks_benchmark.he.ckks_context import (
    create_context,
    load_server_context,
    split_client_server,
)
from ckks_benchmark.he.client_server import CKKSClient, CKKSServer
from ckks_benchmark.he.staged_inference import (
    compute_act3_preactivation,
    compute_final_block_clear,
    verify_staged_equivalence,
)
from ckks_benchmark.model.benchmark_polynomial_cnn import group_into_triplets
from ckks_benchmark.model.polynomial_model import build_polynomial_model
from ckks_benchmark.model.preactivations import load_trained_model
from ckks_benchmark.model.train import TrainingConfig, create_dataloaders

CKPT = "models/reduced_lenet_relu_best.pt"
INTERVALS = Path("results/published/preactivation_intervals.json")
SAMPLE_INDICES = Path("results/published/hito4_ckks_sample_indices.json")
OUT_CSV = Path("results/tables/hito4_ckks_image_results.csv")
OUT_SUMMARY = Path("results/tables/hito4_ckks_validation_summary.csv")

CONFIGURATIONS = [
    "chebyshev_d3_I1",
    "least_squares_d3_I1",
    "chebyshev_d5_I1",
    "least_squares_d5_I1",
    "chebyshev_d5_I2",
    "least_squares_d5_I2",
]
SEED = 20260803
IMAGES_PER_CLASS = 10


def build_stratified_sample(test_loader) -> list[dict]:
    """Selecciona IMAGES_PER_CLASS imágenes por clase, reproducible."""
    if SAMPLE_INDICES.exists():
        data = json.loads(SAMPLE_INDICES.read_text(encoding="utf-8"))
        return data["samples"]

    # Recolectar todas las imágenes con sus etiquetas.
    all_images, all_labels = [], []
    for images, labels in test_loader:
        all_images.append(images)
        all_labels.append(labels)
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)

    rng = np.random.default_rng(SEED)
    samples = []
    for cls in range(10):
        idx = np.where(all_labels.numpy() == cls)[0]
        chosen = rng.choice(idx, size=IMAGES_PER_CLASS, replace=False)
        for i in chosen:
            samples.append({"dataset_index": int(i), "true_label": cls})
    samples.sort(key=lambda s: s["dataset_index"])

    SAMPLE_INDICES.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_INDICES.write_text(
        json.dumps(
            {
                "seed": SEED,
                "images_per_class": IMAGES_PER_CLASS,
                "split": "test",
                "n_total": len(samples),
                "samples": samples,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return samples


def main() -> int:
    print("=" * 72)
    print("VALIDACIÓN CKKS DEL BLOQUE FINAL SOBRE IMÁGENES — Hito 4F")
    print("=" * 72)

    # Datos y referencias.
    intervals = load_and_validate_intervals(INTERVALS)
    triplets = group_into_triplets(generate_all(intervals))
    _t, _v, test_loader = create_dataloaders(TrainingConfig())

    # Todas las imágenes indexables.
    all_images, all_labels = [], []
    for images, labels in test_loader:
        all_images.append(images)
        all_labels.append(labels)
    all_images = torch.cat(all_images)
    all_labels = torch.cat(all_labels)

    samples = build_stratified_sample(test_loader)
    print(f"Muestra estratificada: {len(samples)} imágenes ({IMAGES_PER_CLASS}/clase)")

    # Modelo ReLU (referencia total).
    relu_model, _ = load_trained_model(Path(CKPT))
    relu_model.eval()

    per_image = []
    summary = []

    for config_id in CONFIGURATIONS:
        print(f"\n### {config_id} ###")
        poly_model = build_polynomial_model(CKPT, triplets[config_id])
        poly_model.eval()

        # Verificación de equivalencia estructural (una vez por config).
        img0 = all_images[samples[0]["dataset_index"] : samples[0]["dataset_index"] + 1]
        eq = verify_staged_equivalence(poly_model, img0)
        if not eq["equivalent"]:
            print(f"  ADVERTENCIA: equivalencia por etapas falló (diff={eq['max_abs_diff']:.2e})")
        else:
            print(f"  Equivalencia por etapas OK (diff={eq['max_abs_diff']:.2e})")

        # Coeficientes act3 y pesos fc2.
        act3_coeffs = np.array(triplets[config_id]["act3"].coefficients, dtype=np.float64)
        state = poly_model.state_dict()
        W = state[[k for k in state if "fc2" in k and "weight" in k][0]].numpy().astype(np.float64)
        b = state[[k for k in state if "fc2" in k and "bias" in k][0]].numpy().astype(np.float64)

        # Contexto CKKS (perfil según grado) + cliente/servidor.
        degree = int(config_id.split("_d")[1][0])
        profile = "ckks_n16384_d5" if degree == 5 else "ckks_n16384_d3"
        ctx = create_context(profile, generate_rotation_keys=True)
        client = CKKSClient(ctx)
        _secret, server_material = split_client_server(ctx)
        he_server = load_server_context(server_material)
        server = CKKSServer(he_server, act3_coeffs, W, b, logical_size=120)

        relu_correct = poly_correct = ckks_correct = 0
        poly_ckks_agree = 0
        logit_maes = []

        for s in samples:
            idx, label = s["dataset_index"], s["true_label"]
            img = all_images[idx : idx + 1]

            # Ruta ReLU.
            with torch.no_grad():
                relu_logits = relu_model(img).numpy()[0]
            relu_pred = int(np.argmax(relu_logits))

            # Ruta polinómica clara.
            z = compute_act3_preactivation(poly_model, img)
            poly_logits = compute_final_block_clear(poly_model, z).numpy()[0]
            poly_pred = int(np.argmax(poly_logits))

            # Ruta CKKS.
            t0 = time.perf_counter()
            enc_z = client.encrypt_input(z.numpy()[0])
            logit_cts = server.infer(enc_z)
            ckks_logits = client.decrypt_logits(logit_cts)
            total_s = time.perf_counter() - t0
            ckks_pred = int(np.argmax(ckks_logits))

            logit_mae = float(np.mean(np.abs(poly_logits - ckks_logits)))
            logit_maes.append(logit_mae)

            relu_correct += relu_pred == label
            poly_correct += poly_pred == label
            ckks_correct += ckks_pred == label
            poly_ckks_agree += poly_pred == ckks_pred

            # Márgenes (top1 - top2).
            def margin(logits):
                s = np.sort(logits)[::-1]
                return float(s[0] - s[1])

            per_image.append(
                {
                    "configuration_id": config_id,
                    "dataset_index": idx,
                    "true_label": label,
                    "relu_prediction": relu_pred,
                    "polynomial_clear_prediction": poly_pred,
                    "ckks_prediction": ckks_pred,
                    "relu_correct": int(relu_pred == label),
                    "polynomial_clear_correct": int(poly_pred == label),
                    "ckks_correct": int(ckks_pred == label),
                    "prediction_poly_ckks_equal": int(poly_pred == ckks_pred),
                    "logit_mae_ckks": logit_mae,
                    "logit_max_error_ckks": float(np.max(np.abs(poly_logits - ckks_logits))),
                    "clear_margin": margin(poly_logits),
                    "ckks_margin": margin(ckks_logits),
                    "total_seconds": round(total_s, 4),
                }
            )

        n = len(samples)
        summary.append(
            {
                "configuration_id": config_id,
                "degree": degree,
                "method": config_id.split("_d")[0],
                "interval": config_id.split("_")[-1],
                "n_images": n,
                "relu_accuracy": relu_correct / n,
                "polynomial_clear_accuracy": poly_correct / n,
                "ckks_accuracy": ckks_correct / n,
                "delta_approximation_accuracy": (relu_correct - poly_correct) / n,
                "delta_ckks_accuracy": (poly_correct - ckks_correct) / n,
                "delta_total_accuracy": (relu_correct - ckks_correct) / n,
                "prediction_agreement_poly_ckks": poly_ckks_agree / n,
                "mean_logit_mae_ckks": float(np.mean(logit_maes)),
                "p95_logit_mae_ckks": float(np.percentile(logit_maes, 95)),
                "max_logit_mae_ckks": float(np.max(logit_maes)),
            }
        )
        print(
            f"  acc: ReLU={relu_correct / n:.3f} poly={poly_correct / n:.3f} "
            f"ckks={ckks_correct / n:.3f} | agree_poly_ckks={poly_ckks_agree / n:.3f} "
            f"| logit_MAE={np.mean(logit_maes):.2e}"
        )

    # Export.
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        w.writeheader()
        w.writerows(per_image)
    with OUT_SUMMARY.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print("\n" + "=" * 72)
    print("RESUMEN Δ_CKKS por configuración:")
    for s in summary:
        print(
            f"  {s['configuration_id']:<24} Δ_ckks_acc={s['delta_ckks_accuracy']:+.3f} "
            f"agree={s['prediction_agreement_poly_ckks']:.3f} "
            f"logit_MAE={s['mean_logit_mae_ckks']:.2e}"
        )
    print(f"\nCSV: {OUT_CSV}")
    print(f"Summary: {OUT_SUMMARY}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
