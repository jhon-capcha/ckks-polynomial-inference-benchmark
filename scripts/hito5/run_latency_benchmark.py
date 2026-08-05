"""
Benchmark de latencia del bloque final cifrado (Hito 5B).

Diseño (decisiones del Hito 5):
  - 6 configuraciones x 10 imágenes (1/clase, del subconjunto del Hito 4) x
    30 repeticiones + 3 warm-up.
  - Orden aleatorizado con seed guardada.
  - Timer online AISLADO: solo encrypt/act3/fc2/decrypt. Fuera del timer: carga
    de datos, modelos, cálculo de z (prefijo claro), serialización.
  - z claro precalculado y congelado antes de medir.
  - Contexto, claves y pesos reutilizados entre repeticiones (no se regeneran).
  - Setup medido en corrida separada.

Estadística: mediana y P95 principales; media/desv/CV complementarias.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from ckks_benchmark.approximation.registry import generate_all, load_and_validate_intervals
from ckks_benchmark.benchmark.timing import StageTimer, build_execution_order
from ckks_benchmark.he.ckks_context import create_context, load_server_context, split_client_server
from ckks_benchmark.he.client_server import CKKSClient, CKKSServer
from ckks_benchmark.he.staged_inference import compute_act3_preactivation, compute_final_block_clear
from ckks_benchmark.model.benchmark_polynomial_cnn import group_into_triplets
from ckks_benchmark.model.polynomial_model import build_polynomial_model
from ckks_benchmark.model.train import TrainingConfig, create_dataloaders

CKPT = "models/reduced_lenet_relu_best.pt"
INTERVALS = Path("results/published/preactivation_intervals.json")
SAMPLE_INDICES = Path("results/published/hito4_ckks_sample_indices.json")
OUT_RAW = Path("results/tables/hito5_latency_raw.csv")
OUT_SETUP = Path("results/tables/hito5_setup_latency.csv")

CONFIGURATIONS = [
    "chebyshev_d3_I1",
    "least_squares_d3_I1",
    "chebyshev_d5_I1",
    "least_squares_d5_I1",
    "chebyshev_d5_I2",
    "least_squares_d5_I2",
]
N_IMAGES = 10  # 1 por clase
N_REPETITIONS = 30
N_WARMUP = 3
ORDER_SEED = 20260804


def select_10_images(samples_all: list[dict]) -> list[dict]:
    """Toma 1 imagen por clase del subconjunto congelado del Hito 4."""
    by_class = {}
    for s in samples_all:
        c = s["true_label"]
        if c not in by_class:
            by_class[c] = s
    return [by_class[c] for c in sorted(by_class)]


def measure_setup(profile_id: str) -> dict:
    """Mide el setup criptográfico UNA vez (context/keys/rotation).

    Se mide una sola vez porque rotateKeyGen es costoso; el objetivo es el
    orden de magnitud del setup, no su distribución fina.
    """
    from Pyfhel import Pyfhel

    from ckks_benchmark.he.parameters import get_profile

    profile = get_profile(profile_id)
    he = Pyfhel()
    t0 = time.perf_counter()
    t_ctx = time.perf_counter()
    he.contextGen(
        scheme="ckks",
        n=profile.poly_modulus_degree,
        scale=2**profile.scale_bits,
        qi_sizes=list(profile.coeff_mod_bit_sizes),
    )
    ctx_s = time.perf_counter() - t_ctx
    t_k = time.perf_counter()
    he.keyGen()
    he.relinKeyGen()
    key_s = time.perf_counter() - t_k
    t_r = time.perf_counter()
    he.rotateKeyGen()
    rot_s = time.perf_counter() - t_r
    return {
        "context_gen": ctx_s,
        "key_gen": key_s,
        "rotation_key_gen": rot_s,
        "total_setup": time.perf_counter() - t0,
    }


def main() -> int:
    print("=" * 72)
    print("BENCHMARK DE LATENCIA — Hito 5B")
    print("=" * 72)

    # --- Preparación NO medida ---
    intervals = load_and_validate_intervals(INTERVALS)
    triplets = group_into_triplets(generate_all(intervals))
    _t, _v, test_loader = create_dataloaders(TrainingConfig())
    all_images, all_labels = [], []
    for images, labels in test_loader:
        all_images.append(images)
        all_labels.append(labels)
    all_images = torch.cat(all_images)

    samples_all = json.loads(SAMPLE_INDICES.read_text(encoding="utf-8"))["samples"]
    images_10 = select_10_images(samples_all)
    print(f"Imágenes: {len(images_10)} (1/clase)")

    # Precalcular z y logits claros por (config, imagen) — FUERA del timer.
    print("Precalculando z (preactivación act3) y referencias claras...")
    prep = {}  # (config_id, dataset_index) -> {z, clear_logits}
    contexts = {}  # config_id -> (client, server)
    setup_records = []

    for config_id in CONFIGURATIONS:
        poly_model = build_polynomial_model(CKPT, triplets[config_id])
        poly_model.eval()
        state = poly_model.state_dict()
        W = state[[k for k in state if "fc2" in k and "weight" in k][0]].numpy().astype(np.float64)
        b = state[[k for k in state if "fc2" in k and "bias" in k][0]].numpy().astype(np.float64)
        act3_coeffs = np.array(triplets[config_id]["act3"].coefficients, dtype=np.float64)
        degree = int(config_id.split("_d")[1][0])
        profile = "ckks_n16384_d5" if degree == 5 else "ckks_n16384_d3"

        # z por imagen.
        for s in images_10:
            idx = s["dataset_index"]
            img = all_images[idx : idx + 1]
            z = compute_act3_preactivation(poly_model, img).numpy()[0]
            clear_logits = compute_final_block_clear(
                poly_model, torch.tensor(z).unsqueeze(0)
            ).numpy()[0]
            prep[(config_id, idx)] = {
                "z": z,
                "clear_logits": clear_logits,
                "label": s["true_label"],
            }

        # Contexto + cliente/servidor (reutilizados). Setup medido aparte.
        setup = measure_setup(profile)
        setup_records.append({"configuration_id": config_id, "degree": degree, **setup})

        ctx = create_context(profile, generate_rotation_keys=True)
        client = CKKSClient(ctx)
        _secret, server_material = split_client_server(ctx)
        he_server = load_server_context(server_material)
        server = CKKSServer(he_server, act3_coeffs, W, b, logical_size=120)
        contexts[config_id] = (client, server)

    print(f"Setup medido para {len(CONFIGURATIONS)} configuraciones.")

    # Orden aleatorizado.
    order = build_execution_order(len(CONFIGURATIONS), N_IMAGES, N_REPETITIONS, ORDER_SEED)
    idx_to_image = [s["dataset_index"] for s in images_10]

    # --- Warm-up (por config, descartado) ---
    print(f"Warm-up ({N_WARMUP} por configuración)...")
    for config_id in CONFIGURATIONS:
        client, server = contexts[config_id]
        z = prep[(config_id, idx_to_image[0])]["z"]
        for _ in range(N_WARMUP):
            enc = client.encrypt_input(z)
            logit_cts = server.infer(enc)
            client.decrypt_logits(logit_cts)

    # --- Medición (1800 inferencias, orden aleatorizado, timer aislado) ---
    print(f"Midiendo {len(order)} inferencias...")
    raw = []
    for run_id, (c_idx, i_idx, rep) in enumerate(order):
        config_id = CONFIGURATIONS[c_idx]
        dataset_idx = idx_to_image[i_idx]
        client, server = contexts[config_id]
        data = prep[(config_id, dataset_idx)]
        z = data["z"]

        timer = StageTimer()
        with timer.measure("encrypt"):
            enc = client.encrypt_input(z)
        with timer.measure("act3"):
            act3_ct = server.infer_act3(enc)
        with timer.measure("fc2"):
            logit_cts = server.infer_fc2(act3_ct)
        with timer.measure("decrypt"):
            ckks_logits = client.decrypt_logits(logit_cts)

        online_total = timer.total()
        stage_sum = (
            timer.times["encrypt"]
            + timer.times["act3"]
            + timer.times["fc2"]
            + timer.times["decrypt"]
        )
        residual = online_total - stage_sum
        raw.append(
            {
                "run_id": run_id,
                "configuration_id": config_id,
                "dataset_index": dataset_idx,
                "true_label": data["label"],
                "repetition": rep,
                "execution_order": run_id,
                "encrypt_seconds": timer.times["encrypt"],
                "act3_seconds": timer.times["act3"],
                "fc2_seconds": timer.times["fc2"],
                "decrypt_seconds": timer.times["decrypt"],
                "online_total_seconds": online_total,
                "timing_residual_seconds": residual,
                "timing_residual_ratio": residual / online_total if online_total > 0 else 0.0,
            }
        )

        if (run_id + 1) % 100 == 0:
            print(f"  {run_id + 1}/{len(order)}...")

    # Export crudo.
    OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    with OUT_RAW.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(raw[0].keys()))
        w.writeheader()
        w.writerows(raw)

    with OUT_SETUP.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(setup_records[0].keys()))
        w.writeheader()
        w.writerows(setup_records)

    # Resumen rápido por configuración.
    print("-" * 72)
    print("Latencia online (mediana) por configuración:")
    for config_id in CONFIGURATIONS:
        times = [r["online_total_seconds"] for r in raw if r["configuration_id"] == config_id]
        print(
            f"  {config_id:<24} mediana={np.median(times) * 1000:.1f}ms "
            f"P95={np.percentile(times, 95) * 1000:.1f}ms"
        )
    print("-" * 72)
    print(f"Crudo: {OUT_RAW} ({len(raw)} filas)")
    print(f"Setup: {OUT_SETUP}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
