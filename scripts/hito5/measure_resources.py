"""
Consumo de recursos del bloque final cifrado (Hito 5C).

Mide tamaños en bytes (reproducibles, no memoria pico):
  - Material de sesión: contexto, clave pública, relin, rotación.
  - Ciphertexts: entrada (z), tras act3, salida de fc2 (10 logits).
  - Ratios de expansión: bytes cifrados / bytes claros.

Las claves de rotación son grandes (por eso su generación domina el setup);
este script lo cuantifica.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from ckks_benchmark.approximation.registry import generate_all, load_and_validate_intervals
from ckks_benchmark.he.ckks_context import create_context, load_server_context, split_client_server
from ckks_benchmark.he.client_server import CKKSClient, CKKSServer
from ckks_benchmark.he.staged_inference import compute_act3_preactivation
from ckks_benchmark.model.benchmark_polynomial_cnn import group_into_triplets
from ckks_benchmark.model.polynomial_model import build_polynomial_model
from ckks_benchmark.model.train import TrainingConfig, create_dataloaders

CKPT = "models/reduced_lenet_relu_best.pt"
INTERVALS = Path("results/published/preactivation_intervals.json")
SAMPLE_INDICES = Path("results/published/hito4_ckks_sample_indices.json")
OUT_CSV = Path("results/tables/hito5_resource_consumption.csv")

CONFIGURATIONS = [
    "chebyshev_d3_I1",
    "least_squares_d3_I1",
    "chebyshev_d5_I1",
    "least_squares_d5_I1",
    "chebyshev_d5_I2",
    "least_squares_d5_I2",
]


def ct_bytes(ct) -> int:
    """Tamaño serializado de un ciphertext."""
    try:
        return len(ct.to_bytes())
    except Exception:
        try:
            return int(ct.sizeof_ciphertext())
        except Exception:
            return -1


def key_bytes(he, method: str) -> int:
    """Tamaño serializado de una clave/contexto."""
    try:
        return len(getattr(he, f"to_bytes_{method}")())
    except Exception:
        return -1


def main() -> int:
    print("=" * 72)
    print("CONSUMO DE RECURSOS — Hito 5C")
    print("=" * 72)

    intervals = load_and_validate_intervals(INTERVALS)
    triplets = group_into_triplets(generate_all(intervals))
    _t, _v, test_loader = create_dataloaders(TrainingConfig())
    all_images = torch.cat([im for im, _ in test_loader])

    samples = json.loads(SAMPLE_INDICES.read_text(encoding="utf-8"))["samples"]
    # Una imagen para medir ciphertexts (los tamaños no dependen del contenido).
    idx0 = samples[0]["dataset_index"]

    # Tamaños claros de referencia (float64).
    z_clear_bytes = 120 * 8  # vector de 120 float64
    logits_clear_bytes = 10 * 8  # 10 logits float64

    results = []
    for config_id in CONFIGURATIONS:
        degree = int(config_id.split("_d")[1][0])
        profile = "ckks_n16384_d5" if degree == 5 else "ckks_n16384_d3"

        poly_model = build_polynomial_model(CKPT, triplets[config_id])
        poly_model.eval()
        state = poly_model.state_dict()
        W = state[[k for k in state if "fc2" in k and "weight" in k][0]].numpy().astype(np.float64)
        b = state[[k for k in state if "fc2" in k and "bias" in k][0]].numpy().astype(np.float64)
        act3_coeffs = np.array(triplets[config_id]["act3"].coefficients, dtype=np.float64)

        ctx = create_context(profile, generate_rotation_keys=True)
        client = CKKSClient(ctx)
        _secret, server_material = split_client_server(ctx)
        he_server = load_server_context(server_material)
        server = CKKSServer(he_server, act3_coeffs, W, b, logical_size=120)
        he = ctx.he

        # Material de sesión (bytes).
        context_b = key_bytes(he, "context")
        public_b = key_bytes(he, "public_key")
        relin_b = key_bytes(he, "relin_key")
        rotate_b = key_bytes(he, "rotate_key")
        eval_keys_b = public_b + relin_b + rotate_b

        # Ciphertexts.
        z = compute_act3_preactivation(poly_model, all_images[idx0 : idx0 + 1]).numpy()[0]
        enc_z = client.encrypt_input(z)
        input_ct_b = ct_bytes(enc_z)

        act3_ct = server.infer_act3(enc_z)
        act3_ct_b = ct_bytes(act3_ct)

        logit_cts = server.infer_fc2(act3_ct)
        output_cts_b = sum(ct_bytes(ct) for ct in logit_cts)

        results.append(
            {
                "configuration_id": config_id,
                "degree": degree,
                "method": config_id.split("_d")[0],
                "interval_name": config_id.split("_")[-1],
                "profile": profile,
                "context_bytes": context_b,
                "public_key_bytes": public_b,
                "relin_key_bytes": relin_b,
                "rotation_key_bytes": rotate_b,
                "evaluation_keys_bytes": eval_keys_b,
                "input_ciphertext_bytes": input_ct_b,
                "act3_ciphertext_bytes": act3_ct_b,
                "output_ciphertexts_bytes": output_cts_b,
                "n_output_ciphertexts": len(logit_cts),
                "z_clear_bytes": z_clear_bytes,
                "logits_clear_bytes": logits_clear_bytes,
                "input_expansion_ratio": input_ct_b / z_clear_bytes,
                "output_expansion_ratio": output_cts_b / logits_clear_bytes,
            }
        )

        print(f"\n### {config_id} (perfil {profile}) ###")
        print(
            f"  claves rotación: {rotate_b / 1e6:.1f} MB | relin: {relin_b / 1e6:.1f} MB | "
            f"pública: {public_b / 1e6:.2f} MB"
        )
        print(
            f"  ciphertext entrada: {input_ct_b / 1e6:.2f} MB "
            f"(expansión {input_ct_b / z_clear_bytes:.0f}x vs {z_clear_bytes}B claro)"
        )
        print(
            f"  10 logits cifrados: {output_cts_b / 1e6:.2f} MB "
            f"(expansión {output_cts_b / logits_clear_bytes:.0f}x)"
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print("\n" + "=" * 72)
    print("RESUMEN por grado (perfil):")
    for degree in (3, 5):
        r = next(x for x in results if x["degree"] == degree)
        print(
            f"  grado {degree} ({r['profile']}): "
            f"rotación={r['rotation_key_bytes'] / 1e6:.1f}MB, "
            f"ct_entrada={r['input_ciphertext_bytes'] / 1e6:.2f}MB, "
            f"salida={r['output_ciphertexts_bytes'] / 1e6:.2f}MB"
        )
    print(f"\nCSV: {OUT_CSV}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
