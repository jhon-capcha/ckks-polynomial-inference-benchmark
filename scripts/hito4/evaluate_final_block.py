"""
Evaluación del bloque lineal final cifrado act3 -> fc2 (Hito 4D).

Flujo oficial: preactivación de act3 (vector 120, en claro) -> cifrar SIMD ->
act3 polinómica CKKS (Horner) -> fc2 CKKS -> descifrar 10 logits.

Mide el Delta_CKKS del bloque: logits polinómicos claros vs logits CKKS.
Grados 3 y 5, ambos métodos. Registra niveles, error de logits y coincidencia
de predicción.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ckks_benchmark.he.ciphertext_state import OperationRecorder
from ckks_benchmark.he.ckks_context import create_context
from ckks_benchmark.he.linear_layer import decrypt_logits, encrypted_linear_simd
from ckks_benchmark.he.parameters import profile_for_degree
from ckks_benchmark.he.polynomial_evaluator import evaluate_polynomial_ckks
from ckks_benchmark.model.preactivations import load_trained_model

POLYS_PATH = Path("results/approximations/coefficients/polynomials.json")
FROZEN_PATH = Path("results/published/hito3c_frozen_selection.json")
SAMPLE_PATH = Path("data/processed/preactivations_validation_sample.npz")
CKPT_PATH = Path("models/reduced_lenet_relu_best.pt")
OUT_CSV = Path("results/tables/hito4_final_block_ckks.csv")

OFFICIAL_DEGREES = {3, 5}
N_SAMPLES = 120  # un ciphertext SIMD


def load_fc2_weights():
    model, _ = load_trained_model(CKPT_PATH)
    state = model.state_dict()
    W = state[[k for k in state if "fc2" in k and "weight" in k][0]].numpy().astype(np.float64)
    b = state[[k for k in state if "fc2" in k and "bias" in k][0]].numpy().astype(np.float64)
    return W, b


def main() -> int:
    print("=" * 72)
    print("BLOQUE FINAL CIFRADO act3 -> fc2 — Hito 4D")
    print("=" * 72)

    W, b = load_fc2_weights()
    polys = json.loads(POLYS_PATH.read_text(encoding="utf-8"))
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    selected = {s["configuration_id"] for s in frozen["selected_for_ckks"]}

    # Polinomios act3 de la shortlist, grados 3 y 5.
    act3 = [
        p
        for p in polys["polynomials"]
        if p["activation"] == "act3"
        and p["degree"] in OFFICIAL_DEGREES
        and f"{p['method']}_d{p['degree']}_{p['interval_name']}" in selected
    ]

    samples = np.load(SAMPLE_PATH)["act3"]
    rng = np.random.default_rng(20260803)
    z = samples[rng.choice(len(samples), N_SAMPLES, replace=False)].astype(np.float64)

    results = []
    for poly in sorted(act3, key=lambda p: (p["degree"], p["method"], p["interval_name"])):
        coeffs = np.array(poly["coefficients"], dtype=np.float64)
        degree = poly["degree"]
        profile = profile_for_degree(degree)

        # Referencia clara.
        act3_clear = np.polyval(coeffs[::-1], z)
        logits_clear = W @ act3_clear + b

        # CKKS.
        ctx = create_context(profile.profile_id, generate_rotation_keys=True)
        he = ctx.he
        slots = he.get_nSlots()
        z_packed = np.zeros(slots)
        z_packed[:N_SAMPLES] = z
        rec = OperationRecorder()
        try:
            ct = he.encryptFrac(z_packed)
            act3_ct = evaluate_polynomial_ckks(he, ct, coeffs, strategy="horner", recorder=rec)
            logit_cts = encrypted_linear_simd(
                he, act3_ct, W, b, logical_size=N_SAMPLES, recorder=rec
            )
            logits_ckks = decrypt_logits(he, logit_cts)
            err = np.abs(logits_clear - logits_ckks)
            summ = rec.summary()
            feasible = bool(np.all(np.isfinite(logits_ckks)))
            pred_match = int(np.argmax(logits_clear) == np.argmax(logits_ckks))
            results.append(
                {
                    "configuration_id": f"{poly['method']}_d{degree}_{poly['interval_name']}",
                    "activation": "act3",
                    "method": poly["method"],
                    "degree": degree,
                    "interval_name": poly["interval_name"],
                    "ckks_feasible": feasible,
                    "logit_mae": float(np.mean(err)),
                    "logit_rmse": float(np.sqrt(np.mean(err**2))),
                    "logit_max_error": float(np.max(err)),
                    "levels_consumed": summ["levels_consumed"],
                    "final_mod_level": summ["final_mod_level"],
                    "prediction_match": pred_match,
                }
            )
            print(
                f"  {poly['id']:<28} MAE={np.mean(err):.2e} niveles={summ['levels_consumed']} "
                f"pred_match={pred_match}"
            )
        except Exception as e:
            results.append(
                {
                    "configuration_id": f"{poly['method']}_d{degree}_{poly['interval_name']}",
                    "activation": "act3",
                    "method": poly["method"],
                    "degree": degree,
                    "interval_name": poly["interval_name"],
                    "ckks_feasible": False,
                    "failure_reason": f"{type(e).__name__}: {e}",
                }
            )
            print(f"  {poly['id']:<28} FALLO: {e}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in results for k in r})
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print("-" * 72)
    feasible_count = sum(1 for r in results if r.get("ckks_feasible"))
    print(f"Configuraciones factibles: {feasible_count}/{len(results)}")
    print(f"CSV: {OUT_CSV}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
