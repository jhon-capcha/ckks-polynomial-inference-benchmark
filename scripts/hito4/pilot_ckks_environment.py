"""
Piloto 4-P0b: evaluación polinómica CKKS con alineación correcta (Pyfhel 3.5.0).

Corrige el piloto anterior usando align_mod_n_scale para resolver el
"parameter mismatch" en Horner, y ct.mod_level / ct.scale_bits para leer
nivel y escala reales. Confirma la profundidad por cadena y evalúa
polinomios sintéticos grado 3/5/7.

Escala base: 2^40 (coincide con primos intermedios de 40 bits).
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np

OUT_JSON = Path("results/published/hito4_environment_probe.json")

# Cadenas válidas (scale 2^40, primos de 40 bits). La de 8192 se descarta
# (profundidad 1 insuficiente). Foco en N=16384.
CANDIDATES = [
    ("n16384_c6", 16384, [60, 40, 40, 40, 40, 60], 40),  # profundidad 4
    ("n16384_c8", 16384, [60, 40, 40, 40, 40, 40, 40, 60], 40),  # profundidad 6
    ("n16384_c10", 16384, [60] + [40] * 8 + [60], 40),  # profundidad 8 (probar)
    ("n16384_c12", 16384, [60] + [40] * 10 + [60], 40),  # profundidad 10 (probar seguridad)
]


def _versions() -> dict:
    import Pyfhel

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pyfhel": getattr(Pyfhel, "__version__", "unknown"),
    }


def make_context(N, coeff_bits, scale_bits):
    """Crea contexto y verifica que sea válido y seguro."""
    from Pyfhel import Pyfhel

    he = Pyfhel()
    try:
        he.contextGen(scheme="ckks", n=N, scale=2**scale_bits, qi_sizes=coeff_bits)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    if he.is_context_empty():
        return None, "context_empty"
    return he, None


def context_info(he) -> dict:
    info = {}
    try:
        info["security_bits"] = int(he.sec)
    except Exception:
        info["security_bits"] = "unknown"
    try:
        info["slot_count"] = he.get_nSlots()
    except Exception:
        info["slot_count"] = None
    try:
        info["total_coeff_modulus_bits"] = he.total_coeff_modulus_bit_count()
    except Exception:
        info["total_coeff_modulus_bits"] = None
    return info


def measure_depth(he) -> dict:
    """Cadena de cuadrados: profundidad multiplicativa real."""
    he.keyGen()
    he.relinKeyGen()
    x = np.full(he.get_nSlots(), 1.05, dtype=np.float64)
    ct = he.encryptFrac(x)
    expected = x.copy()
    max_depth = 0
    failure = None

    for i in range(15):
        try:
            he.square(ct)
            he.relinearize(ct)
            he.rescale_to_next(ct)
            expected = expected * expected
            dec = he.decryptFrac(ct)[: len(x)]
            if np.all(np.isfinite(dec)) and np.mean(np.abs(dec - expected)) < 1.0:
                max_depth = i + 1
                mod_level = ct.mod_level
            else:
                failure = f"error en profundidad {i + 1}"
                break
        except Exception as e:
            failure = f"{type(e).__name__}: {e} en profundidad {i + 1}"
            break
    return {"max_multiplicative_depth": max_depth, "depth_failure": failure}


def eval_horner(he, coeffs, x_scalar) -> dict:
    """Horner sobre CKKS con align_mod_n_scale para alinear operandos."""
    he.keyGen()
    he.relinKeyGen()
    slots = he.get_nSlots()
    x = np.full(slots, x_scalar, dtype=np.float64)
    expected = float(np.polyval(coeffs[::-1], x_scalar))
    degree = len(coeffs) - 1

    try:
        # acc = a_d (ciphertext)
        acc = he.encryptFrac(np.full(slots, float(coeffs[-1]), dtype=np.float64))
        for k in range(degree - 1, -1, -1):
            # acc = acc * x
            x_ct = he.encryptFrac(x)
            # Alinear x_ct al nivel/escala de acc.
            acc_a, x_a = he.align_mod_n_scale(acc, x_ct)
            he.multiply(acc_a, x_a)
            he.relinearize(acc_a)
            he.rescale_to_next(acc_a)
            acc = acc_a
            # acc = acc + a_k (plaintext, alineado)
            ptxt = he.encodeFrac(np.full(slots, float(coeffs[k]), dtype=np.float64))
            acc_b, ptxt_a = he.align_mod_n_scale(acc, ptxt)
            he.add_plain(acc_b, ptxt_a)
            acc = acc_b

        dec = float(he.decryptFrac(acc)[0])
        err = abs(dec - expected)
        return {
            "supported": bool(err < 0.5),
            "error": float(err),
            "expected": expected,
            "obtained": dec,
            "final_mod_level": acc.mod_level,
            "final_scale_bits": float(acc.scale_bits),
        }
    except Exception as e:
        return {"supported": False, "error": float("nan"), "failure": f"{type(e).__name__}: {e}"}


def main() -> int:
    versions = _versions()
    print("=" * 72)
    print("PILOTO 4-P0b — Evaluación polinómica CKKS (alineación corregida)")
    print("=" * 72)
    for k, v in versions.items():
        print(f"  {k}: {v}")
    print("-" * 72)

    results = []
    for cand_id, N, coeff_bits, scale_bits in CANDIDATES:
        print(
            f"\n### {cand_id}: chain={coeff_bits} ({sum(coeff_bits)} bits), scale=2^{scale_bits} ###"
        )
        he, err = make_context(N, coeff_bits, scale_bits)
        entry = {
            "candidate_id": cand_id,
            "coeff_bits": coeff_bits,
            "total_bits": sum(coeff_bits),
            "scale_bits": scale_bits,
        }

        if he is None:
            print(f"  RECHAZADO/INVÁLIDO: {err}")
            entry["accepted"] = False
            entry["reason"] = err
            results.append(entry)
            continue

        info = context_info(he)
        entry.update({"accepted": True, **info})
        print(
            f"  OK: sec={info['security_bits']}, slots={info['slot_count']}, "
            f"total_mod={info['total_coeff_modulus_bits']}bits"
        )

        # Profundidad (contexto fresco).
        he_d, _ = make_context(N, coeff_bits, scale_bits)
        depth = measure_depth(he_d)
        entry.update(depth)
        print(
            f"  profundidad máx: {depth['max_multiplicative_depth']} "
            f"({depth['depth_failure'] or 'sin fallo'})"
        )

        # Polinomios grado 3, 5, 7.
        test_coeffs = {
            3: np.array([0.5, 1.0, -0.25, 0.1]),
            5: np.array([0.5, 1.0, -0.25, 0.1, 0.05, -0.02]),
            7: np.array([0.5, 1.0, -0.25, 0.1, 0.05, -0.02, 0.01, 0.005]),
        }
        for deg, coeffs in test_coeffs.items():
            he_p, _ = make_context(N, coeff_bits, scale_bits)
            res = eval_horner(he_p, coeffs, 0.7)
            entry[f"degree{deg}"] = res
            if res.get("error") == res.get("error"):  # no NaN
                lvl = res.get("final_mod_level", "?")
                print(
                    f"  grado {deg}: {'OK' if res['supported'] else 'NO'} "
                    f"(err={res['error']:.2e}, mod_level_final={lvl})"
                )
            else:
                print(f"  grado {deg}: fallo — {res.get('failure')}")

        results.append(entry)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {"versions": versions, "candidates": results}, indent=2, ensure_ascii=False, default=str
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("RESUMEN")
    print("=" * 72)
    for r in results:
        if r.get("accepted"):
            g3 = r.get("degree3", {}).get("supported")
            g5 = r.get("degree5", {}).get("supported")
            g7 = r.get("degree7", {}).get("supported")
            print(
                f"  {r['candidate_id']:<14} sec={r.get('security_bits')} "
                f"prof={r.get('max_multiplicative_depth')} "
                f"g3={g3} g5={g5} g7={g7}"
            )
        else:
            print(f"  {r['candidate_id']:<14} RECHAZADO ({r.get('reason')})")
    print(f"\nJSON: {OUT_JSON}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
