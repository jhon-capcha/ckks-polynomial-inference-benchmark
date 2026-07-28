"""
Piloto Taylor–Softplus (Hito 3A): selección del parámetro beta.

Construye la serie de Taylor de Softplus_beta en x0=0 hasta grado 9 mediante
SymPy (derivación simbólica reproducible), la convierte a coeficientes float64
en base monomial, y mide su error como aproximación de ReLU sobre la muestra
de validación (data/processed/preactivations_validation_sample.npz).

Objetivo: elegir beta con evidencia, no a ciegas. Regla: el menor beta con
error empírico cercano al mínimo, sin explosión de coeficientes en grado 9,
viable para CKKS.

NO forma parte de la API final. Solo decide beta.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import sympy as sp

BETAS = (1, 3)  # solo los viables; beta>=5 ya explotó en act1-I1
DEGREES = (5, 9)
ACTIVATION = "act2"  # escenario extremo
INTERVAL = (-24.8339, 12.2336)  # act2 I2 (el más ancho y asimétrico)
NPZ_PATH = Path("data/processed/preactivations_validation_sample.npz")


def taylor_softplus_coefficients(beta: float, degree: int) -> np.ndarray:
    """Coeficientes monomiales [a0..ad] de la serie de Taylor de Softplus_beta en 0."""
    x = sp.symbols("x", real=True)
    softplus = sp.log(1 + sp.exp(beta * x)) / beta
    series = sp.series(softplus, x, 0, degree + 1).removeO()
    poly = sp.Poly(series, x)
    # Poly.all_coeffs() da [ad..a0]; invertimos a [a0..ad] y rellenamos ceros.
    coeffs_high_to_low = poly.all_coeffs()
    coeffs_low_to_high = list(reversed(coeffs_high_to_low))
    # Asegurar longitud exacta degree+1 (rellenar con ceros si faltan términos altos).
    while len(coeffs_low_to_high) < degree + 1:
        coeffs_low_to_high.append(sp.Integer(0))
    arr = np.array([float(c) for c in coeffs_low_to_high[: degree + 1]], dtype=np.float64)
    return arr


def eval_poly(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evalúa el polinomio (base monomial [a0..ad]) por Horner."""
    result = np.zeros_like(x)
    for c in reversed(coeffs):
        result = result * x + c
    return result


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def uniform_grid(interval: tuple[float, float], n: int = 10_000) -> np.ndarray:
    return np.linspace(interval[0], interval[1], n, dtype=np.float64)


def effective_degree(coeffs: np.ndarray, tol: float = 1e-12) -> int:
    """Mayor índice con coeficiente no despreciable."""
    nz = np.nonzero(np.abs(coeffs) > tol)[0]
    return int(nz[-1]) if nz.size else 0


def validate_derivatives(beta: float, coeffs: np.ndarray, degree: int) -> float:
    """Comprueba que p^(k)(0) coincide con la derivada k-ésima de Softplus en 0.

    Devuelve la máxima discrepancia absoluta encontrada.
    """
    x = sp.symbols("x", real=True)
    softplus = sp.log(1 + sp.exp(beta * x)) / beta
    max_disc = 0.0
    for k in range(degree + 1):
        # Derivada k-ésima de Softplus en 0.
        deriv_sym = sp.diff(softplus, x, k).subs(x, 0)
        deriv_val = float(deriv_sym)
        # Del polinomio: p^(k)(0) = k! * a_k
        poly_deriv = float(math.factorial(k)) * coeffs[k] if k < len(coeffs) else 0.0
        max_disc = max(max_disc, abs(deriv_val - poly_deriv))
    return max_disc


def main() -> int:
    if not NPZ_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {NPZ_PATH}. Ejecuta primero sample_preactivations."
        )

    data = np.load(NPZ_PATH)
    z = data[ACTIVATION].astype(np.float64)  # preactivaciones reales de validación
    relu_z = relu(z)

    grid = uniform_grid(INTERVAL)
    relu_grid = relu(grid)

    rows = []
    coeff_store = {}

    print("=" * 90)
    print(f"PILOTO TAYLOR–SOFTPLUS — {ACTIVATION}, I1={INTERVAL}")
    print("=" * 90)
    print(
        f"{'beta':>5} {'deg':>4} {'MAE_emp':>10} {'RMSE_emp':>10} "
        f"{'MAE_unif':>10} {'Emax':>10} {'C_max':>12} {'R_C':>12} {'deg_ef':>7} {'derivOK':>9}"
    )
    print("-" * 90)

    for beta in BETAS:
        for degree in DEGREES:
            coeffs = taylor_softplus_coefficients(beta, degree)
            coeff_store[f"beta{beta}_deg{degree}"] = coeffs.tolist()

            # Error empírico (sobre preactivaciones reales de validación).
            p_z = eval_poly(coeffs, z)
            mae_emp = float(np.mean(np.abs(relu_z - p_z)))
            rmse_emp = float(np.sqrt(np.mean((relu_z - p_z) ** 2)))

            # Error uniforme (sobre malla).
            p_grid = eval_poly(coeffs, grid)
            mae_unif = float(np.mean(np.abs(relu_grid - p_grid)))
            emax = float(np.max(np.abs(relu_grid - p_grid)))

            # Estabilidad de coeficientes.
            nz = np.abs(coeffs[np.abs(coeffs) > 1e-15])
            c_max = float(nz.max()) if nz.size else 0.0
            r_c = float(nz.max() / nz.min()) if nz.size else float("nan")

            deg_ef = effective_degree(coeffs)
            deriv_disc = validate_derivatives(beta, coeffs, degree)
            deriv_ok = "OK" if deriv_disc < 1e-6 else f"{deriv_disc:.1e}"

            rows.append(
                {
                    "beta": beta,
                    "degree": degree,
                    "mae_empirical": mae_emp,
                    "rmse_empirical": rmse_emp,
                    "mae_uniform": mae_unif,
                    "max_error": emax,
                    "c_max": c_max,
                    "r_c": r_c,
                    "effective_degree": deg_ef,
                    "derivative_discrepancy": deriv_disc,
                }
            )

            print(
                f"{beta:>5} {degree:>4} {mae_emp:>10.5f} {rmse_emp:>10.5f} "
                f"{mae_unif:>10.5f} {emax:>10.5f} {c_max:>12.3e} {r_c:>12.3e} "
                f"{deg_ef:>7} {deriv_ok:>9}"
            )

    print("=" * 90)

    # Guardar resultados.
    out_dir = Path("results/pilots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV.
    import csv

    csv_path = out_dir / "taylor_beta_metrics_act2I2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # JSON de coeficientes.
    json_path = out_dir / "taylor_beta_coefficients_act2I2.json"
    json_path.write_text(
        json.dumps(
            {"activation": ACTIVATION, "interval": list(INTERVAL), "coefficients": coeff_store},
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nGuardado: {csv_path}")
    print(f"Guardado: {json_path}")
    print("\nLectura sugerida: menor beta con MAE_emp cercano al mínimo,")
    print("sin C_max ni R_C explosivos en grado 9. Grado efectivo revela ceros.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
