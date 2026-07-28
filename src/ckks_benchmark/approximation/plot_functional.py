"""
Gráficas mínimas del Hito 3B: validación visual del error funcional.

Lee la tabla maestra (results/tables/hito3b_functional_metrics.csv) y los
polinomios (results/approximations/coefficients/polynomials.json), y genera en
results/figures/hito3b/:
    - relu_vs_poly_<config>.png    : ReLU frente al polinomio (configs clave)
    - error_absoluto_<config>.png  : |ReLU(x) - p(x)| sobre el intervalo
    - comparacion_grados_*.png     : grados 3,5,7,9 por método/activación/intervalo
    - heatmap_mae_empirical.png    : mapa de calor de MAE empírico
    - uniforme_vs_empirico.png     : dispersión MAE uniforme vs empírico

Todas las figuras son diagnósticas del Hito 3B (documentación completa en 3D).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ckks_benchmark.approximation.base import relu_reference  # noqa: E402
from ckks_benchmark.approximation.registry import (  # noqa: E402
    generate_all,
    load_and_validate_intervals,
)

CSV_PATH = Path("results/tables/hito3b_functional_metrics.csv")
INTERVALS_PATH = Path("results/published/preactivation_intervals.json")
OUT_DIR = Path("results/figures/hito3b")

METHODS = ("taylor", "chebyshev", "least_squares")
DEGREES = (3, 5, 7, 9)
ACTIVATIONS = ("act1", "act2", "act3")
METHOD_COLORS = {
    "taylor": "tab:red",
    "chebyshev": "tab:blue",
    "least_squares": "tab:green",
}


def load_metrics() -> list[dict]:
    """Carga la tabla maestra como lista de diccionarios."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"No se encontró {CSV_PATH}. Ejecuta benchmark_functional.")
    with CSV_PATH.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_relu_vs_poly(poly, out_path: Path) -> None:
    """ReLU frente al polinomio sobre su intervalo."""
    a, b = poly.interval
    x = np.linspace(a, b, 2000)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, relu_reference(x), "k-", linewidth=2, label="ReLU")
    ax.plot(
        x,
        poly.evaluate(x),
        "--",
        color=METHOD_COLORS[poly.method],
        linewidth=1.8,
        label=f"{poly.method} (grado {poly.degree})",
    )
    ax.axvline(0.0, color="gray", linewidth=0.6, alpha=0.5)
    ax.set_title(f"ReLU vs polinomio — {poly.identifier()}")
    ax.set_xlabel("x")
    ax.set_ylabel("valor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_error_absolute(poly, out_path: Path) -> None:
    """Error absoluto |ReLU(x) - p(x)| sobre el intervalo."""
    a, b = poly.interval
    x = np.linspace(a, b, 2000)
    err = np.abs(relu_reference(x) - poly.evaluate(x))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, err, color=METHOD_COLORS[poly.method], linewidth=1.5)
    ax.axvline(0.0, color="gray", linewidth=0.6, alpha=0.5)
    ax.set_title(f"Error absoluto — {poly.identifier()}")
    ax.set_xlabel("x")
    ax.set_ylabel("|ReLU(x) - p(x)|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_degree_comparison(
    polys: dict, method: str, activation: str, interval_name: str, out_path: Path
) -> None:
    """Compara grados 3,5,7,9 de un método sobre una activación e intervalo."""
    fig, ax = plt.subplots(figsize=(9, 5))
    # Tomar el intervalo del grado 3 (todos comparten intervalo).
    ref_poly = polys[f"{activation}_{method}_d3_{interval_name}"]
    a, b = ref_poly.interval
    x = np.linspace(a, b, 2000)
    ax.plot(x, relu_reference(x), "k-", linewidth=2, label="ReLU")
    shades = ["#cce", "#88c", "#44a", "#008"]
    for degree, shade in zip(DEGREES, shades):
        poly = polys[f"{activation}_{method}_d{degree}_{interval_name}"]
        ax.plot(x, poly.evaluate(x), "--", color=shade, linewidth=1.3, label=f"grado {degree}")
    ax.axvline(0.0, color="gray", linewidth=0.6, alpha=0.5)
    ax.set_title(f"Comparación por grado — {method}, {activation}, {interval_name}")
    ax.set_xlabel("x")
    ax.set_ylabel("valor")
    ax.legend()
    # Limitar el eje y para que Taylor grado alto no rompa la escala.
    ax.set_ylim(min(a, 0) - 1, b + 2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_heatmap_mae(metrics: list[dict], out_path: Path) -> None:
    """Mapa de calor de MAE empírico: filas=método×grado, columnas=act×intervalo."""
    row_labels = [f"{m}_d{d}" for m in METHODS for d in DEGREES]
    col_labels = [f"{a}_{i}" for a in ACTIVATIONS for i in ("I1", "I2")]

    matrix = np.full((len(row_labels), len(col_labels)), np.nan)
    lookup = {r["configuration_id"]: r for r in metrics}
    for ri, (m, d) in enumerate([(m, d) for m in METHODS for d in DEGREES]):
        for ci, (a, iv) in enumerate([(a, i) for a in ACTIVATIONS for i in ("I1", "I2")]):
            cid = f"{a}_{m}_d{d}_{iv}"
            if cid in lookup:
                matrix[ri, ci] = float(lookup[cid]["mae_empirical"])

    # Escala logarítmica para manejar el rango enorme (Taylor vs Chebyshev).
    log_matrix = np.log10(np.clip(matrix, 1e-4, None))

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(log_matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("MAE empírico (log10) — 72 configuraciones")
    fig.colorbar(im, ax=ax, label="log10(MAE empírico)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_uniform_vs_empirical(metrics: list[dict], out_path: Path) -> None:
    """Dispersión MAE uniforme vs empírico, coloreado por método."""
    fig, ax = plt.subplots(figsize=(8, 8))
    for method in METHODS:
        xs = [float(r["mae_uniform"]) for r in metrics if r["method"] == method]
        ys = [float(r["mae_empirical"]) for r in metrics if r["method"] == method]
        ax.scatter(xs, ys, color=METHOD_COLORS[method], label=method, alpha=0.7, s=40)

    # Línea y=x (donde uniforme = empírico).
    lims = [1e-3, 1e6]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="uniforme = empírico")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("MAE uniforme (log)")
    ax.set_ylabel("MAE empírico (log)")
    ax.set_title("MAE uniforme vs empírico — 72 configuraciones")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics()
    intervals = load_and_validate_intervals(INTERVALS_PATH)
    polys = generate_all(intervals)  # reconstruye los 72 (rápido)

    # Configs representativas para ReLU-vs-poly y error absoluto.
    representative = [
        "act1_chebyshev_d5_I1",
        "act1_least_squares_d5_I1",
        "act1_taylor_d5_I1",
        "act2_taylor_d9_I2",  # caso extremo (no viable)
    ]

    print("Generando gráficas del Hito 3B...")

    for cid in representative:
        poly = polys[cid]
        plot_relu_vs_poly(poly, OUT_DIR / f"relu_vs_poly_{cid}.png")
        plot_error_absolute(poly, OUT_DIR / f"error_absoluto_{cid}.png")

    # Comparación por grado: los tres métodos en act1-I1.
    for method in METHODS:
        plot_degree_comparison(
            polys,
            method,
            "act1",
            "I1",
            OUT_DIR / f"comparacion_grados_{method}_act1_I1.png",
        )

    # Mapa de calor y dispersión.
    plot_heatmap_mae(metrics, OUT_DIR / "heatmap_mae_empirical.png")
    plot_uniform_vs_empirical(metrics, OUT_DIR / "uniforme_vs_empirico.png")

    print(f"Gráficas generadas en {OUT_DIR}/")
    generated = sorted(OUT_DIR.glob("*.png"))
    for p in generated:
        print(f"  {p.name}")
    print(f"Total: {len(generated)} figuras")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
