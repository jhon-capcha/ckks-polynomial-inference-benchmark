"""
Gráficas de cierre del Hito 4 (validación CKKS del bloque final).

Genera:
    hito4_delta_decomposition.png    - Δ_aproximación, Δ_CKKS, Δ_total por config
    hito4_logit_error_by_degree.png  - error de logits CKKS por grado
    hito4_prediction_agreement.png   - concordancia poly vs CKKS por config
    hito4_levels_consumed.png        - niveles consumidos por config/grado
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SUMMARY_CSV = Path("results/tables/hito4_ckks_validation_summary.csv")
IMAGE_CSV = Path("results/tables/hito4_ckks_image_results.csv")
OUT_DIR = Path("results/figures/hito4")

METHOD_COLORS = {"chebyshev": "tab:blue", "least_squares": "tab:green"}


def _load_summary():
    with SUMMARY_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_delta_decomposition(rows, out):
    labels = [r["configuration_id"] for r in rows]
    d_approx = [float(r["delta_approximation_accuracy"]) for r in rows]
    d_ckks = [float(r["delta_ckks_accuracy"]) for r in rows]
    d_total = [float(r["delta_total_accuracy"]) for r in rows]

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w, d_approx, w, label="Δ aproximación", color="tab:orange")
    ax.bar(x, d_ckks, w, label="Δ CKKS", color="tab:red")
    ax.bar(x + w, d_total, w, label="Δ total", color="tab:purple")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Δ accuracy (vs referencia)")
    ax.set_title(
        "Descomposición del error: Δ_total = Δ_aproximación + Δ_CKKS\n"
        "(Δ_CKKS ≈ 0: el cifrado no degrada la clasificación)"
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_logit_error_by_degree(rows, out):
    fig, ax = plt.subplots(figsize=(9, 6))
    for method in ("chebyshev", "least_squares"):
        pts = [
            (int(r["degree"]), float(r["mean_logit_mae_ckks"]))
            for r in rows
            if r["method"] == method
        ]
        pts.sort()
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "o-", color=METHOD_COLORS[method], label=method, markersize=8)
    ax.set_xlabel("grado")
    ax.set_ylabel("MAE de logits CKKS (media)")
    ax.set_yscale("log")
    ax.set_xticks([3, 5])
    ax.set_title("Error CKKS de logits por grado\n(crece con el grado; se mantiene ~1e-4)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_prediction_agreement(rows, out):
    labels = [r["configuration_id"] for r in rows]
    agree = [float(r["prediction_agreement_poly_ckks"]) for r in rows]
    colors = [METHOD_COLORS[r["method"]] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(range(len(labels)), agree, color=colors)
    for b, a in zip(bars, agree):
        ax.text(
            b.get_x() + b.get_width() / 2,
            a - 0.05,
            f"{a:.3f}",
            ha="center",
            va="top",
            fontsize=8,
            color="white",
            fontweight="bold",
        )
    ax.axhline(0.99, color="red", linestyle="--", linewidth=1, label="umbral 0.99")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("concordancia predicción poly ↔ CKKS")
    ax.set_ylim(0.9, 1.02)
    ax.set_title(
        "Concordancia de predicción: polinómica clara vs bloque CKKS\n"
        "(1.000 = el cifrado no cambia ninguna predicción)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_accuracy_comparison(rows, out):
    """Tres accuracies (ReLU, poly clara, CKKS) por configuración."""
    labels = [r["configuration_id"] for r in rows]
    relu = [float(r["relu_accuracy"]) for r in rows]
    poly = [float(r["polynomial_clear_accuracy"]) for r in rows]
    ckks = [float(r["ckks_accuracy"]) for r in rows]

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w, relu, w, label="ReLU", color="black", alpha=0.7)
    ax.bar(x, poly, w, label="polinómica clara", color="tab:orange")
    ax.bar(x + w, ckks, w, label="bloque CKKS", color="tab:red")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("accuracy (100 imágenes)")
    ax.set_ylim(0.90, 1.0)
    ax.set_title(
        "Accuracy por ruta: poly clara y CKKS coinciden en todas\n"
        "(las barras naranja y roja son idénticas: Δ_CKKS=0)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_summary()
    rows.sort(key=lambda r: (int(r["degree"]), r["method"], r["interval"]))

    plot_delta_decomposition(rows, OUT_DIR / "hito4_delta_decomposition.png")
    plot_logit_error_by_degree(rows, OUT_DIR / "hito4_logit_error_by_degree.png")
    plot_prediction_agreement(rows, OUT_DIR / "hito4_prediction_agreement.png")
    plot_accuracy_comparison(rows, OUT_DIR / "hito4_accuracy_comparison.png")

    print("Gráficas del Hito 4:")
    for name in (
        "hito4_delta_decomposition",
        "hito4_logit_error_by_degree",
        "hito4_prediction_agreement",
        "hito4_accuracy_comparison",
    ):
        print(f"  {name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
