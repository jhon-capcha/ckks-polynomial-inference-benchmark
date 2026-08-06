"""
Figuras integradoras del Hito 6 (analisis final).

Genera:
    hito6_pareto_material.png    - frontera material vs estricta, con las dos
                                   configuraciones oficiales destacadas
    hito6_tradeoff_regimes.png   - sintesis del trade-off entre los dos regimenes
                                   (accuracy, latencia, niveles, huella)
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MASTER = Path("results/tables/hito6_master_results.csv")
OUT_DIR = Path("results/figures/hito6")


def _load():
    with MASTER.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_pareto_material(rows, out):
    fig, ax = plt.subplots(figsize=(10, 7))
    for r in rows:
        acc = float(r["accuracy_polynomial_test"])
        lat = float(r["median_latency_ms"])
        is_strict = r["pareto_strict_test"] == "1"
        is_material = r["pareto_material_test"] == "1"
        degree = int(r["degree"])
        color = "tab:green" if degree == 5 else "tab:blue"

        if is_material:
            ax.scatter(
                lat, acc, c=color, marker="*", s=500, edgecolors="black", linewidths=1.5, zorder=4
            )
        elif is_strict:
            ax.scatter(
                lat,
                acc,
                c=color,
                marker="o",
                s=160,
                edgecolors="black",
                linewidths=1,
                zorder=3,
                alpha=0.7,
            )
        else:
            ax.scatter(
                lat,
                acc,
                c=color,
                marker="o",
                s=90,
                edgecolors="gray",
                linewidths=0.8,
                zorder=2,
                alpha=0.5,
            )
        # Etiqueta.
        label = r["configuration_id"].replace("_", "\n")
        ax.annotate(
            label, (lat, acc), fontsize=6.5, xytext=(8, -2), textcoords="offset points", va="center"
        )

    # Linea de frontera material.
    mat = sorted(
        [r for r in rows if r["pareto_material_test"] == "1"],
        key=lambda r: float(r["median_latency_ms"]),
    )
    ax.plot(
        [float(r["median_latency_ms"]) for r in mat],
        [float(r["accuracy_polynomial_test"]) for r in mat],
        "k--",
        alpha=0.5,
        zorder=1,
        label="frontera material (ε=0.5ms)",
    )

    # Leyenda.
    ax.scatter([], [], c="tab:blue", marker="o", label="grado 3")
    ax.scatter([], [], c="tab:green", marker="o", label="grado 5")
    ax.scatter([], [], c="gray", marker="*", s=250, label="frontera material")
    ax.scatter([], [], c="gray", marker="o", s=120, alpha=0.7, label="solo frontera estricta")

    ax.set_xlabel("latencia online (ms, mediana)")
    ax.set_ylabel("accuracy sobre test completo")
    ax.set_title(
        "Frontera de Pareto: material vs estricta\n"
        "(★ = frontera material; ○ grande = entra solo en la estricta "
        "por <0.5ms)"
    )
    ax.legend(loc="center right", fontsize=8)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_tradeoff_regimes(rows, out):
    # Las dos configuraciones de la frontera material.
    d3 = next(r for r in rows if r["configuration_id"] == "least_squares_d3_I1")
    d5 = next(r for r in rows if r["configuration_id"] == "chebyshev_d5_I1")

    metrics = ["accuracy\n(test)", "latencia\n(ms)", "niveles", "claves rot.\n(MB)"]
    d3_vals = [
        float(d3["accuracy_polynomial_test"]),
        float(d3["median_latency_ms"]),
        int(d3["levels_block"]),
        float(d3["rotation_keys_mb"]),
    ]
    d5_vals = [
        float(d5["accuracy_polynomial_test"]),
        float(d5["median_latency_ms"]),
        int(d5["levels_block"]),
        float(d5["rotation_keys_mb"]),
    ]

    # Normalizar cada metrica a [0,1] respecto al maximo para comparar en un radar/barras.
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    colors = {"d3": "tab:blue", "d5": "tab:green"}
    labels = ["least_squares_d3_I1\n(menor costo)", "chebyshev_d5_I1\n(max precision)"]

    for ax, metric, v3, v5 in zip(axes, metrics, d3_vals, d5_vals):
        bars = ax.bar([0, 1], [v3, v5], color=[colors["d3"], colors["d5"]], alpha=0.75)
        ax.set_title(metric, fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["d3", "d5"], fontsize=10)
        for b, v in zip(bars, [v3, v5]):
            fmt = f"{v:.4f}" if v < 2 else (f"{v:.0f}" if v > 10 else f"{v:.1f}")
            ax.text(b.get_x() + b.get_width() / 2, v, fmt, ha="center", va="bottom", fontsize=9)
        ax.margins(y=0.15)

    fig.suptitle(
        "Trade-off entre los dos regimenes de la frontera material\n"
        "least_squares_d3_I1 (menor costo) vs chebyshev_d5_I1 (maxima precision viable): "
        "+2.15pp accuracy por +29.5% latencia, +2 niveles, +81.9% claves",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load()
    plot_pareto_material(rows, OUT_DIR / "hito6_pareto_material.png")
    plot_tradeoff_regimes(rows, OUT_DIR / "hito6_tradeoff_regimes.png")
    print("Figuras del Hito 6:")
    for name in ("hito6_pareto_material", "hito6_tradeoff_regimes"):
        print(f"  {name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
