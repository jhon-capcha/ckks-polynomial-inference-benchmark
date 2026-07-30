"""
Gráficas del análisis y selección (fase 3C-C).

Lee results/tables/hito3c_combined_analysis.csv y la selección congelada, y
genera en results/figures/hito3c/:
    - accuracy_by_configuration.png     : accuracy de las 24, base ReLU, seleccionadas
    - accuracy_by_degree.png            : accuracy vs grado, por método e intervalo
    - functional_error_vs_accuracy.png  : MAE funcional vs Δaccuracy (dispersión)
    - activation_magnitude_vs_accuracy.png : log(act3_max_abs) vs accuracy (cascada)
    - configuration_status_heatmap.png  : matriz método×grado × intervalo, estado
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COMBINED_CSV = Path("results/tables/hito3c_combined_analysis.csv")
FROZEN = Path("results/published/hito3c_frozen_selection.json")
OUT_DIR = Path("results/figures/hito3c")

RELU_BASELINE = 0.9887
METHODS = ("chebyshev", "least_squares", "taylor")
DEGREES = (3, 5, 7, 9)
METHOD_COLORS = {"chebyshev": "tab:blue", "least_squares": "tab:green", "taylor": "tab:red"}


def _load() -> tuple[list[dict], set[str]]:
    with COMBINED_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    selection = json.loads(FROZEN.read_text(encoding="utf-8"))
    selected_ids = {s["configuration_id"] for s in selection["selected_for_ckks"]}
    return rows, selected_ids


def _fv(row: dict, key: str) -> float:
    v = row.get(key, "")
    return float(v) if v not in ("", None) else float("nan")


def plot_accuracy_by_configuration(rows, selected_ids, out: Path) -> None:
    valid = [r for r in rows if r["valid"].lower() == "true"]
    valid_sorted = sorted(valid, key=lambda r: _fv(r, "validation_accuracy"), reverse=True)
    labels = [r["configuration_id"] for r in valid_sorted]
    accs = [_fv(r, "validation_accuracy") for r in valid_sorted]
    colors = [
        "tab:orange" if r["configuration_id"] in selected_ids else "lightgray" for r in valid_sorted
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(labels)), accs, color=colors)
    ax.axhline(
        RELU_BASELINE,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"ReLU baseline ({RELU_BASELINE})",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("validation accuracy")
    ax.set_title("Accuracy por configuración (naranja = seleccionada para CKKS)")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_accuracy_by_degree(rows, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for method in ("chebyshev", "least_squares"):
        for interval, style in (("I1", "o-"), ("I2", "s--")):
            xs, ys = [], []
            for degree in DEGREES:
                match = [
                    r
                    for r in rows
                    if r["method"] == method
                    and int(r["degree"]) == degree
                    and r["interval_name"] == interval
                    and r["valid"].lower() == "true"
                ]
                if match:
                    xs.append(degree)
                    ys.append(_fv(match[0], "validation_accuracy"))
            if xs:
                ax.plot(
                    xs,
                    ys,
                    style,
                    color=METHOD_COLORS[method],
                    label=f"{method} {interval}",
                    alpha=0.8,
                )
    ax.axhline(RELU_BASELINE, color="black", linestyle=":", linewidth=1, label="ReLU")
    ax.set_xlabel("grado")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Accuracy vs grado (por método e intervalo)")
    ax.set_xticks(DEGREES)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_functional_vs_accuracy(rows, selected_ids, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in METHODS:
        pts = [
            (_fv(r, "functional_mae_mean"), _fv(r, "delta_accuracy_vs_relu"))
            for r in rows
            if r["method"] == method and r["valid"].lower() == "true"
        ]
        pts = [(x, y) for x, y in pts if np.isfinite(x) and np.isfinite(y)]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, color=METHOD_COLORS[method], label=method, alpha=0.7, s=45)
    ax.set_xscale("log")
    ax.set_xlabel("MAE funcional medio (log)")
    ax.set_ylabel("Δ accuracy vs ReLU")
    ax.set_title("Error funcional vs pérdida de accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_magnitude_vs_accuracy(rows, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for method in METHODS:
        pts = [
            (_fv(r, "act3_max_abs"), _fv(r, "validation_accuracy"))
            for r in rows
            if r["method"] == method and r["valid"].lower() == "true"
        ]
        pts = [(x, y) for x, y in pts if np.isfinite(x) and np.isfinite(y) and x > 0]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(np.log10(xs), ys, color=METHOD_COLORS[method], label=method, alpha=0.7, s=45)
    ax.set_xlabel("log10(act3 max_abs)")
    ax.set_ylabel("validation accuracy")
    ax.set_title("Magnitud interna (act3) vs accuracy — visualiza el colapso por cascada")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_status_heatmap(rows, out: Path) -> None:
    row_labels = [f"{m}_d{d}" for m in METHODS for d in DEGREES]
    col_labels = ["I1", "I2"]
    # Estado numérico: 2=elegible, 1=válido no elegible, 0=inválido.
    matrix = np.zeros((len(row_labels), len(col_labels)))
    lookup = {r["configuration_id"]: r for r in rows}
    for ri, (m, d) in enumerate([(m, d) for m in METHODS for d in DEGREES]):
        for ci, iv in enumerate(col_labels):
            cid = f"{m}_d{d}_{iv}"
            r = lookup.get(cid)
            if r is None:
                matrix[ri, ci] = -1
            elif r["valid"].lower() != "true":
                matrix[ri, ci] = 0
            elif r["eligible_for_ckks"].lower() == "true":
                matrix[ri, ci] = 2
            else:
                matrix[ri, ci] = 1

    fig, ax = plt.subplots(figsize=(5, 9))
    cmap = plt.get_cmap("RdYlGn", 3)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=2)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Estado por configuración\n(verde=elegible, amarillo=válido, rojo=inválido)")
    for ri in range(len(row_labels)):
        for ci in range(len(col_labels)):
            r = lookup.get(f"{row_labels[ri]}_{col_labels[ci]}")
            if r and r["valid"].lower() == "true":
                acc = _fv(r, "validation_accuracy")
                ax.text(ci, ri, f"{acc:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, ticks=[0, 1, 2], label="estado")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, selected_ids = _load()

    print("Generando gráficas del Hito 3C-C...")
    plot_accuracy_by_configuration(rows, selected_ids, OUT_DIR / "accuracy_by_configuration.png")
    plot_accuracy_by_degree(rows, OUT_DIR / "accuracy_by_degree.png")
    plot_functional_vs_accuracy(rows, selected_ids, OUT_DIR / "functional_error_vs_accuracy.png")
    plot_magnitude_vs_accuracy(rows, OUT_DIR / "activation_magnitude_vs_accuracy.png")
    plot_status_heatmap(rows, OUT_DIR / "configuration_status_heatmap.png")

    generated = sorted(OUT_DIR.glob("*.png"))
    hito3c_new = [
        p
        for p in generated
        if p.name
        in (
            "accuracy_by_configuration.png",
            "accuracy_by_degree.png",
            "functional_error_vs_accuracy.png",
            "activation_magnitude_vs_accuracy.png",
            "configuration_status_heatmap.png",
        )
    ]
    for p in hito3c_new:
        print(f"  {p.name}")
    print(f"Total nuevas: {len(hito3c_new)} figuras")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
