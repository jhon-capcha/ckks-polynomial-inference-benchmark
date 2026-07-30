"""
Gráficas y tabla de cierre del Hito 3C-D (evaluación sobre test).

Genera la comparación validación↔test (hallazgo central de consistencia) más
las gráficas estándar de test. Produce:
    results/tables/hito3c_validation_test_comparison.csv
    results/figures/hito3c/test_accuracy_by_configuration.png
    results/figures/hito3c/test_delta_accuracy.png
    results/figures/hito3c/validation_vs_test_accuracy.png
    results/figures/hito3c/test_accuracy_by_degree.png
    results/figures/hito3c/functional_error_vs_test_delta_accuracy.png
    results/figures/hito3c/frozen_shortlist_test_summary.png
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEST_JSON = Path("results/published/hito3c_test_results.json")
COMBINED_CSV = Path("results/tables/hito3c_combined_analysis.csv")
FROZEN = Path("results/published/hito3c_frozen_selection.json")
COMPARISON_CSV = Path("results/tables/hito3c_validation_test_comparison.csv")
OUT_DIR = Path("results/figures/hito3c")

METHOD_COLORS = {"chebyshev": "tab:blue", "least_squares": "tab:green", "taylor": "tab:red"}


def _load():
    test_data = json.loads(TEST_JSON.read_text(encoding="utf-8"))
    test_results = {r["configuration_id"]: r for r in test_data["results"]}
    relu_test = test_data["metadata"]["relu_baseline_test"]["accuracy"]

    with COMBINED_CSV.open(encoding="utf-8") as f:
        combined = {r["configuration_id"]: r for r in csv.DictReader(f)}

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    selected = {s["configuration_id"]: s for s in frozen["selected_for_ckks"]}
    diagnostic = {d["configuration_id"] for d in frozen["diagnostic_baselines"]}

    return test_results, relu_test, combined, selected, diagnostic


def build_comparison_table(test_results, combined, selected) -> list[dict]:
    """Tabla validación↔test con generalization_gap, solo para la shortlist."""
    rows = []
    for cid, sel in selected.items():
        tr = test_results[cid]
        val_acc = float(combined[cid]["validation_accuracy"])
        test_acc = tr[
            "validation_accuracy"
        ]  # el campo del dataclass guarda accuracy del split evaluado
        gap = test_acc - val_acc
        rows.append(
            {
                "configuration_id": cid,
                "method": tr["method"],
                "degree": tr["degree"],
                "interval_name": tr["interval_name"],
                "validation_accuracy": val_acc,
                "test_accuracy": test_acc,
                "generalization_gap": gap,
                "absolute_generalization_gap": abs(gap),
                "validation_macro_f1": float(combined[cid]["validation_macro_f1"]),
                "test_macro_f1": tr["validation_macro_f1"],
                "delta_accuracy_vs_relu_test": tr["delta_accuracy_vs_relu"],
                "selection_category": sel["selection_category"],
            }
        )
    rows.sort(key=lambda r: r["test_accuracy"], reverse=True)
    return rows


def export_comparison(rows) -> None:
    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    with COMPARISON_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_test_accuracy_by_config(test_results, relu_test, selected, diagnostic, out):
    items = [(cid, r) for cid, r in test_results.items() if r["valid"]]
    items.sort(key=lambda kv: kv[1]["validation_accuracy"], reverse=True)
    labels = [cid for cid, _ in items]
    accs = [r["validation_accuracy"] for _, r in items]
    colors = [
        "tab:orange" if cid in selected else "tab:red" if cid in diagnostic else "lightgray"
        for cid, _ in items
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(range(len(labels)), accs, color=colors)
    ax.axhline(
        relu_test, color="black", linestyle="--", linewidth=1, label=f"ReLU test ({relu_test:.4f})"
    )
    for b, a in zip(bars, accs):
        ax.text(
            b.get_x() + b.get_width() / 2,
            a + 0.005,
            f"{a:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy en test (naranja=shortlist, rojo=diagnóstico)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_test_delta(test_results, selected, out):
    items = [(cid, test_results[cid]) for cid in selected if test_results[cid]["valid"]]
    items.sort(key=lambda kv: kv[1]["delta_accuracy_vs_relu"])
    labels = [cid for cid, _ in items]
    deltas = [r["delta_accuracy_vs_relu"] for _, r in items]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [METHOD_COLORS[test_results[cid]["method"]] for cid in labels]
    bars = ax.barh(range(len(labels)), deltas, color=colors)
    for b, d in zip(bars, deltas):
        ax.text(d + 0.0005, b.get_y() + b.get_height() / 2, f"{d:.4f}", va="center", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Δ accuracy vs ReLU (test) — menor es mejor")
    ax.set_title("Pérdida de accuracy por aproximación (shortlist, test)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_validation_vs_test(comparison_rows, out):
    fig, ax = plt.subplots(figsize=(7, 7))
    for r in comparison_rows:
        ax.scatter(
            r["validation_accuracy"],
            r["test_accuracy"],
            color=METHOD_COLORS[r["method"]],
            s=60,
            zorder=3,
        )
        ax.annotate(
            f"d{r['degree']}_{r['interval_name']}",
            (r["validation_accuracy"], r["test_accuracy"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    lo = (
        min(
            min(r["validation_accuracy"] for r in comparison_rows),
            min(r["test_accuracy"] for r in comparison_rows),
        )
        - 0.01
    )
    hi = 1.0
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.6, label="y = x")
    # Leyenda de métodos.
    for m, c in METHOD_COLORS.items():
        if m != "taylor":
            ax.scatter([], [], color=c, label=m)
    ax.set_xlabel("validation accuracy")
    ax.set_ylabel("test accuracy")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_title("Validación vs test — consistencia de la shortlist congelada")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_test_accuracy_by_degree(comparison_rows, out):
    fig, ax = plt.subplots(figsize=(9, 6))
    for method in ("chebyshev", "least_squares"):
        for interval, style in (("I1", "o-"), ("I2", "s--")):
            pts = sorted(
                [
                    (r["degree"], r["test_accuracy"])
                    for r in comparison_rows
                    if r["method"] == method and r["interval_name"] == interval
                ],
                key=lambda t: t[0],
            )
            if pts:
                xs, ys = zip(*pts)
                ax.plot(
                    xs,
                    ys,
                    style,
                    color=METHOD_COLORS[method],
                    label=f"{method} {interval}",
                    alpha=0.8,
                )
    ax.set_xlabel("grado")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy en test vs grado (shortlist)")
    ax.set_xticks([3, 5, 7])
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_functional_vs_test_delta(test_results, combined, selected, diagnostic, out):
    fig, ax = plt.subplots(figsize=(8, 6))
    for cid, r in test_results.items():
        if not r["valid"]:
            continue
        mae = float(combined[cid]["functional_mae_mean"])
        d_acc = r["delta_accuracy_vs_relu"]
        color = METHOD_COLORS[r["method"]]
        marker = "*" if cid in diagnostic else "o"
        size = 180 if cid in diagnostic else 55
        ax.scatter(mae, d_acc, color=color, marker=marker, s=size, alpha=0.75)
    ax.set_xscale("log")
    ax.set_xlabel("MAE funcional medio (log)")
    ax.set_ylabel("Δ accuracy vs ReLU (test)")
    ax.set_title(
        "Error funcional vs pérdida de accuracy en test\n(★ = Taylor diagnóstico, outlier)"
    )
    for m, c in METHOD_COLORS.items():
        ax.scatter([], [], color=c, label=m)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_shortlist_summary(comparison_rows, out):
    rows = sorted(comparison_rows, key=lambda r: r["test_accuracy"], reverse=True)
    labels = [r["configuration_id"] for r in rows]
    accs = [r["test_accuracy"] for r in rows]
    colors = [METHOD_COLORS[r["method"]] for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(range(len(labels)), accs, color=colors)
    for b, r in zip(bars, rows):
        ax.text(
            b.get_x() + b.get_width() / 2,
            r["test_accuracy"] + 0.003,
            f"{r['test_accuracy']:.3f}\n[{r['selection_category'][:6]}]",
            ha="center",
            va="bottom",
            fontsize=6,
        )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.90, 1.0)
    ax.set_title("Resumen de la shortlist congelada (test)")
    for m, c in METHOD_COLORS.items():
        if m != "taylor":
            ax.bar([], [], color=c, label=m)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test_results, relu_test, combined, selected, diagnostic = _load()

    comparison = build_comparison_table(test_results, combined, selected)
    export_comparison(comparison)
    print(f"Tabla comparación validación↔test: {COMPARISON_CSV}")

    max_gap = max(comparison, key=lambda r: r["absolute_generalization_gap"])
    print(
        f"Máxima brecha val↔test: {max_gap['configuration_id']} "
        f"({max_gap['generalization_gap']:+.4f})"
    )

    print("Generando gráficas del Hito 3C-D...")
    plot_test_accuracy_by_config(
        test_results,
        relu_test,
        selected,
        diagnostic,
        OUT_DIR / "test_accuracy_by_configuration.png",
    )
    plot_test_delta(test_results, selected, OUT_DIR / "test_delta_accuracy.png")
    plot_validation_vs_test(comparison, OUT_DIR / "validation_vs_test_accuracy.png")
    plot_test_accuracy_by_degree(comparison, OUT_DIR / "test_accuracy_by_degree.png")
    plot_functional_vs_test_delta(
        test_results,
        combined,
        selected,
        diagnostic,
        OUT_DIR / "functional_error_vs_test_delta_accuracy.png",
    )
    plot_shortlist_summary(comparison, OUT_DIR / "frozen_shortlist_test_summary.png")

    for name in (
        "test_accuracy_by_configuration",
        "test_delta_accuracy",
        "validation_vs_test_accuracy",
        "test_accuracy_by_degree",
        "functional_error_vs_test_delta_accuracy",
        "frozen_shortlist_test_summary",
    ):
        print(f"  {name}.png")
    print("Total: 6 figuras + 1 tabla")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
