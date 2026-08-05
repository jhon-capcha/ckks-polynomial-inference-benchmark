"""
Gráficas de cierre del Hito 5 (benchmarking).

Genera:
    hito5_latency_by_stage.png       - barras apiladas encrypt/act3/fc2/decrypt
    hito5_pareto_frontier.png        - accuracy vs latencia, frontera marcada
    hito5_grade_increment.png        - incremento grado 3->5 por etapa
    hito5_storage_footprint.png      - huella de claves y ciphertexts
    hito5_latency_distribution.png   - boxplot de latencia por configuración
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BY_CONFIG = Path("results/tables/hito5_latency_by_config.csv")
RAW = Path("results/tables/hito5_latency_raw.csv")
TRADEOFF = Path("results/tables/hito5_precision_depth_latency.csv")
RESOURCE = Path("results/tables/hito5_resource_consumption.csv")
OUT_DIR = Path("results/figures/hito5")

STAGE_COLORS = {"encrypt": "tab:gray", "act3": "tab:blue", "fc2": "tab:red", "decrypt": "tab:green"}


def _load(path):
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_latency_by_stage(out):
    rows = sorted(_load(BY_CONFIG), key=lambda r: (int(r["degree"]), r["configuration_id"]))
    labels = [r["configuration_id"] for r in rows]
    stages = ["encrypt", "act3", "fc2", "decrypt"]

    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(labels))
    for stage in stages:
        vals = np.array([float(r[f"{stage}_median_ms"]) for r in rows])
        ax.bar(range(len(labels)), vals, bottom=bottom, label=stage, color=STAGE_COLORS[stage])
        bottom += vals
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("latencia (ms, mediana)")
    ax.set_title("Latencia por etapa del bloque cifrado\n(fc2 domina; act3 crece con el grado)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_pareto(out):
    rows = _load(TRADEOFF)
    fig, ax = plt.subplots(figsize=(9, 7))
    for r in rows:
        acc = float(r["ckks_accuracy"])
        lat = float(r["median_online_ms"])
        is_pareto = r["pareto_accuracy_latency"] == "1"
        color = "tab:green" if int(r["degree"]) == 5 else "tab:blue"
        marker = "*" if is_pareto else "o"
        size = 300 if is_pareto else 100
        ax.scatter(
            lat, acc, c=color, marker=marker, s=size, edgecolors="black", linewidths=1, zorder=3
        )
        ax.annotate(
            r["configuration_id"].replace("_", "\n"),
            (lat, acc),
            fontsize=6,
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
        )
    # Línea de la frontera (configs Pareto ordenadas por latencia).
    pareto = sorted(
        [r for r in rows if r["pareto_accuracy_latency"] == "1"],
        key=lambda r: float(r["median_online_ms"]),
    )
    ax.plot(
        [float(r["median_online_ms"]) for r in pareto],
        [float(r["ckks_accuracy"]) for r in pareto],
        "k--",
        alpha=0.4,
        zorder=1,
        label="frontera de Pareto",
    )
    ax.scatter([], [], c="tab:blue", marker="o", label="grado 3")
    ax.scatter([], [], c="tab:green", marker="o", label="grado 5")
    ax.scatter([], [], c="gray", marker="*", s=200, label="Pareto-óptima")
    ax.set_xlabel("latencia online (ms, mediana)")
    ax.set_ylabel("accuracy CKKS")
    ax.set_title(
        "Frontera de Pareto: accuracy vs latencia\n"
        "(★ = no dominada; el trade-off precisión–latencia)"
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_grade_increment(out):
    rows = _load(BY_CONFIG)
    d3 = next(r for r in rows if int(r["degree"]) == 3 and "chebyshev" in r["configuration_id"])
    d5 = next(
        r
        for r in rows
        if int(r["degree"]) == 5
        and "chebyshev" in r["configuration_id"]
        and "I1" in r["configuration_id"]
    )
    stages = ["encrypt", "act3", "fc2", "decrypt"]
    d3_vals = [float(d3[f"{s}_median_ms"]) for s in stages]
    d5_vals = [float(d5[f"{s}_median_ms"]) for s in stages]

    x = np.arange(len(stages))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - w / 2, d3_vals, w, label="grado 3", color="tab:blue")
    ax.bar(x + w / 2, d5_vals, w, label="grado 5", color="tab:green")
    for i, (v3, v5) in enumerate(zip(d3_vals, d5_vals)):
        delta = v5 - v3
        ax.text(i, max(v3, v5) + 2, f"Δ{delta:+.0f}ms", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("latencia (ms, mediana)")
    ax.set_title(
        "Incremento de latencia grado 3 → 5 por etapa\n"
        "(el aumento viene casi todo de act3; fc2 casi constante)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_storage(out):
    rows = _load(RESOURCE)
    # Un representante por perfil.
    d3 = next(r for r in rows if int(r["degree"]) == 3)
    d5 = next(r for r in rows if int(r["degree"]) == 5)
    components = [
        "rotation_key_bytes",
        "relin_key_bytes",
        "public_key_bytes",
        "input_ciphertext_bytes",
        "output_ciphertexts_bytes",
    ]
    comp_labels = [
        "claves\nrotación",
        "clave\nrelin",
        "clave\npública",
        "ct\nentrada",
        "ct\nsalida",
    ]
    d3_mb = [float(d3[c]) / 1e6 for c in components]
    d5_mb = [float(d5[c]) / 1e6 for c in components]

    x = np.arange(len(components))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w / 2, d3_mb, w, label="grado 3 (perfil d3)", color="tab:blue")
    ax.bar(x + w / 2, d5_mb, w, label="grado 5 (perfil d5)", color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels, fontsize=8)
    ax.set_ylabel("huella de almacenamiento (MB)")
    ax.set_yscale("log")
    ax.set_title(
        "Huella de almacenamiento: claves y ciphertexts\n"
        "(las claves de rotación dominan; escala log)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_latency_distribution(out):
    raw = _load(RAW)
    by_config = defaultdict(list)
    for r in raw:
        by_config[r["configuration_id"]].append(float(r["online_total_seconds"]) * 1000)
    configs = sorted(by_config.keys(), key=lambda c: (int(c.split("_d")[1][0]), c))
    data = [by_config[c] for c in configs]

    fig, ax = plt.subplots(figsize=(11, 6))
    bp = ax.boxplot(data, tick_labels=configs, showfliers=True, patch_artist=True)
    for patch, cid in zip(bp["boxes"], configs):
        patch.set_facecolor("tab:green" if "_d5" in cid else "tab:blue")
        patch.set_alpha(0.6)
    ax.set_xticklabels(configs, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("latencia online (ms)")
    ax.set_title(
        "Distribución de latencia por configuración (30 rep × 10 img)\n"
        "(baja varianza; grado 5 en verde, grado 3 en azul)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_latency_by_stage(OUT_DIR / "hito5_latency_by_stage.png")
    plot_pareto(OUT_DIR / "hito5_pareto_frontier.png")
    plot_grade_increment(OUT_DIR / "hito5_grade_increment.png")
    plot_storage(OUT_DIR / "hito5_storage_footprint.png")
    plot_latency_distribution(OUT_DIR / "hito5_latency_distribution.png")
    print("Gráficas del Hito 5:")
    for name in (
        "hito5_latency_by_stage",
        "hito5_pareto_frontier",
        "hito5_grade_increment",
        "hito5_storage_footprint",
        "hito5_latency_distribution",
    ):
        print(f"  {name}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
