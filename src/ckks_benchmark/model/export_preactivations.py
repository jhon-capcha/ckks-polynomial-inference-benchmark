"""
Exportación de artefactos del Hito 2: estadísticas, intervalos y resumen.

Genera, a partir de la extracción de preactivaciones, los archivos definitivos:
    results/published/preactivation_statistics.json
    results/published/preactivation_intervals.json
    results/published/preactivation_summary.md

Los intervalos se congelan asimétricos, como política global por activación.
Las figuras (histogramas) se generan en un módulo aparte.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from ckks_benchmark.model.preactivations import (
    ACTIVATIONS,
    EXPECTED_COUNTS,
    MAX_SAMPLES,
    STABILITY_SEEDS,
    ExactStreamingStats,
    PerChannelStats,
    PreactivationCapture,
    collect_preactivations,
    compute_percentiles,
    describe_checkpoint,
    load_trained_model,
)

# Metadata del checkpoint y dataset (constantes del Hito 2).
CHECKPOINT_META = {
    "path": "models/reduced_lenet_relu_best.pt",
    "best_epoch": 8,
    "activation": "ReLU",
    "torch_version": "2.13.0+cpu",
}
DATASET_META = {
    "name": "MNIST",
    "split": "training",
    "images": 54000,
    "augmentation": False,
    "seed_official": STABILITY_SEEDS[0],
    "diagnostic_seed": STABILITY_SEEDS[1],
}
SAMPLING_METHOD = (
    "Submuestreo aleatorio reproducible, estratificado proporcionalmente por batch, "
    "sobre el conjunto completo de preactivaciones."
)


def _round_interval(interval: list[float], ndigits: int = 4) -> list[float]:
    return [round(interval[0], ndigits), round(interval[1], ndigits)]


def build_intervals(pct: dict[str, dict[str, float]]) -> dict[str, dict[str, list[float]]]:
    """Construye los intervalos I1/I2 asimétricos por activación desde percentiles."""
    return {
        "I1": {name: [pct[name]["p0.5"], pct[name]["p99.5"]] for name in ACTIVATIONS},
        "I2": {name: [pct[name]["p0.05"], pct[name]["p99.95"]] for name in ACTIVATIONS},
    }


def export_intervals_json(
    path: Path,
    intervals: dict[str, dict[str, list[float]]],
    stability: dict[str, Any],
) -> None:
    """Escribe preactivation_intervals.json (insumo directo del Hito 3)."""
    payload = {
        "checkpoint": CHECKPOINT_META,
        "dataset": DATASET_META,
        "interval_policy": {
            "I1": "central_99_percent",
            "I2": "central_99_9_percent",
            "shape": "asymmetric",
            "scope": "global_per_activation",
        },
        "intervals": {
            band: {name: _round_interval(intervals[band][name]) for name in ACTIVATIONS}
            for band in ("I1", "I2")
        },
        "stability": stability,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_statistics_json(
    path: Path,
    exact: dict[str, ExactStreamingStats],
    pct_a: dict[str, dict[str, float]],
    pct_b: dict[str, dict[str, float]],
    channel_stats: dict[str, PerChannelStats],
) -> None:
    """Escribe preactivation_statistics.json (estadísticas completas + por canal)."""
    seed_a, seed_b = STABILITY_SEEDS
    activations_payload: dict[str, Any] = {}
    for name in ACTIVATIONS:
        s = exact[name].summary()
        activations_payload[name] = {
            "count": s["count"],
            "expected_total": EXPECTED_COUNTS[name],
            "min": s["min"],
            "max": s["max"],
            "mean": s["mean"],
            "std": s["std"],
            "sample_size": MAX_SAMPLES[name],
            "sampling_method": SAMPLING_METHOD,
            f"percentiles_seed_{seed_a}": pct_a[name],
            f"percentiles_seed_{seed_b}": pct_b[name],
        }

    channels_payload: dict[str, Any] = {}
    for act_name, pcs in channel_stats.items():
        channels_payload[act_name] = {
            "num_channels": pcs.num_channels,
            "channels": {str(ch): pcs.channel_summary(ch) for ch in range(pcs.num_channels)},
        }

    payload = {
        "checkpoint": CHECKPOINT_META,
        "dataset": DATASET_META,
        "activations": activations_payload,
        "per_channel": channels_payload,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def export_summary_md(
    path: Path,
    exact: dict[str, ExactStreamingStats],
    intervals: dict[str, dict[str, list[float]]],
    stability: dict[str, Any],
) -> None:
    """Escribe preactivation_summary.md con tablas para la metodología."""
    lines: list[str] = []
    lines.append("# Resumen de preactivaciones — Hito 2\n")
    lines.append(
        "Caracterización de las preactivaciones de la CNN base (ReducedLeNet + ReLU), "
        "entrenada sobre MNIST, para definir los intervalos de aproximación polinómica. "
        "Checkpoint: mejor época de validación (época 8). Todo en texto plano.\n"
    )

    lines.append("## Estadísticas exactas globales\n")
    lines.append("| Activación | Count | Min | Max | Mean | Std |")
    lines.append("|---|---|---|---|---|---|")
    for name in ACTIVATIONS:
        s = exact[name].summary()
        lines.append(
            f"| {name} | {s['count']:,} | {s['min']:.3f} | {s['max']:.3f} | "
            f"{s['mean']:.3f} | {s['std']:.3f} |"
        )
    lines.append("")

    lines.append("## Intervalos definitivos (asimétricos, por activación)\n")
    lines.append("| Activación | I1 (99% central) | I2 (99.9% central) |")
    lines.append("|---|---|---|")
    for name in ACTIVATIONS:
        i1 = intervals["I1"][name]
        i2 = intervals["I2"][name]
        lines.append(f"| {name} | [{i1[0]:.3f}, {i1[1]:.3f}] | [{i2[0]:.3f}, {i2[1]:.3f}] |")
    lines.append("")

    lines.append("## Estabilidad del muestreo (seeds 42 vs 123)\n")
    lines.append(
        f"- Límites comparados: {stability['limits_compared']}\n"
        f"- Límites estables: {stability['limits_stable']}\n"
        f"- Máxima diferencia absoluta: {stability['maximum_absolute_difference']:.4f}\n"
        f"- Máxima diferencia relativa: {stability['maximum_relative_difference_percent']:.3f}%\n"
        f"- Estado: {stability['status']}\n"
    )

    lines.append("## Nota metodológica\n")
    lines.append(
        "Los intervalos se mantienen asimétricos porque reflejan la distribución "
        "empírica de cada punto de activación. La simetrización habría ampliado "
        "innecesariamente el dominio, especialmente en act2, reduciendo la capacidad "
        "efectiva de los polinomios de grado fijo alrededor del origen.\n\n"
        "Los intervalos son globales por activación, no por canal. La cobertura central "
        "global del 99% o 99.9% no implica idéntica cobertura dentro de cada canal; el "
        "canal 4 de act1 (unilateral, con cola negativa dominante) es evidencia concreta "
        "de esa heterogeneidad. Se aplica un único polinomio por activación de forma "
        "deliberada.\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def _collect_channel_stats(model, data_loader, seed: int) -> dict[str, PerChannelStats]:
    """Recorre una vez para las estadísticas por canal (act1 y act2)."""
    capture = PreactivationCapture(model)
    capture.register()
    pcs = {
        "act1": PerChannelStats("act1", 6, 400_000, 254_016_000 // 6, seed),
        "act2": PerChannelStats("act2", 16, 200_000, 86_400_000 // 16, seed + 100),
    }
    with torch.inference_mode():
        for inputs, _t in data_loader:
            model(inputs)
            pcs["act1"].update(capture.values["act1"])
            pcs["act2"].update(capture.values["act2"])
    capture.remove()
    return pcs


def main() -> int:
    from ckks_benchmark.model.train import TrainingConfig, create_dataloaders

    checkpoint_path = Path("models") / "reduced_lenet_relu_best.pt"
    model, checkpoint = load_trained_model(checkpoint_path)
    describe_checkpoint(checkpoint, model)

    config = TrainingConfig()
    train_loader, _val, _test = create_dataloaders(config)

    seed_a, seed_b = STABILITY_SEEDS

    # Recorrido principal: exactas + samplers de ambas seeds.
    exact, samplers = collect_preactivations(
        model,
        train_loader,
        MAX_SAMPLES,
        EXPECTED_COUNTS,
        seeds=STABILITY_SEEDS,
    )
    pct_a = compute_percentiles(samplers[seed_a])
    pct_b = compute_percentiles(samplers[seed_b])

    # Recalcular estabilidad para persistir (mismos cálculos que compare_sampling_stability).
    limits = ("p0.05", "p0.5", "p99.5", "p99.95")
    max_abs, max_rel, n_stable = 0.0, 0.0, 0
    for name in ACTIVATIONS:
        for limit in limits:
            va, vb = pct_a[name][limit], pct_b[name][limit]
            abs_d = abs(va - vb)
            rel_d = abs_d / max(abs(va), abs(vb), 1e-12)
            if rel_d <= 0.01 or abs_d <= 0.05:
                n_stable += 1
            max_abs = max(max_abs, abs_d)
            max_rel = max(max_rel, rel_d)
    stability = {
        "seeds": list(STABILITY_SEEDS),
        "limits_compared": 12,
        "limits_stable": n_stable,
        "maximum_absolute_difference": round(max_abs, 4),
        "maximum_relative_difference_percent": round(max_rel * 100, 3),
        "status": "stable" if n_stable == 12 else "review",
    }

    # Estadísticas por canal (recorrido aparte).
    channel_stats = _collect_channel_stats(model, train_loader, seed_a)

    intervals = build_intervals(pct_a)  # valores oficiales: seed 42

    # Crear carpeta de salida y exportar.
    out_dir = Path("results") / "published"
    out_dir.mkdir(parents=True, exist_ok=True)

    export_intervals_json(out_dir / "preactivation_intervals.json", intervals, stability)
    export_statistics_json(
        out_dir / "preactivation_statistics.json", exact, pct_a, pct_b, channel_stats
    )
    export_summary_md(out_dir / "preactivation_summary.md", exact, intervals, stability)

    print("\n" + "=" * 60)
    print("EXPORTACIÓN COMPLETADA — Hito 2")
    print("=" * 60)
    for fname in (
        "preactivation_intervals.json",
        "preactivation_statistics.json",
        "preactivation_summary.md",
    ):
        print(f"  Generado: {out_dir / fname}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
