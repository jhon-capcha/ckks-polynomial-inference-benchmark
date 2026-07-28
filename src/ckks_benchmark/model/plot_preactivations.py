"""
Histogramas de preactivaciones (Hito 2).

Genera las figuras del Hito 2 en results/figures/:
    preactivation_act1_histogram.png
    preactivation_act2_histogram.png
    preactivation_act3_histogram.png
    preactivation_channel_ranges.png

Usa las submuestras reproducibles (seed oficial 42) y marca los intervalos
I1 e I2 sobre cada histograma para visualizar la cobertura.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana (guarda a archivo)
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from ckks_benchmark.model.preactivations import (  # noqa: E402
    ACTIVATIONS,
    EXPECTED_COUNTS,
    MAX_SAMPLES,
    PerChannelStats,
    PreactivationCapture,
    collect_preactivations,
    compute_percentiles,
    load_trained_model,
)

SEED = 42


def plot_activation_histogram(
    samples: torch.Tensor,
    name: str,
    i1: tuple[float, float],
    i2: tuple[float, float],
    out_path: Path,
) -> None:
    """Histograma de una activación con I1 e I2 marcados."""
    values = samples.numpy()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=200, color="steelblue", alpha=0.7, density=True)

    # Marcar los intervalos.
    for x in i1:
        ax.axvline(x, color="green", linestyle="--", linewidth=1.5)
    for x in i2:
        ax.axvline(x, color="red", linestyle=":", linewidth=1.5)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)

    ax.set_title(f"Distribución de preactivaciones — {name}")
    ax.set_xlabel("Valor de preactivación (entrada a ReLU)")
    ax.set_ylabel("Densidad")
    ax.legend(
        [
            f"I1 (99%): [{i1[0]:.2f}, {i1[1]:.2f}]",
            f"I2 (99.9%): [{i2[0]:.2f}, {i2[1]:.2f}]",
        ],
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_channel_ranges(
    channel_stats: dict[str, PerChannelStats],
    out_path: Path,
) -> None:
    """Gráfico de rangos por canal (anchoI2) para act1 y act2."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (act_name, pcs) in zip(axes, channel_stats.items()):
        channels = list(range(pcs.num_channels))
        summaries = [pcs.channel_summary(ch) for ch in channels]
        lowers = [s["p0.05"] for s in summaries]
        uppers = [s["p99.95"] for s in summaries]
        centers = [(low + up) / 2 for low, up in zip(lowers, uppers)]

        ax.errorbar(
            channels,
            centers,
            yerr=[
                [c - low for c, low in zip(centers, lowers)],
                [up - c for c, up in zip(centers, uppers)],
            ],
            fmt="o",
            capsize=4,
            color="darkorange",
        )
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(f"Rango [P0.05, P99.95] por canal — {act_name}")
        ax.set_xlabel("Canal")
        ax.set_ylabel("Valor de preactivación")
        ax.set_xticks(channels)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    from ckks_benchmark.model.train import TrainingConfig, create_dataloaders

    checkpoint_path = Path("models") / "reduced_lenet_relu_best.pt"
    model, _checkpoint = load_trained_model(checkpoint_path)

    config = TrainingConfig()
    train_loader, _val, _test = create_dataloaders(config)

    # Submuestras (seed oficial) + percentiles para marcar intervalos.
    exact, samplers = collect_preactivations(
        model,
        train_loader,
        MAX_SAMPLES,
        EXPECTED_COUNTS,
        seeds=(SEED,),
    )
    pct = compute_percentiles(samplers[SEED])

    out_dir = Path("results") / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Histograma por activación.
    for name in ACTIVATIONS:
        i1 = (pct[name]["p0.5"], pct[name]["p99.5"])
        i2 = (pct[name]["p0.05"], pct[name]["p99.95"])
        plot_activation_histogram(
            samplers[SEED][name].samples(),
            name,
            i1,
            i2,
            out_dir / f"preactivation_{name}_histogram.png",
        )

    # Rangos por canal.
    capture = PreactivationCapture(model)
    capture.register()
    channel_stats = {
        "act1": PerChannelStats("act1", 6, 400_000, 254_016_000 // 6, SEED),
        "act2": PerChannelStats("act2", 16, 200_000, 86_400_000 // 16, SEED + 100),
    }
    with torch.inference_mode():
        for inputs, _t in train_loader:
            model(inputs)
            channel_stats["act1"].update(capture.values["act1"])
            channel_stats["act2"].update(capture.values["act2"])
    capture.remove()

    plot_channel_ranges(channel_stats, out_dir / "preactivation_channel_ranges.png")

    print("=" * 60)
    print("HISTOGRAMAS GENERADOS — Hito 2")
    print("=" * 60)
    for name in ACTIVATIONS:
        print(f"  results/figures/preactivation_{name}_histogram.png")
    print("  results/figures/preactivation_channel_ranges.png")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
