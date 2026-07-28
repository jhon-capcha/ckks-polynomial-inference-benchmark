"""
Muestreo reproducible de preactivaciones de VALIDACIÓN para los pilotos del Hito 3.

Genera un archivo .npz con muestras de act1/act2/act3 sobre el conjunto de
validación (no usado para construir los intervalos, que vienen de entrenamiento).
Esto permite iterar los pilotos de Taylor/Chebyshev sin recorrer MNIST cada vez.

Reutiliza ProportionalBatchSampler (validado en el Hito 2): muestreo proporcional
por batch, vectorizado, reproducible, con tamaño exacto. No estratifica por clase
MNIST (se muestrean valores internos de activación, no imágenes).

Salidas:
    data/processed/preactivations_validation_sample.npz   (act1, act2, act3)
    data/processed/preactivations_validation_sample.json  (metadatos + hash)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Sized
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ckks_benchmark.model.preactivations import (
    ACTIVATIONS,
    PreactivationCapture,
    ProportionalBatchSampler,
    load_trained_model,
)

# Intervalos definitivos del Hito 2 (para calcular P_fuera). Seed oficial 42.
INTERVALS = {
    "act1": {"I1": (-6.6096, 7.2350), "I2": (-8.4503, 8.7557)},
    "act2": {"I1": (-18.4731, 9.0930), "I2": (-24.8339, 12.2336)},
    "act3": {"I1": (-15.7802, 12.5193), "I2": (-22.1493, 16.3974)},
}

ELEMENTS_PER_IMAGE = {"act1": 6 * 28 * 28, "act2": 16 * 10 * 10, "act3": 120}


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _fraction_outside(samples: np.ndarray, interval: tuple[float, float]) -> float:
    low, high = interval
    outside = np.count_nonzero((samples < low) | (samples > high))
    return float(outside / samples.size) if samples.size else float("nan")


def sample_validation_preactivations(
    checkpoint_path: Path,
    max_samples: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Recorre el conjunto de validación y devuelve muestras + metadatos."""
    from ckks_benchmark.model.train import TrainingConfig, create_dataloaders

    model, checkpoint = load_trained_model(checkpoint_path)

    config = TrainingConfig()
    _train, validation_loader, _test = create_dataloaders(config)

    num_images = config.validation_size
    dataset = validation_loader.dataset
    if not isinstance(dataset, Sized):
        raise TypeError("El dataset de validación no expone una longitud.")
    observed_images = len(dataset)
    if observed_images != num_images:
        raise ValueError(
            "El tamaño del dataset de validación no coincide con la configuración: "
            f"esperado={num_images}, observado={observed_images}"
        )

    expected_counts = {name: num_images * ELEMENTS_PER_IMAGE[name] for name in ACTIVATIONS}

    capture = PreactivationCapture(model)
    capture.register()

    samplers = {
        name: ProportionalBatchSampler(name, max_samples, expected_counts[name], seed + i)
        for i, name in enumerate(ACTIVATIONS)
    }
    exact_count = {name: 0 for name in ACTIVATIONS}

    print(f"Recorriendo validación ({len(validation_loader)} batches)...")
    with torch.inference_mode():
        for inputs, _targets in validation_loader:
            model(inputs)
            for name in ACTIVATIONS:
                values = capture.values[name]
                samplers[name].update(values)
                exact_count[name] += values.numel()
    capture.remove()

    # Verificación: conteo real == esperado.
    for name in ACTIVATIONS:
        if exact_count[name] != expected_counts[name]:
            print(
                f"  [AVISO] {name}: conteo real ({exact_count[name]:,}) != "
                f"esperado ({expected_counts[name]:,})."
            )

    # Extraer muestras como arrays float64.
    samples = {
        name: samplers[name].samples().to(dtype=torch.float64).numpy() for name in ACTIVATIONS
    }

    # Validación de integridad.
    for name in ACTIVATIONS:
        arr = samples[name]
        if arr.size == 0:
            raise ValueError(f"{name}: muestra vacía.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name}: contiene NaN o infinito.")

    # Metadatos por activación.
    layer_meta: dict[str, Any] = {}
    for name in ACTIVATIONS:
        arr = samples[name]
        layer_meta[name] = {
            "source_layer": PreactivationCapture.LAYER_MAP[name],
            "sample_count": int(arr.size),
            "expected_total": expected_counts[name],
            "observed_total": exact_count[name],
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "fraction_outside_I1": _fraction_outside(arr, INTERVALS[name]["I1"]),
            "fraction_outside_I2": _fraction_outside(arr, INTERVALS[name]["I2"]),
        }

    metadata = {
        "dataset": "MNIST",
        "split": "validation",
        "split_size": num_images,
        "checkpoint": str(checkpoint_path).replace("\\", "/"),
        "checkpoint_sha256": _sha256_of_file(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "architecture": checkpoint.get("architecture"),
        "activation_original": checkpoint.get("activation"),
        "sampling_seed": seed,
        "max_samples_per_activation": max_samples,
        "sampling_method": (
            "Muestreo proporcional por batch del flujo de validación "
            "(ProportionalBatchSampler). No estratificado por clase MNIST."
        ),
        "dtype": "float64",
        "layers": layer_meta,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "generated_at": datetime.now().isoformat(),
    }

    return samples, metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Muestrea preactivaciones de validación para los pilotos del Hito 3."
    )
    parser.add_argument("--checkpoint", type=str, default="models/reduced_lenet_relu_best.pt")
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/preactivations_validation_sample.npz",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples, metadata = sample_validation_preactivations(
        checkpoint_path, args.max_samples, args.seed
    )

    # Guardar .npz comprimido.
    np.savez_compressed(
        output_path,
        act1=samples["act1"],
        act2=samples["act2"],
        act3=samples["act3"],
    )

    # Hash del .npz y JSON de metadatos.
    metadata["npz_sha256"] = _sha256_of_file(output_path)
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # Resumen.
    print("\n" + "=" * 66)
    print("MUESTRA DE PREACTIVACIONES (VALIDACIÓN) — Hito 3A")
    print("=" * 66)
    print(f"Checkpoint:   {metadata['checkpoint']} (época {metadata['checkpoint_epoch']})")
    print(f"Split:        {metadata['split']} ({metadata['split_size']} imágenes)")
    print(f"Seed:         {metadata['sampling_seed']}")
    for name in ACTIVATIONS:
        m = metadata["layers"][name]
        print(
            f"  {name}: {m['sample_count']:,} muestras | "
            f"min={m['min']:.3f} max={m['max']:.3f} | "
            f"fuera I1={m['fraction_outside_I1'] * 100:.3f}% "
            f"fuera I2={m['fraction_outside_I2'] * 100:.3f}%"
        )
    print(f"\n.npz:  {output_path}")
    print(f".json: {json_path}")
    print(f"SHA-256 (.npz): {metadata['npz_sha256'][:16]}...")
    print("=" * 66)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
