"""
Entrenamiento reproducible de la CNN base sobre MNIST.

Este módulo entrena ReducedLeNet con ReLU en texto plano y conserva el
checkpoint con mejor accuracy de validación.

Partición utilizada:
    - Entrenamiento: 54 000 imágenes.
    - Validación:     6 000 imágenes.
    - Prueba:        10 000 imágenes oficiales de MNIST.

La métrica final sobre test se calcula únicamente después de seleccionar
el mejor checkpoint mediante el conjunto de validación.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from ckks_benchmark.model.architecture import (
    ReducedLeNet,
    count_trainable_parameters,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuración reproducible del entrenamiento base."""

    seed: int = 42
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 0.001
    train_size: int = 54_000
    validation_size: int = 6_000
    num_workers: int = 0
    data_dir: str = "data"
    model_dir: str = "models"
    checkpoint_name: str = "reduced_lenet_relu_best.pt"
    metrics_name: str = "reduced_lenet_relu_metrics.json"


@dataclass
class ClassificationMetrics:
    """Métricas de clasificación de una época o conjunto de evaluación."""

    loss: float
    accuracy: float
    macro_f1: float


def set_global_seed(seed: int) -> None:
    """Fija las fuentes de aleatoriedad utilizadas durante el entrenamiento."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # El proyecto se ejecuta en CPU, pero se deja definido para consistencia.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Fuerza algoritmos deterministas cuando PyTorch dispone de una ruta válida.
    torch.use_deterministic_algorithms(True)


def create_dataloaders(
    config: TrainingConfig,
) -> tuple[DataLoader[Any], DataLoader[Any], DataLoader[Any]]:
    """Descarga MNIST y construye los DataLoaders reproducibles."""

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    data_root = Path(config.data_dir)

    full_train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transform,
    )

    expected_total = config.train_size + config.validation_size

    if len(full_train_dataset) != expected_total:
        raise ValueError(
            "La partición solicitada no coincide con el tamaño de MNIST: "
            f"{config.train_size} + {config.validation_size} != "
            f"{len(full_train_dataset)}."
        )

    split_generator = torch.Generator().manual_seed(config.seed)

    train_dataset, validation_dataset = random_split(
        full_train_dataset,
        lengths=[config.train_size, config.validation_size],
        generator=split_generator,
    )

    # Generador independiente para el orden aleatorio de los batches.
    loader_generator = torch.Generator().manual_seed(config.seed)

    common_loader_args = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": False,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=loader_generator,
        **common_loader_args,
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **common_loader_args,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_loader_args,
    )

    return train_loader, validation_loader, test_loader


def update_confusion_matrix(
    confusion_matrix: Tensor,
    targets: Tensor,
    predictions: Tensor,
    num_classes: int = 10,
) -> None:
    """Acumula una matriz de confusión sin depender de scikit-learn."""
    encoded = targets * num_classes + predictions

    counts = torch.bincount(
        encoded,
        minlength=num_classes * num_classes,
    )

    confusion_matrix += counts.reshape(num_classes, num_classes).cpu()


def macro_f1_from_confusion_matrix(confusion_matrix: Tensor) -> float:
    """Calcula F1 macro a partir de una matriz de confusión."""
    matrix = confusion_matrix.to(dtype=torch.float64)

    true_positives = matrix.diag()
    false_positives = matrix.sum(dim=0) - true_positives
    false_negatives = matrix.sum(dim=1) - true_positives

    denominator = 2.0 * true_positives + false_positives + false_negatives

    per_class_f1 = torch.where(
        denominator > 0,
        (2.0 * true_positives) / denominator,
        torch.zeros_like(denominator),
    )

    return float(per_class_f1.mean().item())


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: Adam,
    device: torch.device,
) -> ClassificationMetrics:
    """Entrena el modelo durante una época."""
    model.train()

    running_loss = 0.0
    correct_predictions = 0
    processed_samples = 0
    confusion_matrix = torch.zeros((10, 10), dtype=torch.int64)

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(inputs)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = targets.size(0)

        running_loss += loss.item() * batch_size
        correct_predictions += (predictions == targets).sum().item()
        processed_samples += batch_size

        update_confusion_matrix(
            confusion_matrix,
            targets.detach().cpu(),
            predictions.detach().cpu(),
        )

    return ClassificationMetrics(
        loss=running_loss / processed_samples,
        accuracy=correct_predictions / processed_samples,
        macro_f1=macro_f1_from_confusion_matrix(confusion_matrix),
    )


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
) -> ClassificationMetrics:
    """Evalúa el modelo sin modificar sus parámetros."""
    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    processed_samples = 0
    confusion_matrix = torch.zeros((10, 10), dtype=torch.int64)

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = criterion(logits, targets)

        predictions = logits.argmax(dim=1)
        batch_size = targets.size(0)

        running_loss += loss.item() * batch_size
        correct_predictions += (predictions == targets).sum().item()
        processed_samples += batch_size

        update_confusion_matrix(
            confusion_matrix,
            targets.cpu(),
            predictions.cpu(),
        )

    return ClassificationMetrics(
        loss=running_loss / processed_samples,
        accuracy=correct_predictions / processed_samples,
        macro_f1=macro_f1_from_confusion_matrix(confusion_matrix),
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Adam,
    epoch: int,
    validation_metrics: ClassificationMetrics,
    config: TrainingConfig,
) -> None:
    """Guarda el checkpoint con mejor accuracy de validación."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_metrics": asdict(validation_metrics),
        "training_config": asdict(config),
        "architecture": model.__class__.__name__,
        "activation": "ReLU",
        "torch_version": torch.__version__,
    }

    torch.save(checkpoint, path)


def load_best_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Carga en el modelo el mejor checkpoint guardado."""
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def run_training(config: TrainingConfig) -> dict[str, Any]:
    """Ejecuta el entrenamiento completo y devuelve el resumen final."""
    set_global_seed(config.seed)

    device = torch.device("cpu")
    model_directory = Path(config.model_dir)
    model_directory.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_directory / config.checkpoint_name
    metrics_path = model_directory / config.metrics_name

    train_loader, validation_loader, test_loader = create_dataloaders(config)

    model = ReducedLeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    print("=" * 72)
    print("ENTRENAMIENTO BASE — REDUCED LENET + RELU + MNIST")
    print("=" * 72)
    print(f"Dispositivo:             {device}")
    print(f"PyTorch:                 {torch.__version__}")
    print(f"Seed:                    {config.seed}")
    print(f"Train / validación:      {config.train_size} / {config.validation_size}")
    print(f"Test:                    {len(test_loader.dataset)}")
    print(f"Batch size:              {config.batch_size}")
    print(f"Épocas:                  {config.epochs}")
    print(f"Learning rate:           {config.learning_rate}")
    print(f"Parámetros entrenables:  {count_trainable_parameters(model):,}")
    print(f"Checkpoint:              {checkpoint_path}")
    print("=" * 72)

    best_validation_accuracy = float("-inf")
    history: list[dict[str, Any]] = []
    total_start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        validation_metrics = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        elapsed_seconds = time.perf_counter() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "elapsed_seconds": elapsed_seconds,
                "train": asdict(train_metrics),
                "validation": asdict(validation_metrics),
            }
        )

        improved = validation_metrics.accuracy > best_validation_accuracy

        if improved:
            best_validation_accuracy = validation_metrics.accuracy

            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                validation_metrics,
                config,
            )

        marker = " [MEJOR]" if improved else ""

        print(
            f"Época {epoch:02d}/{config.epochs} | "
            f"train loss={train_metrics.loss:.4f} "
            f"acc={train_metrics.accuracy:.4f} "
            f"f1={train_metrics.macro_f1:.4f} | "
            f"val loss={validation_metrics.loss:.4f} "
            f"acc={validation_metrics.accuracy:.4f} "
            f"f1={validation_metrics.macro_f1:.4f} | "
            f"{elapsed_seconds:.2f}s{marker}"
        )

    total_elapsed_seconds = time.perf_counter() - total_start

    best_checkpoint = load_best_checkpoint(
        checkpoint_path,
        model,
        device,
    )

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    summary = {
        "config": asdict(config),
        "device": str(device),
        "torch_version": torch.__version__,
        "architecture": model.__class__.__name__,
        "trainable_parameters": count_trainable_parameters(model),
        "best_epoch": best_checkpoint["epoch"],
        "best_validation_metrics": best_checkpoint["validation_metrics"],
        "test_metrics": asdict(test_metrics),
        "total_elapsed_seconds": total_elapsed_seconds,
        "checkpoint_path": str(checkpoint_path),
        "history": history,
    }

    metrics_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("RESULTADO FINAL — MEJOR CHECKPOINT")
    print("=" * 72)
    print(f"Mejor época:             {summary['best_epoch']}")
    print(f"Accuracy validación:    {summary['best_validation_metrics']['accuracy']:.4f}")
    print(f"F1 macro validación:    {summary['best_validation_metrics']['macro_f1']:.4f}")
    print(f"Accuracy test:           {test_metrics.accuracy:.4f}")
    print(f"F1 macro test:           {test_metrics.macro_f1:.4f}")
    print(f"Loss test:               {test_metrics.loss:.4f}")
    print(f"Tiempo total:            {total_elapsed_seconds:.2f}s")
    print(f"Checkpoint guardado en:  {checkpoint_path}")
    print(f"Métricas guardadas en:   {metrics_path}")
    print("=" * 72)

    return summary


def parse_args() -> argparse.Namespace:
    """Procesa argumentos opcionales de línea de comandos."""
    parser = argparse.ArgumentParser(description="Entrena ReducedLeNet con ReLU sobre MNIST.")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> int:
    """Punto de entrada del entrenamiento."""
    args = parse_args()

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
