"""
Monitor de activaciones por hooks (Hito 3C-B).

Acumula en streaming estadísticas ligeras de act1/act2/act3 y logits durante la
inferencia, sin guardar tensores completos. Usa el algoritmo de Welford para
media y varianza numéricamente estables (importante cuando las activaciones
polinómicas producen valores grandes, p. ej. Taylor de grado alto).

Uso:
    monitor = ActivationMonitor(model)
    monitor.register()
    with torch.inference_mode():
        for x, _ in loader:
            model(x)
    stats = monitor.summary()
    monitor.remove()
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

# Nombres lógicos monitoreados. act1/act2/act3 son las activaciones; "logits"
# es la salida final del modelo (capturada con un hook sobre fc2).
MONITORED_LAYERS = ("act1", "act2", "act3")
LOGITS_LAYER = "fc2"


@dataclass
class RunningTensorStats:
    """Estadísticas en streaming de un flujo de valores (Welford + min/max)."""

    count: int = 0
    finite_count: int = 0
    non_finite_count: int = 0
    minimum: float = float("inf")
    maximum: float = float("-inf")
    max_abs: float = 0.0
    _mean: float = 0.0  # media de Welford (solo sobre finitos)
    _m2: float = 0.0  # suma de cuadrados de diferencias (Welford)

    def update(self, tensor: torch.Tensor) -> None:
        flat = tensor.detach().reshape(-1)
        n_total = flat.numel()
        self.count += n_total

        finite_mask = torch.isfinite(flat)
        n_non_finite = int((~finite_mask).sum().item())
        self.non_finite_count += n_non_finite

        finite = flat[finite_mask]
        n_finite = finite.numel()
        if n_finite == 0:
            return
        self.finite_count += n_finite

        # min / max / max_abs sobre finitos.
        fmin = float(finite.min().item())
        fmax = float(finite.max().item())
        self.minimum = min(self.minimum, fmin)
        self.maximum = max(self.maximum, fmax)
        self.max_abs = max(self.max_abs, abs(fmin), abs(fmax))

        # Welford por lotes (Chan et al.): combina el bloque con lo acumulado.
        block = finite.to(dtype=torch.float64)
        block_mean = float(block.mean().item())
        block_m2 = float(((block - block_mean) ** 2).sum().item())

        if self.finite_count == n_finite:
            # Primer bloque.
            self._mean = block_mean
            self._m2 = block_m2
        else:
            prev_n = self.finite_count - n_finite
            delta = block_mean - self._mean
            total_n = self.finite_count
            self._mean += delta * (n_finite / total_n)
            self._m2 += block_m2 + (delta**2) * prev_n * n_finite / total_n

    @property
    def mean(self) -> float:
        return self._mean if self.finite_count > 0 else float("nan")

    @property
    def std(self) -> float:
        if self.finite_count < 2:
            return float("nan")
        return float((self._m2 / self.finite_count) ** 0.5)

    def as_dict(self) -> dict[str, float]:
        return {
            "count": self.count,
            "finite_count": self.finite_count,
            "non_finite_count": self.non_finite_count,
            "min": self.minimum if self.finite_count else float("nan"),
            "max": self.maximum if self.finite_count else float("nan"),
            "mean": self.mean,
            "std": self.std,
            "max_abs": self.max_abs if self.finite_count else float("nan"),
        }


@dataclass
class ActivationMonitor:
    """Registra hooks sobre act1/act2/act3 y fc2 (logits) para acumular stats."""

    model: nn.Module
    stats: dict[str, RunningTensorStats] = field(default_factory=dict)
    _handles: list = field(default_factory=list)
    first_non_finite_layer: str | None = None

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.stats = {name: RunningTensorStats() for name in (*MONITORED_LAYERS, "logits")}
        self.first_non_finite_layer = None

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
            self.stats[name].update(output)
            # Registrar la primera capa donde aparece no-finitud.
            if self.first_non_finite_layer is None:
                if not torch.all(torch.isfinite(output)):
                    self.first_non_finite_layer = name

        return hook

    def register(self) -> None:
        submodules = dict(self.model.named_modules())
        # Activaciones.
        for name in MONITORED_LAYERS:
            if name not in submodules:
                raise KeyError(f"La capa '{name}' no existe en el modelo.")
            self._handles.append(submodules[name].register_forward_hook(self._make_hook(name)))
        # Logits (salida de fc2).
        if LOGITS_LAYER not in submodules:
            raise KeyError(f"La capa '{LOGITS_LAYER}' no existe en el modelo.")
        self._handles.append(
            submodules[LOGITS_LAYER].register_forward_hook(self._make_hook("logits"))
        )

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def summary(self) -> dict[str, dict[str, float]]:
        return {name: s.as_dict() for name, s in self.stats.items()}

    def __enter__(self) -> ActivationMonitor:
        self.register()
        return self

    def __exit__(self, *exc) -> None:
        self.remove()
