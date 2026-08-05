"""
Infraestructura de medición de latencia (Hito 5A).

Timers de alta resolución con repeticiones, warm-up y agregación estadística.
Diseñado para aislar la latencia CKKS online (encrypt/act3/fc2/decrypt) de todo
lo auxiliar (carga de datos, construcción de modelos, serialización).

Estadística jerarquizada: mediana y P95 como principales (robustas ante
interrupciones del SO); media/desv/P99/CV como complementarias. El mínimo NO se
usa como cifra principal.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StageTimer:
    """Acumula tiempos por etapa dentro de una inferencia."""

    times: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def measure(self, stage: str):
        """Context manager que mide el tiempo de una etapa."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.times[stage] = time.perf_counter() - t0

    def total(self, stages: list[str] | None = None) -> float:
        """Suma de los tiempos de las etapas indicadas (o todas)."""
        if stages is None:
            return sum(self.times.values())
        return sum(self.times.get(s, 0.0) for s in stages)


def summarize_latency(samples: list[float]) -> dict:
    """Estadística descriptiva de una lista de tiempos (segundos).

    Jerarquía: median y p95 principales; el resto complementarias.
    """
    arr = np.asarray(samples, dtype=np.float64)
    if arr.size == 0:
        return {"sample_count": 0}
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "sample_count": int(arr.size),
        # Principales.
        "median_seconds": float(np.median(arr)),
        "p95_seconds": float(np.percentile(arr, 95)),
        # Complementarias.
        "mean_seconds": mean,
        "std_seconds": std,
        "min_seconds": float(np.min(arr)),
        "max_seconds": float(np.max(arr)),
        "p05_seconds": float(np.percentile(arr, 5)),
        "p25_seconds": float(np.percentile(arr, 25)),
        "p75_seconds": float(np.percentile(arr, 75)),
        "p99_seconds": float(np.percentile(arr, 99)),
        "coefficient_of_variation": (std / mean) if mean > 0 else 0.0,
    }


def amortized_latency(
    setup_seconds: float, inference_seconds: float, m_values: list[int]
) -> dict[int, float]:
    """Latencia amortizada T(M) = T_setup/M + T_inferencia para varios M.

    Muestra cuánto pesa el setup cuando se hacen pocas inferencias.
    """
    return {m: setup_seconds / m + inference_seconds for m in m_values}


def build_execution_order(
    n_configs: int, n_images: int, n_repetitions: int, seed: int
) -> list[tuple[int, int, int]]:
    """Genera el orden de ejecución aleatorizado (config, imagen, repetición).

    Aleatorizar evita que cambios térmicos o de carga afecten sistemáticamente
    a una configuración. La seed se guarda para reproducibilidad.
    """
    rng = np.random.default_rng(seed)
    order = [
        (c, i, r) for r in range(n_repetitions) for c in range(n_configs) for i in range(n_images)
    ]
    rng.shuffle(order)
    return order
