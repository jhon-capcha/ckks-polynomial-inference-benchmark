"""
Extracción y caracterización de preactivaciones de la CNN base (Hito 2).

Este módulo NO cifra nada: opera en texto plano sobre el modelo ya entrenado.
Caracteriza la distribución de las preactivaciones —los valores que entran a
cada ReLU— para definir los intervalos de aproximación polinómica I1 e I2.

Preactivación de cada punto = salida de la capa lineal previa a la ReLU:
    conv1 -> (preactivación act1) -> ReLU
    conv2 -> (preactivación act2) -> ReLU
    fc1   -> (preactivación act3) -> ReLU

Diseño (responsabilidades separadas):
    ExactStreamingStats      -> count, min, max, mean, std (exactas, sin seed)
    ProportionalBatchSampler -> submuestra reproducible por seed (percentiles)

El submuestreo es aleatorio reproducible, estratificado proporcionalmente por
batch, sobre el conjunto completo de preactivaciones. No es reservoir sampling
uniforme exacto elemento por elemento; dentro de cada batch la selección es
uniforme y entre batches la asignación es proporcional. Con batches homogéneos
y recorrido completo, la diferencia frente a un muestreo uniforme global es
insignificante.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from ckks_benchmark.model.architecture import ReducedLeNet

# Total de preactivaciones esperadas por activación (split de 54 000 imágenes):
#   act1: 54000 * 6 * 28 * 28 = 254_016_000
#   act2: 54000 * 16 * 10 * 10 =  86_400_000
#   act3: 54000 * 120          =   6_480_000
EXPECTED_COUNTS = {
    "act1": 254_016_000,
    "act2": 86_400_000,
    "act3": 6_480_000,
}

MAX_SAMPLES = {
    "act1": 2_000_000,
    "act2": 2_000_000,
    "act3": 1_000_000,
}

ACTIVATIONS = ("act1", "act2", "act3")
STABILITY_SEEDS = (42, 123)

# Criterio de estabilidad (diagnóstico, no regla universal).
STABILITY_REL_TOL = 0.01  # 1 %
STABILITY_ABS_TOL = 0.05


# --------------------------------------------------------------------------- #
# Carga y validación del modelo
# --------------------------------------------------------------------------- #
def load_trained_model(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Carga el checkpoint entrenado y reconstruye ReducedLeNet en modo eval."""
    if device is None:
        device = torch.device("cpu")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No se encontró el checkpoint en: {checkpoint_path}. "
            "Ejecuta primero el entrenamiento (Hito 1)."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    required_keys = ("model_state_dict", "epoch", "architecture", "activation")
    missing = [key for key in required_keys if key not in checkpoint]
    if missing:
        raise KeyError(
            f"Al checkpoint le faltan claves esperadas: {missing}. "
            "¿Es un checkpoint generado por train.py?"
        )

    model = ReducedLeNet(activation_factory=nn.ReLU).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def describe_checkpoint(checkpoint: dict[str, Any], model: nn.Module) -> None:
    """Imprime un resumen de validación del checkpoint cargado."""
    val_metrics = checkpoint.get("validation_metrics", {})

    print("=" * 60)
    print("VALIDACIÓN DEL CHECKPOINT — Hito 2, paso 1")
    print("=" * 60)
    print("Checkpoint cargado:      OK")
    print(f"Arquitectura:            {checkpoint.get('architecture', 'N/D')}")
    print(f"Activación:              {checkpoint.get('activation', 'N/D')}")
    print(f"Mejor época:             {checkpoint.get('epoch', 'N/D')}")
    print(f"Accuracy validación:     {val_metrics.get('accuracy', float('nan')):.4f}")
    print(f"Modo entrenamiento:      {model.training}")
    print(f"PyTorch (checkpoint):    {checkpoint.get('torch_version', 'N/D')}")
    print("=" * 60)

    if model.training:
        raise RuntimeError(
            "El modelo está en modo entrenamiento (training=True). "
            "Debe estar en eval() para extraer preactivaciones."
        )


# --------------------------------------------------------------------------- #
# Captura de preactivaciones (forward hooks)
# --------------------------------------------------------------------------- #
class PreactivationCapture:
    """Captura las preactivaciones (salidas de conv1, conv2 y fc1)."""

    LAYER_MAP = {"act1": "conv1", "act2": "conv2", "act3": "fc1"}

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.values: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
            self.values[name] = output.detach().cpu()

        return hook

    def register(self) -> None:
        submodules = dict(self.model.named_modules())
        for act_name, layer_name in self.LAYER_MAP.items():
            if layer_name not in submodules:
                raise KeyError(
                    f"La capa '{layer_name}' no existe. Disponibles: {list(submodules.keys())}"
                )
            handle = submodules[layer_name].register_forward_hook(self._make_hook(act_name))
            self._handles.append(handle)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


def verify_hooks(model: nn.Module) -> None:
    """Ejecuta un batch de prueba y verifica los shapes de las preactivaciones."""
    capture = PreactivationCapture(model)
    capture.register()

    with torch.inference_mode():
        model(torch.randn(4, 1, 28, 28))

    print("=" * 60)
    print("VERIFICACIÓN DE HOOKS — Hito 2, paso 2")
    print("=" * 60)
    expected_shapes = {
        "act1": (4, 6, 28, 28),
        "act2": (4, 16, 10, 10),
        "act3": (4, 120),
    }
    for name in ACTIVATIONS:
        captured = capture.values[name]
        expected = expected_shapes[name]
        status = "OK" if tuple(captured.shape) == expected else "ERROR"
        print(f"{name}: shape {tuple(captured.shape)} | esperado {expected} | {status}")

    capture.remove()
    print("Hooks eliminados correctamente.")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# Estadísticas exactas (independientes de seed)
# --------------------------------------------------------------------------- #
class ExactStreamingStats:
    """Acumula count, min, max, mean y std exactos sobre todo el flujo."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.count: int = 0
        self.total_sum: float = 0.0
        self.total_sq_sum: float = 0.0
        self.minimum: float = float("inf")
        self.maximum: float = float("-inf")

    def update(self, values: torch.Tensor) -> None:
        flat = values.reshape(-1).to(dtype=torch.float64)
        self.count += flat.numel()
        self.total_sum += float(flat.sum().item())
        self.total_sq_sum += float((flat * flat).sum().item())
        self.minimum = min(self.minimum, float(flat.min().item()))
        self.maximum = max(self.maximum, float(flat.max().item()))

    @property
    def mean(self) -> float:
        return self.total_sum / self.count if self.count else float("nan")

    @property
    def std(self) -> float:
        if self.count < 2:
            return float("nan")
        variance = (self.total_sq_sum / self.count) - (self.mean**2)
        return float(max(variance, 0.0) ** 0.5)

    def summary(self) -> dict[str, float]:
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": self.mean,
            "std": self.std,
        }


# --------------------------------------------------------------------------- #
# Submuestreo proporcional reproducible (uno por seed)
# --------------------------------------------------------------------------- #
class ProportionalBatchSampler:
    """Submuestra estratificada proporcional por batch, con tamaño final exacto.

    Usa un objetivo acumulado para garantizar exactamente max_samples al final,
    sin error de redondeo. La selección dentro de cada batch es aleatoria y
    reproducible (generador con seed propia).
    """

    def __init__(self, name: str, max_samples: int, expected_total: int, seed: int) -> None:
        self.name = name
        self.max_samples = max_samples
        self.expected_total = expected_total
        self.seed = seed
        self._generator = torch.Generator().manual_seed(seed)
        self._chunks: list[torch.Tensor] = []
        self._seen: int = 0
        self._sampled: int = 0

    def update(self, values: torch.Tensor) -> None:
        flat32 = values.reshape(-1).to(dtype=torch.float32)
        self._seen += flat32.numel()

        target = min(
            self.max_samples,
            int(self._seen * self.max_samples / self.expected_total),
        )
        n_take = target - self._sampled
        if n_take > 0:
            n_take = min(n_take, flat32.numel())
            idx = torch.randperm(flat32.numel(), generator=self._generator)[:n_take]
            self._chunks.append(flat32[idx])
            self._sampled += n_take

    def samples(self) -> torch.Tensor:
        if not self._chunks:
            return torch.empty(0)
        return torch.cat(self._chunks)


# --------------------------------------------------------------------------- #
# Recorrido único: estadísticas exactas + samplers para cada seed
# --------------------------------------------------------------------------- #
def collect_preactivations(
    model: nn.Module,
    data_loader,
    max_samples: dict[str, int],
    expected_counts: dict[str, int],
    seeds: tuple[int, ...] = STABILITY_SEEDS,
) -> tuple[dict[str, ExactStreamingStats], dict[int, dict[str, ProportionalBatchSampler]]]:
    """Un solo recorrido: acumula estadísticas exactas y samplers por seed."""
    capture = PreactivationCapture(model)
    capture.register()

    exact = {name: ExactStreamingStats(name) for name in ACTIVATIONS}
    samplers = {
        seed: {
            name: ProportionalBatchSampler(name, max_samples[name], expected_counts[name], seed + i)
            for i, name in enumerate(ACTIVATIONS)
        }
        for seed in seeds
    }

    total_batches = len(data_loader)
    print(f"\nRecorriendo {total_batches} batches (un único forward)...")

    with torch.inference_mode():
        for batch_idx, (inputs, _targets) in enumerate(data_loader):
            model(inputs)
            for name in ACTIVATIONS:
                values = capture.values[name]
                exact[name].update(values)
                for seed in seeds:
                    samplers[seed][name].update(values)

            if (batch_idx + 1) % 200 == 0:
                print(f"  {batch_idx + 1}/{total_batches} batches procesados")

    capture.remove()
    print("Recorrido completo. Hooks eliminados.")

    for name in ACTIVATIONS:
        if exact[name].count != expected_counts[name]:
            print(
                f"  [AVISO] {name}: count real ({exact[name].count:,}) != "
                f"esperado ({expected_counts[name]:,})."
            )

    return exact, samplers


# --------------------------------------------------------------------------- #
# Percentiles y comparación de estabilidad
# --------------------------------------------------------------------------- #
_PCT_LEVELS = {
    "p0.05": 0.0005,
    "p0.5": 0.005,
    "p1": 0.01,
    "p5": 0.05,
    "p95": 0.95,
    "p99": 0.99,
    "p99.5": 0.995,
    "p99.95": 0.9995,
}

# Los 4 límites que definen I1/I2 y que se comparan para estabilidad.
_INTERVAL_LIMITS = ("p0.05", "p0.5", "p99.5", "p99.95")


def compute_percentiles(
    samplers: dict[str, ProportionalBatchSampler],
) -> dict[str, dict[str, float]]:
    """Calcula percentiles bilaterales sobre la submuestra de cada activación."""
    result: dict[str, dict[str, float]] = {}
    for name in ACTIVATIONS:
        samples = samplers[name].samples()
        qs = torch.quantile(
            samples.to(dtype=torch.float64),
            torch.tensor(list(_PCT_LEVELS.values()), dtype=torch.float64),
        )
        result[name] = {label: float(q.item()) for label, q in zip(_PCT_LEVELS, qs)}
    return result


def print_percentiles(percentiles: dict[str, dict[str, float]], seed: int) -> None:
    print("\n" + "=" * 60)
    print(f"PERCENTILES BILATERALES (seed {seed}) — Hito 2, paso 4")
    print("=" * 60)
    for name in ACTIVATIONS:
        p = percentiles[name]
        print(f"\n{name}:")
        print(f"  Cobertura 99%   [P0.5, P99.5]:   [{p['p0.5']:.3f}, {p['p99.5']:.3f}]")
        print(f"  Cobertura 99.9% [P0.05, P99.95]: [{p['p0.05']:.3f}, {p['p99.95']:.3f}]")
    print("=" * 60)


def compare_sampling_stability(
    pct_a: dict[str, dict[str, float]],
    pct_b: dict[str, dict[str, float]],
    seed_a: int,
    seed_b: int,
) -> None:
    """Compara los 12 límites (3 act × 4 percentiles) entre dos seeds."""
    print("\n" + "=" * 84)
    print(f"ESTABILIDAD DE MUESTREO — seed {seed_a} vs seed {seed_b}")
    print("=" * 84)
    header = (
        f"{'act':>4} {'límite':>8} {'seed ' + str(seed_a):>12} "
        f"{'seed ' + str(seed_b):>12} {'dif.abs':>10} {'dif.rel':>10} {'estado':>9}"
    )
    print(header)
    print("-" * 84)

    total = 0
    stable = 0
    max_abs = 0.0
    max_rel = 0.0

    for name in ACTIVATIONS:
        for limit in _INTERVAL_LIMITS:
            va = pct_a[name][limit]
            vb = pct_b[name][limit]
            abs_diff = abs(va - vb)
            rel_diff = abs_diff / max(abs(va), abs(vb), 1e-12)

            is_stable = (rel_diff <= STABILITY_REL_TOL) or (abs_diff <= STABILITY_ABS_TOL)
            total += 1
            stable += int(is_stable)
            max_abs = max(max_abs, abs_diff)
            max_rel = max(max_rel, rel_diff)

            estado = "ESTABLE" if is_stable else "REVISAR"
            print(
                f"{name:>4} {limit:>8} {va:>12.4f} {vb:>12.4f} "
                f"{abs_diff:>10.4f} {rel_diff * 100:>9.3f}% {estado:>9}"
            )

    print("-" * 84)
    global_ok = "OK" if stable == total else "REVISAR"
    print(f"Límites comparados:       {total}")
    print(f"Límites estables:         {stable}")
    print(f"Máxima diferencia abs.:   {max_abs:.4f}")
    print(f"Máxima diferencia rel.:   {max_rel * 100:.3f}%")
    print(f"Estabilidad global:       {global_ok}")
    print("=" * 84)


# --------------------------------------------------------------------------- #
# Revisión por canal (diagnóstico estructural)
# --------------------------------------------------------------------------- #
class PerChannelStats:
    """Estadísticas exactas + submuestra por canal para una activación conv."""

    def __init__(
        self,
        name: str,
        num_channels: int,
        max_samples_per_channel: int,
        expected_total_per_channel: int,
        seed: int,
    ) -> None:
        self.name = name
        self.num_channels = num_channels
        self.max_samples = max_samples_per_channel
        self.expected_total = expected_total_per_channel
        self._generators = [torch.Generator().manual_seed(seed + c) for c in range(num_channels)]
        self.minimum = [float("inf")] * num_channels
        self.maximum = [float("-inf")] * num_channels
        self._chunks: list[list[torch.Tensor]] = [[] for _ in range(num_channels)]
        self._seen = [0] * num_channels
        self._sampled = [0] * num_channels

    def update(self, values: torch.Tensor) -> None:
        c = values.shape[1]
        per_channel = values.permute(1, 0, 2, 3).reshape(c, -1).to(dtype=torch.float32)
        for ch in range(c):
            flat = per_channel[ch]
            self._seen[ch] += flat.numel()
            self.minimum[ch] = min(self.minimum[ch], float(flat.min().item()))
            self.maximum[ch] = max(self.maximum[ch], float(flat.max().item()))
            target = min(
                self.max_samples,
                int(self._seen[ch] * self.max_samples / self.expected_total),
            )
            n_take = target - self._sampled[ch]
            if n_take > 0:
                n_take = min(n_take, flat.numel())
                idx = torch.randperm(flat.numel(), generator=self._generators[ch])[:n_take]
                self._chunks[ch].append(flat[idx])
                self._sampled[ch] += n_take

    def channel_summary(self, ch: int) -> dict[str, float]:
        samples = torch.cat(self._chunks[ch]) if self._chunks[ch] else torch.empty(0)
        qs = torch.quantile(
            samples.to(dtype=torch.float64),
            torch.tensor([0.0005, 0.005, 0.995, 0.9995], dtype=torch.float64),
        )
        p005, p05, p995, p9995 = (float(q.item()) for q in qs)
        pos = max(abs(p995), 1e-9)
        return {
            "p0.05": p005,
            "p0.5": p05,
            "p99.5": p995,
            "p99.95": p9995,
            "width_i1": p995 - p05,
            "width_i2": p9995 - p005,
            "tail_ratio": abs(p05) / pos,
        }


def review_by_channel(model: nn.Module, data_loader, seed: int = 42) -> None:
    """Revisión diagnóstica por canal de act1 (6) y act2 (16)."""
    capture = PreactivationCapture(model)
    capture.register()

    pcs = {
        "act1": PerChannelStats("act1", 6, 400_000, 254_016_000 // 6, seed),
        "act2": PerChannelStats("act2", 16, 200_000, 86_400_000 // 16, seed + 100),
    }

    total_batches = len(data_loader)
    print(f"\nRevisión por canal — recorriendo {total_batches} batches...")
    with torch.inference_mode():
        for batch_idx, (inputs, _t) in enumerate(data_loader):
            model(inputs)
            pcs["act1"].update(capture.values["act1"])
            pcs["act2"].update(capture.values["act2"])
            if (batch_idx + 1) % 200 == 0:
                print(f"  {batch_idx + 1}/{total_batches}")
    capture.remove()

    for act_name, stats in pcs.items():
        print("\n" + "=" * 78)
        print(f"REVISIÓN POR CANAL — {act_name} ({stats.num_channels} canales)")
        print("=" * 78)
        print(
            f"{'ch':>3} | {'P0.05':>9} {'P0.5':>8} {'P99.5':>8} {'P99.95':>9} | "
            f"{'anchoI1':>8} {'anchoI2':>8} {'ratio':>6}"
        )
        print("-" * 78)
        rows = [(ch, stats.channel_summary(ch)) for ch in range(stats.num_channels)]
        rows.sort(key=lambda r: r[1]["width_i2"], reverse=True)
        for ch, s in rows:
            print(
                f"{ch:>3} | {s['p0.05']:>9.3f} {s['p0.5']:>8.3f} "
                f"{s['p99.5']:>8.3f} {s['p99.95']:>9.3f} | "
                f"{s['width_i1']:>8.3f} {s['width_i2']:>8.3f} {s['tail_ratio']:>6.2f}"
            )
    print("=" * 78)


# --------------------------------------------------------------------------- #
# Punto de entrada
# --------------------------------------------------------------------------- #
def main() -> int:
    from ckks_benchmark.model.train import TrainingConfig, create_dataloaders

    checkpoint_path = Path("models") / "reduced_lenet_relu_best.pt"
    model, checkpoint = load_trained_model(checkpoint_path)
    describe_checkpoint(checkpoint, model)
    verify_hooks(model)

    config = TrainingConfig()
    train_loader, _val, _test = create_dataloaders(config)

    # Recorrido único: estadísticas exactas + samplers para ambas seeds.
    exact, samplers = collect_preactivations(
        model,
        train_loader,
        MAX_SAMPLES,
        EXPECTED_COUNTS,
        seeds=STABILITY_SEEDS,
    )

    print("\n" + "=" * 60)
    print("ESTADÍSTICAS EXACTAS (streaming) — Hito 2, paso 3")
    print("=" * 60)
    for name in ACTIVATIONS:
        s = exact[name].summary()
        print(
            f"{name}: count={s['count']:,} | min={s['min']:.3f} max={s['max']:.3f} | "
            f"mean={s['mean']:.3f} std={s['std']:.3f}"
        )

    # Percentiles para cada seed.
    seed_a, seed_b = STABILITY_SEEDS
    pct_a = compute_percentiles(samplers[seed_a])
    pct_b = compute_percentiles(samplers[seed_b])

    print_percentiles(pct_a, seed_a)
    print_percentiles(pct_b, seed_b)

    # Comparación de estabilidad (12 límites).
    compare_sampling_stability(pct_a, pct_b, seed_a, seed_b)

    # Revisión por canal (diagnóstico estructural, con seed principal).
    review_by_channel(model, train_loader, seed=seed_a)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
