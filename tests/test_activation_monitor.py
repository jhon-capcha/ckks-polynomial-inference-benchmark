"""
Pruebas del monitor de activaciones (Hito 3C-B).

Cubren estadísticas conocidas, acumulación multi-batch, detección de no-finitud,
registro/retiro de hooks, no-modificación de la salida, reinicio, y resumen
sobre estado vacío.
"""

from __future__ import annotations

import numpy as np
import torch

from ckks_benchmark.model.activation_monitor import (
    ActivationMonitor,
    RunningTensorStats,
)


# --- RunningTensorStats ---
def test_known_statistics():
    s = RunningTensorStats()
    s.update(torch.tensor([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]))
    d = s.as_dict()
    assert d["mean"] == 5.0
    assert d["min"] == 2.0
    assert d["max"] == 9.0
    assert d["max_abs"] == 9.0
    # std poblacional de ese conjunto clásico = 2.0
    assert abs(d["std"] - 2.0) < 1e-9


def test_multi_batch_accumulation():
    s = RunningTensorStats()
    s.update(torch.tensor([1.0, 2.0, 3.0]))
    s.update(torch.tensor([4.0, 5.0]))
    s.update(torch.tensor([6.0]))
    d = s.as_dict()
    assert d["mean"] == 3.5  # media de 1..6
    assert d["count"] == 6
    assert d["finite_count"] == 6


def test_detects_nan():
    s = RunningTensorStats()
    s.update(torch.tensor([1.0, float("nan"), 3.0]))
    d = s.as_dict()
    assert d["non_finite_count"] == 1
    assert d["finite_count"] == 2
    assert d["mean"] == 2.0  # media de [1, 3]


def test_detects_infinity():
    s = RunningTensorStats()
    s.update(torch.tensor([1.0, float("inf"), float("-inf"), 5.0]))
    d = s.as_dict()
    assert d["non_finite_count"] == 2
    assert d["mean"] == 3.0  # media de [1, 5]


def test_max_abs_with_negatives():
    s = RunningTensorStats()
    s.update(torch.tensor([-10.0, 2.0, 3.0]))
    d = s.as_dict()
    assert d["max_abs"] == 10.0  # |−10| domina


def test_empty_stats():
    s = RunningTensorStats()
    d = s.as_dict()
    assert d["count"] == 0
    assert np.isnan(d["mean"])
    assert np.isnan(d["std"])


# --- ActivationMonitor con un modelo simple ---
class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.act1 = torch.nn.Identity()
        self.act2 = torch.nn.Identity()
        self.act3 = torch.nn.Identity()
        self.fc2 = torch.nn.Linear(4, 10)

    def forward(self, x):
        x = self.act1(x)
        x = self.act2(x)
        x = self.act3(x)
        return self.fc2(x)


def test_monitor_registers_and_removes():
    model = _TinyModel()
    monitor = ActivationMonitor(model)
    monitor.register()
    assert len(monitor._handles) == 4  # act1, act2, act3, logits
    monitor.remove()
    assert len(monitor._handles) == 0


def test_monitor_does_not_change_output():
    model = _TinyModel()
    x = torch.randn(8, 4)
    with torch.inference_mode():
        out_before = model(x).clone()
    monitor = ActivationMonitor(model)
    monitor.register()
    with torch.inference_mode():
        out_after = model(x)
    monitor.remove()
    assert torch.equal(out_before, out_after)


def test_monitor_collects_stats():
    model = _TinyModel()
    monitor = ActivationMonitor(model)
    monitor.register()
    with torch.inference_mode():
        model(torch.randn(16, 4))
    stats = monitor.summary()
    monitor.remove()
    for name in ("act1", "act2", "act3", "logits"):
        assert name in stats
        assert stats[name]["count"] > 0


def test_monitor_context_manager():
    model = _TinyModel()
    with ActivationMonitor(model) as monitor:
        with torch.inference_mode():
            model(torch.randn(8, 4))
        stats = monitor.summary()
    assert stats["logits"]["count"] > 0
    assert len(monitor._handles) == 0  # removido al salir del with


def test_monitor_reset():
    model = _TinyModel()
    monitor = ActivationMonitor(model)
    monitor.register()
    with torch.inference_mode():
        model(torch.randn(8, 4))
    monitor.reset()
    for name in ("act1", "act2", "act3", "logits"):
        assert monitor.stats[name].count == 0
    monitor.remove()


def test_monitor_tracks_first_non_finite_layer():
    """Un modelo que produce NaN en act2 debe registrarlo como primera capa."""

    class _NanModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.act1 = torch.nn.Identity()
            self.act2 = torch.nn.Identity()
            self.act3 = torch.nn.Identity()
            self.fc2 = torch.nn.Linear(4, 10)

        def forward(self, x):
            x = self.act1(x)
            x = self.act2(x)
            x = x * float("nan")  # introduce NaN después de act2
            x = self.act3(x)
            return self.fc2(x)

    model = _NanModel()
    monitor = ActivationMonitor(model)
    monitor.register()
    with torch.inference_mode():
        model(torch.randn(8, 4))
    monitor.remove()
    # act3 es la primera capa monitoreada que ve el NaN.
    assert monitor.first_non_finite_layer == "act3"
