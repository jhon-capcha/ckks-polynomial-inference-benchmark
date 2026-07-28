"""
Pruebas de la agrupación y benchmark de la CNN polinómica (Hito 3C-B).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ckks_benchmark.approximation.registry import (
    generate_all,
    load_and_validate_intervals,
)
from ckks_benchmark.model.benchmark_polynomial_cnn import (
    accuracy_and_f1,
    group_into_triplets,
)

INTERVALS = Path("results/published/preactivation_intervals.json")


@pytest.fixture(scope="module")
def polys():
    return generate_all(load_and_validate_intervals(INTERVALS))


def test_grouping_produces_24_triplets(polys):
    triplets = group_into_triplets(polys)
    assert len(triplets) == 24


def test_each_triplet_has_three_activations(polys):
    triplets = group_into_triplets(polys)
    for config_id, triplet in triplets.items():
        assert set(triplet) == {"act1", "act2", "act3"}


def test_triplet_ids_unique(polys):
    triplets = group_into_triplets(polys)
    assert len(set(triplets)) == 24


def test_triplet_coherent_signature(polys):
    """Cada terna comparte método, grado e intervalo."""
    triplets = group_into_triplets(polys)
    for config_id, triplet in triplets.items():
        sigs = {(p.method, p.degree, p.interval_name) for p in triplet.values()}
        assert len(sigs) == 1


def test_accuracy_f1_perfect():
    # Las 10 clases presentes y todas correctas -> F1 macro = 1.0
    preds = np.arange(10)
    labels = np.arange(10)
    acc, f1 = accuracy_and_f1(preds, labels)
    assert acc == 1.0
    assert f1 == pytest.approx(1.0)


def test_accuracy_f1_empty():
    acc, f1 = accuracy_and_f1(np.array([]), np.array([]))
    assert np.isnan(acc)
    assert np.isnan(f1)


def test_accuracy_half():
    preds = np.array([0, 9, 2, 9])  # 0 y 2 correctos, dos errados
    labels = np.array([0, 1, 2, 3])
    acc, _ = accuracy_and_f1(preds, labels)
    assert acc == 0.5
