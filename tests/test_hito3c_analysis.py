"""
Pruebas del análisis conjunto Hito 3B ↔ 3C-B (fase 3C-C, Parte 1).
"""

from __future__ import annotations

from ckks_benchmark.analysis.hito3c_analysis import (
    ELIGIBLE_ACCURACY_MIN,
    VIABLE_ACCURACY_FLOOR,
    build_combined_analysis,
    compute_correlation_hierarchy,
)


def test_produces_24_configurations():
    analyses = build_combined_analysis()
    assert len(analyses) == 24


def test_configuration_ids_unique():
    analyses = build_combined_analysis()
    ids = [a.configuration_id for a in analyses]
    assert len(set(ids)) == 24


def test_state_hierarchy_consistent():
    """eligible ⊆ viable ⊆ valid (cadena de estados)."""
    for a in build_combined_analysis():
        if a.eligible_for_ckks:
            assert a.practically_viable
        if a.practically_viable:
            assert a.valid


def test_eligibility_respects_thresholds():
    for a in build_combined_analysis():
        if a.eligible_for_ckks:
            assert a.validation_accuracy >= ELIGIBLE_ACCURACY_MIN
        if a.practically_viable and a.valid:
            assert a.validation_accuracy >= VIABLE_ACCURACY_FLOOR


def test_invalid_configs_not_eligible():
    """Las inválidas nunca son viables ni elegibles."""
    for a in build_combined_analysis():
        if not a.valid:
            assert not a.practically_viable
            assert not a.eligible_for_ckks


def test_taylor_collapsed_not_viable():
    """Taylor d3/d5 (colapsado ~0.1) es válido pero no viable."""
    analyses = {a.configuration_id: a for a in build_combined_analysis()}
    for cid in ("taylor_d3_I1", "taylor_d5_I1"):
        a = analyses[cid]
        assert a.valid  # finito
        assert not a.practically_viable  # colapsado


def test_aggregation_uses_max_of_three():
    """functional_mae_max debe ser >= functional_mae_mean."""
    for a in build_combined_analysis():
        assert a.functional_mae_max >= a.functional_mae_mean


def test_amplification_computed():
    """Las amplificaciones deben ser positivas para configs finitas."""
    for a in build_combined_analysis():
        if a.valid:
            assert a.amplification_act1_act2 > 0
            assert a.amplification_act2_act3 > 0


def test_expected_counts():
    analyses = build_combined_analysis()
    assert sum(1 for a in analyses if a.valid) == 18
    assert sum(1 for a in analyses if a.practically_viable) == 14
    assert sum(1 for a in analyses if a.eligible_for_ckks) == 10


def test_correlation_hierarchy_labeled():
    """La jerarquía de correlaciones debe tener las etiquetas predefinidas."""
    analyses = build_combined_analysis()
    corr = compute_correlation_hierarchy(analyses)
    assert "primary_analysis" in corr
    assert "sensitivity_all_valid" in corr
    assert "sensitivity_valid_no_taylor" in corr


def test_reproducible():
    import numpy as np

    a1 = build_combined_analysis()
    a2 = build_combined_analysis()

    def equal_or_both_nan(x: float, y: float) -> bool:
        if np.isnan(x) and np.isnan(y):
            return True
        return x == y

    for x, y in zip(a1, a2):
        assert x.configuration_id == y.configuration_id
        assert equal_or_both_nan(x.validation_accuracy, y.validation_accuracy)
        assert equal_or_both_nan(x.functional_mae_mean, y.functional_mae_mean)
        assert equal_or_both_nan(x.act3_max_abs, y.act3_max_abs)
