"""
Pruebas del motor de selección de shortlist (fase 3C-C, Parte 2).
"""

from __future__ import annotations

from ckks_benchmark.analysis.hito3c_analysis import build_combined_analysis
from ckks_benchmark.analysis.selection import select_shortlist


def _run():
    analyses = build_combined_analysis()
    return analyses, *select_shortlist(analyses)


def test_selection_only_eligible():
    """Todas las seleccionadas deben ser elegibles para CKKS."""
    analyses, selected, _diag, _excl = _run()
    by_id = {a.configuration_id: a for a in analyses}
    for s in selected:
        assert by_id[s["configuration_id"]].eligible_for_ckks


def test_no_invalid_selected():
    """Ninguna inválida puede estar seleccionada."""
    analyses, selected, _diag, _excl = _run()
    by_id = {a.configuration_id: a for a in analyses}
    for s in selected:
        assert by_id[s["configuration_id"]].valid


def test_selection_ids_unique():
    _analyses, selected, _diag, _excl = _run()
    ids = [s["configuration_id"] for s in selected]
    assert len(set(ids)) == len(ids)


def test_method_diversity():
    """La shortlist debe incluir ambos métodos (Cheby y LSQ)."""
    _analyses, selected, _diag, _excl = _run()
    methods = {s["method"] for s in selected}
    assert "chebyshev" in methods
    assert "least_squares" in methods


def test_degree_diversity():
    """Debe cubrir varios grados (no solo el de máxima accuracy)."""
    _analyses, selected, _diag, _excl = _run()
    degrees = {s["degree"] for s in selected}
    assert len(degrees) >= 3  # al menos grados 3, 5, 7


def test_taylor_not_in_shortlist():
    """Taylor no debe estar en selected_for_ckks (solo como diagnóstico)."""
    _analyses, selected, _diag, _excl = _run()
    for s in selected:
        assert s["method"] != "taylor"


def test_diagnostic_is_taylor_not_selected():
    _analyses, _selected, diagnostics, _excl = _run()
    assert len(diagnostics) >= 1
    for d in diagnostics:
        assert d["method"] == "taylor"
        assert d["selected_for_ckks"] is False


def test_all_configs_accounted():
    """selected + diagnostics + excluded = 24 (ninguna desaparece)."""
    analyses, selected, diagnostics, excluded = _run()
    total = len(selected) + len(diagnostics) + len(excluded)
    assert total == len(analyses) == 24


def test_excluded_have_reasons():
    _analyses, _selected, _diag, excluded = _run()
    for e in excluded:
        assert e["excluded_reason"]  # no vacío


def test_maximum_accuracy_category_picks_best():
    """La categoría maximum_accuracy debe elegir la config de mayor accuracy por método."""
    analyses, selected, _diag, _excl = _run()
    eligible = [a for a in analyses if a.eligible_for_ckks]
    for method in ("chebyshev", "least_squares"):
        best_acc = max(a.validation_accuracy for a in eligible if a.method == method)
        max_acc_selected = [
            s
            for s in selected
            if s["method"] == method and s["selection_category"] == "maximum_accuracy"
        ]
        assert len(max_acc_selected) == 1
        assert abs(max_acc_selected[0]["validation_accuracy"] - best_acc) < 1e-9


def test_reproducible_selection():
    """Dos ejecuciones producen la misma shortlist."""
    _a1, sel1, _d1, _e1 = _run()
    _a2, sel2, _d2, _e2 = _run()
    ids1 = [s["configuration_id"] for s in sel1]
    ids2 = [s["configuration_id"] for s in sel2]
    assert ids1 == ids2
