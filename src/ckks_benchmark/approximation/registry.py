"""
Registro y generación de las aproximaciones polinómicas (Hito 3A).

Consume los intervalos validados del Hito 2 y genera las 72 aproximaciones:
    3 métodos (taylor, chebyshev, least_squares)
    × 4 grados (3, 5, 7, 9)
    × 2 intervalos (I1, I2)
    × 3 activaciones (act1, act2, act3)
    = 72 polinomios

Fuente de los intervalos: results/published/preactivation_intervals.json
(resultado empírico congelado del Hito 2, NO configs/experiment.yaml, que
guarda parámetros a priori). Esto preserva la trazabilidad:
    preactivaciones -> percentiles -> I1/I2 -> registry -> polinomios.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ckks_benchmark.approximation import chebyshev, least_squares, taylor
from ckks_benchmark.approximation.base import PolynomialApproximation

METHODS = {
    "taylor": taylor.fit,
    "chebyshev": chebyshev.fit,
    "least_squares": least_squares.fit,
}
DEGREES = (3, 5, 7, 9)
ACTIVATIONS = ("act1", "act2", "act3")
INTERVAL_NAMES = ("I1", "I2")
EXPECTED_POLYNOMIALS = len(METHODS) * len(DEGREES) * len(ACTIVATIONS) * len(INTERVAL_NAMES)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_validate_intervals(path: Path) -> dict[str, dict[str, tuple[float, float]]]:
    """Carga los intervalos del JSON del Hito 2 y valida su estructura.

    Returns:
        {activation: {interval_name: (a, b)}}
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró {path}. Ejecuta primero el Hito 2 (export_preactivations)."
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    intervals_raw = data.get("intervals")
    if intervals_raw is None:
        raise KeyError("El JSON no contiene la clave 'intervals'.")

    # Reorganizar {I1: {act1: [a,b]}} -> {act1: {I1: (a,b)}} y validar.
    result: dict[str, dict[str, tuple[float, float]]] = {a: {} for a in ACTIVATIONS}
    for interval_name in INTERVAL_NAMES:
        if interval_name not in intervals_raw:
            raise KeyError(f"Falta el intervalo '{interval_name}' en el JSON.")
        for activation in ACTIVATIONS:
            if activation not in intervals_raw[interval_name]:
                raise KeyError(
                    f"Falta la activación '{activation}' en el intervalo '{interval_name}'."
                )
            bounds = intervals_raw[interval_name][activation]
            if len(bounds) != 2:
                raise ValueError(f"El intervalo {interval_name}/{activation} debe tener 2 límites.")
            a, b = float(bounds[0]), float(bounds[1])
            if not (np.isfinite(a) and np.isfinite(b)):
                raise ValueError(f"Límites no finitos en {interval_name}/{activation}.")
            if a >= b:
                raise ValueError(
                    f"El intervalo {interval_name}/{activation} requiere a < b: ({a}, {b})."
                )
            result[activation][interval_name] = (a, b)

    return result


def generate_all(
    intervals: dict[str, dict[str, tuple[float, float]]],
) -> dict[str, PolynomialApproximation]:
    """Genera los 72 polinomios, indexados por su identificador."""
    polynomials: dict[str, PolynomialApproximation] = {}

    for method_name, fit_fn in METHODS.items():
        for degree in DEGREES:
            for activation in ACTIVATIONS:
                for interval_name in INTERVAL_NAMES:
                    interval = intervals[activation][interval_name]
                    poly = fit_fn(
                        degree=degree,
                        interval=interval,
                        activation=activation,
                        interval_name=interval_name,
                    )
                    polynomials[poly.identifier()] = poly

    if len(polynomials) != EXPECTED_POLYNOMIALS:
        raise RuntimeError(
            f"Se esperaban {EXPECTED_POLYNOMIALS} polinomios, "
            f"pero se generaron {len(polynomials)} (¿identificadores duplicados?)."
        )

    return polynomials


def export_polynomials(
    polynomials: dict[str, PolynomialApproximation],
    output_path: Path,
    intervals_source: Path,
) -> None:
    """Exporta los polinomios a JSON con metadatos de trazabilidad."""
    entries = []
    for identifier, poly in sorted(polynomials.items()):
        entries.append(
            {
                "id": identifier,
                "method": poly.method,
                "activation": poly.activation,
                "interval_name": poly.interval_name,
                "interval": list(poly.interval),
                "degree": poly.degree,
                "effective_degree": poly.effective_degree,
                "coefficients": poly.coefficients.tolist(),
                "coefficient_basis": poly.coefficient_basis,
                "max_abs_coefficient": poly.max_abs_coefficient,
                "metadata": _json_safe(poly.metadata),
            }
        )

    payload = {
        "intervals_source": str(intervals_source).replace("\\", "/"),
        "intervals_sha256": _sha256_of_file(intervals_source),
        "methods": list(METHODS.keys()),
        "degrees": list(DEGREES),
        "activations": list(ACTIVATIONS),
        "interval_names": list(INTERVAL_NAMES),
        "expected_polynomials": EXPECTED_POLYNOMIALS,
        "generated_polynomials": len(polynomials),
        "polynomials": entries,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _json_safe(metadata: dict[str, Any]) -> dict[str, Any]:
    """Convierte valores no serializables (numpy) a tipos nativos."""
    safe = {}
    for key, value in metadata.items():
        if isinstance(value, (np.floating, np.integer)):
            safe[key] = float(value)
        elif isinstance(value, np.ndarray):
            safe[key] = value.tolist()
        else:
            safe[key] = value
    return safe


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Genera las 72 aproximaciones polinómicas del Hito 3A."
    )
    parser.add_argument(
        "--intervals",
        type=str,
        default="results/published/preactivation_intervals.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/approximations/coefficients/polynomials.json",
    )
    args = parser.parse_args()

    intervals_path = Path(args.intervals)
    output_path = Path(args.output)

    intervals = load_and_validate_intervals(intervals_path)
    polynomials = generate_all(intervals)
    export_polynomials(polynomials, output_path, intervals_path)

    # Resumen.
    print("=" * 72)
    print("GENERACIÓN DE APROXIMACIONES — Hito 3A")
    print("=" * 72)
    print(f"Fuente de intervalos: {intervals_path}")
    print(f"Polinomios generados: {len(polynomials)} / {EXPECTED_POLYNOMIALS}")
    print(f"Salida:               {output_path}")
    print("-" * 72)

    # Muestra de diagnóstico: grados efectivos y C_max por método (act1-I1, grado 5).
    print("Muestra (act1, I1, grado 5):")
    for method in METHODS:
        pid = f"act1_{method}_d5_I1"
        p = polynomials[pid]
        print(f"  {method:>14}: grado_ef={p.effective_degree} C_max={p.max_abs_coefficient:.4f}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
