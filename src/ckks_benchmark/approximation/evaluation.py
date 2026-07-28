"""
Métricas de evaluación funcional de las aproximaciones polinómicas (Hito 3B).

Cuantifica el error de un polinomio frente a ReLU, sin CKKS y sin integrar en
la CNN. Calcula error uniforme (malla) y empírico (preactivaciones reales de
validación), error dentro/fuera del intervalo, error cerca del origen, y
diagnósticos de estabilidad numérica.

Jerarquía de métricas (criterio principal = empírico):
    1. MAE empírico   2. RMSE empírico   3. percentiles empíricos
    4. error uniforme  5. error máximo

Invalidez: SOLO por no-finitud (NaN/inf/overflow/forma incompatible/vacío).
Un error alto pero finito es un resultado válido (p.ej. Taylor grado 9). La
"viabilidad práctica" se registra en columna aparte, sin excluir la config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Parámetros congelados de evaluación.
UNIFORM_GRID_POINTS = 10_000
ORIGIN_HALF_WIDTH = 0.5
ORIGIN_GRID_POINTS = 2_001
EVALUATION_DTYPE = np.float64

BIAS_RATIO_EPSILON = 1e-12
# Umbral operativo provisional de viabilidad práctica (NO límite CKKS real).
PRACTICAL_OUTPUT_THRESHOLD = 1e6


@dataclass(frozen=True)
class ApproximationMetrics:
    """Métricas completas de una configuración polinómica (una de las 72)."""

    # Identificación
    configuration_id: str
    activation: str
    method: str
    degree: int
    effective_degree: int
    interval_name: str
    lower_bound: float
    upper_bound: float

    # Uniforme sobre el intervalo completo
    mae_uniform: float
    rmse_uniform: float
    max_error_uniform: float
    x_at_max_error_uniform: float

    # Empírico sobre validación
    mae_empirical: float
    rmse_empirical: float
    median_abs_error_empirical: float
    p95_abs_error_empirical: float
    p99_abs_error_empirical: float
    max_error_empirical: float

    # Dentro y fuera del intervalo
    fraction_outside_interval: float
    mae_inside_interval: float
    mae_outside_interval: float
    outside_error_contribution: float

    # Origen uniforme (principal)
    mae_origin_uniform: float
    rmse_origin_uniform: float
    max_error_origin_uniform: float

    # Origen empírico (complementario)
    origin_empirical_sample_count: int
    origin_empirical_fraction: float
    mae_origin_empirical: float
    rmse_origin_empirical: float
    max_error_origin_empirical: float

    # Diagnóstico
    bias_ratio: float
    bias_ratio_denominator_clamped: bool
    max_abs_output_uniform: float
    max_abs_output_empirical: float

    # Estado
    valid: bool
    invalid_reason: str | None
    practically_viable: bool
    practical_viability_reason: str | None


# --- Métricas escalares básicas ---
def _check_arrays(expected: np.ndarray, observed: np.ndarray) -> None:
    if expected.size == 0 or observed.size == 0:
        raise ValueError("Arreglo vacío en el cálculo de error.")
    if expected.shape != observed.shape:
        raise ValueError(f"Formas incompatibles: {expected.shape} vs {observed.shape}.")


def mean_absolute_error(expected: np.ndarray, observed: np.ndarray) -> float:
    _check_arrays(expected, observed)
    return float(np.mean(np.abs(expected - observed)))


def root_mean_squared_error(expected: np.ndarray, observed: np.ndarray) -> float:
    _check_arrays(expected, observed)
    return float(np.sqrt(np.mean((expected - observed) ** 2)))


def max_error_with_location(
    expected: np.ndarray, observed: np.ndarray, x: np.ndarray
) -> tuple[float, float]:
    """Error máximo absoluto y la posición x donde ocurre."""
    _check_arrays(expected, observed)
    abs_err = np.abs(expected - observed)
    idx = int(np.argmax(abs_err))
    return float(abs_err[idx]), float(x[idx])


def relu_reference(x: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(x, dtype=EVALUATION_DTYPE), 0.0)


# --- Evaluación completa de una configuración ---
def evaluate_configuration(
    approximation,
    samples: np.ndarray,
) -> ApproximationMetrics:
    """Calcula todas las métricas de un polinomio contra ReLU.

    Args:
        approximation: PolynomialApproximation (del registry del Hito 3A).
        samples: preactivaciones reales de validación de la MISMA activación.

    Returns:
        ApproximationMetrics con todas las métricas y el estado de validez.
    """
    a, b = approximation.interval
    config_id = approximation.identifier()

    # Verificaciones de no-finitud tempranas (invalidez estructural).
    try:
        samples = np.asarray(samples, dtype=EVALUATION_DTYPE)
        if samples.size == 0:
            raise ValueError("muestra empírica vacía")

        # --- Malla uniforme ---
        x_uniform = np.linspace(a, b, UNIFORM_GRID_POINTS, dtype=EVALUATION_DTYPE)
        y_ref_u = relu_reference(x_uniform)
        y_poly_u = approximation.evaluate(x_uniform)

        # --- Empírico ---
        y_ref_e = relu_reference(samples)
        y_poly_e = approximation.evaluate(samples)

        # Detección de no-finitud en las salidas del polinomio.
        if not (np.all(np.isfinite(y_poly_u)) and np.all(np.isfinite(y_poly_e))):
            return _invalid_metrics(approximation, "salida no finita (NaN/inf) del polinomio")

    except (ValueError, FloatingPointError, OverflowError) as exc:
        return _invalid_metrics(approximation, f"error de evaluación: {exc}")

    # --- Métricas uniformes ---
    mae_u = mean_absolute_error(y_ref_u, y_poly_u)
    rmse_u = root_mean_squared_error(y_ref_u, y_poly_u)
    emax_u, x_emax_u = max_error_with_location(y_ref_u, y_poly_u, x_uniform)

    # --- Métricas empíricas ---
    abs_err_e = np.abs(y_ref_e - y_poly_e)
    mae_e = float(np.mean(abs_err_e))
    rmse_e = float(np.sqrt(np.mean(abs_err_e**2)))
    median_e = float(np.median(abs_err_e))
    p95_e = float(np.percentile(abs_err_e, 95))
    p99_e = float(np.percentile(abs_err_e, 99))
    emax_e = float(np.max(abs_err_e))

    # --- Dentro / fuera del intervalo ---
    inside = (samples >= a) & (samples <= b)
    outside = ~inside
    n_out = int(np.count_nonzero(outside))
    frac_out = n_out / samples.size
    mae_in = float(np.mean(abs_err_e[inside])) if np.any(inside) else float("nan")
    mae_out = float(np.mean(abs_err_e[outside])) if n_out > 0 else float("nan")
    total_err = float(np.sum(abs_err_e))
    out_contrib = float(np.sum(abs_err_e[outside])) / total_err if total_err > 0 else 0.0

    # --- Origen uniforme (principal) ---
    x_origin = np.linspace(
        -ORIGIN_HALF_WIDTH, ORIGIN_HALF_WIDTH, ORIGIN_GRID_POINTS, dtype=EVALUATION_DTYPE
    )
    y_ref_o = relu_reference(x_origin)
    y_poly_o = approximation.evaluate(x_origin)
    mae_o_u = mean_absolute_error(y_ref_o, y_poly_o)
    rmse_o_u = root_mean_squared_error(y_ref_o, y_poly_o)
    emax_o_u, _ = max_error_with_location(y_ref_o, y_poly_o, x_origin)

    # --- Origen empírico (complementario) ---
    origin_mask = np.abs(samples) <= ORIGIN_HALF_WIDTH
    n_origin = int(np.count_nonzero(origin_mask))
    origin_frac = n_origin / samples.size
    if n_origin > 0:
        mae_o_e = float(np.mean(abs_err_e[origin_mask]))
        rmse_o_e = float(np.sqrt(np.mean(abs_err_e[origin_mask] ** 2)))
        emax_o_e = float(np.max(abs_err_e[origin_mask]))
    else:
        mae_o_e = rmse_o_e = emax_o_e = float("nan")

    # --- Diagnóstico ---
    clamped = mae_e < BIAS_RATIO_EPSILON
    bias_ratio = mae_u / max(mae_e, BIAS_RATIO_EPSILON)
    max_out_u = float(np.max(np.abs(y_poly_u)))
    max_out_e = float(np.max(np.abs(y_poly_e)))

    # --- Viabilidad práctica (columna aparte, umbral operativo provisional) ---
    if max_out_u <= PRACTICAL_OUTPUT_THRESHOLD and max_out_e <= PRACTICAL_OUTPUT_THRESHOLD:
        viable, viability_reason = True, None
    else:
        viable = False
        viability_reason = (
            f"salida máxima {max(max_out_u, max_out_e):.2e} > "
            f"umbral operativo {PRACTICAL_OUTPUT_THRESHOLD:.0e}"
        )

    return ApproximationMetrics(
        configuration_id=config_id,
        activation=approximation.activation,
        method=approximation.method,
        degree=approximation.degree,
        effective_degree=approximation.effective_degree,
        interval_name=approximation.interval_name,
        lower_bound=float(a),
        upper_bound=float(b),
        mae_uniform=mae_u,
        rmse_uniform=rmse_u,
        max_error_uniform=emax_u,
        x_at_max_error_uniform=x_emax_u,
        mae_empirical=mae_e,
        rmse_empirical=rmse_e,
        median_abs_error_empirical=median_e,
        p95_abs_error_empirical=p95_e,
        p99_abs_error_empirical=p99_e,
        max_error_empirical=emax_e,
        fraction_outside_interval=frac_out,
        mae_inside_interval=mae_in,
        mae_outside_interval=mae_out,
        outside_error_contribution=out_contrib,
        mae_origin_uniform=mae_o_u,
        rmse_origin_uniform=rmse_o_u,
        max_error_origin_uniform=emax_o_u,
        origin_empirical_sample_count=n_origin,
        origin_empirical_fraction=origin_frac,
        mae_origin_empirical=mae_o_e,
        rmse_origin_empirical=rmse_o_e,
        max_error_origin_empirical=emax_o_e,
        bias_ratio=bias_ratio,
        bias_ratio_denominator_clamped=clamped,
        max_abs_output_uniform=max_out_u,
        max_abs_output_empirical=max_out_e,
        valid=True,
        invalid_reason=None,
        practically_viable=viable,
        practical_viability_reason=viability_reason,
    )


def _invalid_metrics(approximation, reason: str) -> ApproximationMetrics:
    """Construye un resultado inválido (config numéricamente no evaluable)."""
    a, b = approximation.interval
    nan = float("nan")
    return ApproximationMetrics(
        configuration_id=approximation.identifier(),
        activation=approximation.activation,
        method=approximation.method,
        degree=approximation.degree,
        effective_degree=approximation.effective_degree,
        interval_name=approximation.interval_name,
        lower_bound=float(a),
        upper_bound=float(b),
        mae_uniform=nan,
        rmse_uniform=nan,
        max_error_uniform=nan,
        x_at_max_error_uniform=nan,
        mae_empirical=nan,
        rmse_empirical=nan,
        median_abs_error_empirical=nan,
        p95_abs_error_empirical=nan,
        p99_abs_error_empirical=nan,
        max_error_empirical=nan,
        fraction_outside_interval=nan,
        mae_inside_interval=nan,
        mae_outside_interval=nan,
        outside_error_contribution=nan,
        mae_origin_uniform=nan,
        rmse_origin_uniform=nan,
        max_error_origin_uniform=nan,
        origin_empirical_sample_count=0,
        origin_empirical_fraction=nan,
        mae_origin_empirical=nan,
        rmse_origin_empirical=nan,
        max_error_origin_empirical=nan,
        bias_ratio=nan,
        bias_ratio_denominator_clamped=False,
        max_abs_output_uniform=nan,
        max_abs_output_empirical=nan,
        valid=False,
        invalid_reason=reason,
        practically_viable=False,
        practical_viability_reason="configuración inválida",
    )
