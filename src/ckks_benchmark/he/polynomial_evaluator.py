"""
Evaluador homomórfico de polinomios CKKS (Hito 4C).

Evalúa p(x) = a0 + a1·x + ... + ad·x^d sobre un ciphertext, con dos estrategias
en base monomial (sin conversión de base):

    horner       - secuencial, profundidad ~ grado efectivo. Referencia.
    power_basis  - DAG de potencias reutilizadas, profundidad de potencias
                   ~log2(d), pero las ramas deben alinearse antes de sumar,
                   por lo que la profundidad OBSERVADA se mide empíricamente.

Ambas consumen el mismo arreglo de coeficientes congelado del Hito 3A.
El veredicto de factibilidad (grado 7) se basa en la medición real, no en el
DAG teórico.

Paterson-Stockmeyer general queda diferido (sin beneficio claro para d<=7).
"""

from __future__ import annotations

import numpy as np

from ckks_benchmark.he.ciphertext_state import OperationRecorder
from ckks_benchmark.he.operations import (
    add_plain,
    multiply_ciphertexts,
    multiply_plain,
)

STRATEGIES = ("horner", "power_basis")
OFFICIAL_STRATEGIES = ("horner",)
EXPERIMENTAL_STRATEGIES = ("power_basis",)
OFFICIAL_SUPPORTED_DEGREES = (3, 5)
COEFFICIENT_TOLERANCE = 0.0  # solo se omiten coeficientes EXACTAMENTE cero


def _active_exponents(coefficients: np.ndarray) -> list[int]:
    """Exponentes con coeficiente no exactamente cero."""
    return [i for i, c in enumerate(coefficients) if abs(c) > COEFFICIENT_TOLERANCE]


def evaluate_polynomial_ckks(
    he,
    encrypted_x,
    coefficients: np.ndarray,
    *,
    strategy: str,
    recorder: OperationRecorder | None = None,
    allow_experimental: bool = False,
):
    """Evalúa un polinomio cifrado con la estrategia indicada.

    Args:
        he: contexto Pyfhel.
        encrypted_x: ciphertext de la entrada x (nivel 0, fresco).
        coefficients: [a0, a1, ..., ad] en base monomial (float64).
        strategy: 'horner' o 'power_basis'.
        recorder: opcional, para instrumentar niveles/escala.

    Returns:
        ciphertext de p(x).
    """
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if strategy in EXPERIMENTAL_STRATEGIES and not allow_experimental:
        raise ValueError(
            f"La estrategia '{strategy}' es experimental (no adoptada). "
            f"Estrategia oficial: 'horner'. Use allow_experimental=True solo en pilotos."
        )
    if strategy == "horner":
        return _evaluate_horner(he, encrypted_x, coefficients, recorder=recorder)
    if strategy == "power_basis":
        return _evaluate_power_basis(he, encrypted_x, coefficients, recorder=recorder)
    raise ValueError(f"Estrategia desconocida: {strategy}. Use {STRATEGIES}.")


def _evaluate_horner(he, encrypted_x, coefficients, *, recorder=None):
    """Horner: p = a_d; for k=d-1..0: p = p*x + a_k.

    Cada iteración: una mult ciphertext-ciphertext (p*x) + un add_plain (a_k).
    Profundidad ~ grado efectivo.
    """
    slots = he.get_nSlots()
    degree = len(coefficients) - 1

    # acc = a_d (constante cifrada al nivel de x para poder multiplicar).
    acc = he.encryptFrac(np.full(slots, float(coefficients[-1]), dtype=np.float64))
    if recorder is not None:
        recorder.record(acc, "horner_init")

    for k in range(degree - 1, -1, -1):
        # acc = acc * x  (fresco x en cada iter, la operación alinea).
        x_copy = encrypted_x.copy()
        acc = multiply_ciphertexts(he, acc, x_copy, recorder=recorder)
        # acc = acc + a_k
        acc = add_plain(
            he, acc, np.full(slots, float(coefficients[k]), dtype=np.float64), recorder=recorder
        )
    return acc


def _evaluate_power_basis(he, encrypted_x, coefficients, *, recorder=None):
    """Power basis: construye x^2..x^d reutilizando, luego combina términos.

    Las potencias se construyen con un DAG de baja profundidad, pero los
    términos a_k·x^k terminan en niveles distintos y se alinean antes de sumar.
    La profundidad observada se mide con el recorder.
    """
    slots = he.get_nSlots()
    degree = len(coefficients) - 1
    active = _active_exponents(coefficients)

    # Construir las potencias necesarias (hasta el mayor exponente activo).
    max_exp = max(active) if active else 0
    powers = _build_reused_powers(he, encrypted_x, max_exp, recorder=recorder)
    # powers[k] = ciphertext de x^k, para k en 1..max_exp. powers[0] no se usa
    # (término constante se suma como plaintext).

    # Combinar: sum_k a_k · x^k, alineando ramas.
    term_ct = None
    for k in active:
        if k == 0:
            continue  # constante se añade al final como plaintext
        # a_k · x^k  (mult por plaintext, consume nivel + rescale).
        branch = powers[k].copy()
        branch = multiply_plain(
            he, branch, np.full(slots, float(coefficients[k]), dtype=np.float64), recorder=recorder
        )
        if term_ct is None:
            term_ct = branch
        else:
            # Alinear y sumar (add_ciphertexts alinea nivel/escala).
            from ckks_benchmark.he.operations import add_ciphertexts

            term_ct = add_ciphertexts(he, term_ct, branch, recorder=recorder)

    # Añadir el término constante a_0.
    if 0 in active or coefficients[0] != 0.0:
        if term_ct is None:
            term_ct = he.encryptFrac(np.full(slots, float(coefficients[0]), dtype=np.float64))
        else:
            term_ct = add_plain(
                he,
                term_ct,
                np.full(slots, float(coefficients[0]), dtype=np.float64),
                recorder=recorder,
            )
    return term_ct


def _build_reused_powers(he, encrypted_x, max_exp, *, recorder=None):
    """Construye x^1..x^max_exp reutilizando potencias (DAG de baja profundidad).

    Estrategia: x^2 = x·x, luego cada x^k se forma multiplicando dos potencias
    ya calculadas que sumen k, prefiriendo el par más balanceado (menor
    profundidad). Devuelve dict {exponente: ciphertext}.
    """
    powers = {1: encrypted_x.copy()}
    if max_exp < 2:
        return powers

    for k in range(2, max_exp + 1):
        # Elegir a+b=k con a,b ya calculados y |a-b| mínimo (balanceado).
        best = None
        for a in range(1, k):
            b = k - a
            if a in powers and b in powers:
                diff = abs(a - b)
                if best is None or diff < best[0]:
                    best = (diff, a, b)
        if best is None:
            # No debería ocurrir si construimos en orden, pero por seguridad.
            _, a, b = 0, k - 1, 1
        else:
            _, a, b = best
        pa = powers[a].copy()
        pb = powers[b].copy()
        powers[k] = multiply_ciphertexts(he, pa, pb, recorder=recorder)
        if recorder is not None:
            recorder.record(powers[k], f"power_x^{k}")
    return powers
