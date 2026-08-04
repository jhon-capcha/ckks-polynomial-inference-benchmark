"""
Operaciones SIMD para el bloque lineal cifrado (Hito 4D).

Reducción de slots (suma) mediante rotate+add en árbol, y utilidades de
empaquetado. La suma de N slots deja el total en el slot 0.

Convención de rotación (verificada en el entorno): rotate(+shift) trae el valor
del slot i+shift al slot i (desplazamiento a la izquierda). Por eso rotamos con
desplazamientos positivos potencia de 2 y sumamos: el slot 0 acumula el total.

Precaución de máscara: los slots fuera de la región lógica DEBEN ser cero antes
de reducir, o contaminarían la suma. Se reduce sobre reduction_size (potencia
de 2 >= logical_size), con los slots de padding en cero.
"""

from __future__ import annotations

import numpy as np

from ckks_benchmark.he.ciphertext_state import OperationRecorder


def next_power_of_two(n: int) -> int:
    """Menor potencia de 2 >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


def pad_to_slots(values: np.ndarray, slots: int) -> np.ndarray:
    """Coloca `values` en los primeros slots, resto en cero."""
    out = np.zeros(slots, dtype=np.float64)
    out[: len(values)] = values
    return out


def sum_slots_power_of_two(
    he, ciphertext, *, reduction_size: int, recorder: OperationRecorder | None = None
):
    """Suma los primeros `reduction_size` slots mediante rotate+add en árbol.

    reduction_size debe ser potencia de 2. Los slots [logical_size, reduction_size)
    deben ser cero (padding). Tras la reducción, el slot 0 contiene la suma de
    los primeros reduction_size slots.

    No consume niveles (rotaciones y sumas no rescatan).
    """
    if reduction_size & (reduction_size - 1) != 0:
        raise ValueError(f"reduction_size={reduction_size} debe ser potencia de 2.")

    acc = ciphertext.copy()
    shift = 1
    while shift < reduction_size:
        rotated = acc.copy()
        he.rotate(rotated, shift)
        he.add(acc, rotated)
        if recorder is not None:
            recorder.record(acc, f"rotate{shift}+add")
        shift *= 2
    return acc
