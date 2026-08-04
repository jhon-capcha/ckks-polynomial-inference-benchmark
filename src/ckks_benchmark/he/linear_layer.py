"""
Capa lineal cifrada SIMD (Hito 4D).

Implementa fc2: R^120 -> R^10 bajo CKKS. Cada logit y_j = w_j^T x + b_j se
calcula como:
    1. multiply_plain: x (cifrado) por la fila de pesos w_j (plaintext SIMD)
    2. sum_slots: reducir los primeros slots para obtener el producto punto
    3. add_plain: sumar el bias b_j

Empaquetado: un ciphertext de entrada (120 slots activos, padding a 128).
Salida: un ciphertext por logit (10 ciphertexts), con el resultado en el slot 0.
La reducción de rotaciones a un solo ciphertext de salida se difiere al Hito 5.

Pesos y bias en claro (modelo de amenaza: el servidor los tiene en plaintext).
"""

from __future__ import annotations

import numpy as np

from ckks_benchmark.he.ciphertext_state import OperationRecorder
from ckks_benchmark.he.simd_operations import next_power_of_two, sum_slots_power_of_two


def encrypted_dot_product(
    he,
    encrypted_vector,
    weights: np.ndarray,
    *,
    logical_size: int,
    bias: float = 0.0,
    recorder: OperationRecorder | None = None,
):
    """Producto punto cifrado: w^T x + b.

    Args:
        encrypted_vector: ciphertext con x en los primeros logical_size slots.
        weights: vector de pesos (longitud logical_size), en claro.
        logical_size: número de valores útiles.
        bias: sesgo escalar en claro.

    Returns:
        ciphertext con el resultado en el slot 0.
    """
    slots = he.get_nSlots()
    reduction_size = next_power_of_two(logical_size)

    # 1. Empaquetar pesos como plaintext SIMD (padding a slots con ceros).
    w_packed = np.zeros(slots, dtype=np.float64)
    w_packed[:logical_size] = weights[:logical_size]
    w_ptxt = he.encodeFrac(w_packed)

    # 2. multiply_plain: x * w (alineado), luego rescale.
    ct = encrypted_vector.copy()
    ct_a, w_a = he.align_mod_n_scale(ct, w_ptxt, copy_this=False, copy_other=True)
    he.multiply_plain(ct_a, w_a)
    he.rescale_to_next(ct_a)
    if recorder is not None:
        recorder.record(ct_a, "fc_multiply_plain+rescale")

    # 3. Reducir slots (suma) -> producto punto en slot 0.
    ct_sum = sum_slots_power_of_two(he, ct_a, reduction_size=reduction_size, recorder=recorder)

    # 4. Sumar bias (plaintext alineado).
    if bias != 0.0:
        b_packed = np.zeros(slots, dtype=np.float64)
        b_packed[0] = bias  # el resultado está en el slot 0
        b_ptxt = he.encodeFrac(b_packed)
        ct_sum_a, b_a = he.align_mod_n_scale(ct_sum, b_ptxt, copy_this=False, copy_other=True)
        he.add_plain(ct_sum_a, b_a)
        ct_sum = ct_sum_a
        if recorder is not None:
            recorder.record(ct_sum, "fc_add_bias")

    return ct_sum


def encrypted_linear_simd(
    he,
    encrypted_input,
    weight_matrix: np.ndarray,
    bias_vector: np.ndarray,
    *,
    logical_size: int,
    recorder: OperationRecorder | None = None,
) -> list:
    """Capa lineal completa: produce un ciphertext por fila (logit).

    Args:
        encrypted_input: ciphertext con la entrada (logical_size slots activos).
        weight_matrix: matriz de pesos (n_out, logical_size), en claro.
        bias_vector: vector de sesgos (n_out,), en claro.

    Returns:
        lista de n_out ciphertexts, cada uno con su logit en el slot 0.
    """
    n_out = weight_matrix.shape[0]
    outputs = []
    for j in range(n_out):
        ct_j = encrypted_dot_product(
            he,
            encrypted_input,
            weight_matrix[j],
            logical_size=logical_size,
            bias=float(bias_vector[j]),
            recorder=recorder,
        )
        outputs.append(ct_j)
    return outputs


def decrypt_logits(he, logit_ciphertexts: list) -> np.ndarray:
    """Descifra una lista de ciphertexts de logits (lee el slot 0 de cada uno)."""
    return np.array([he.decryptFrac(ct)[0] for ct in logit_ciphertexts])
