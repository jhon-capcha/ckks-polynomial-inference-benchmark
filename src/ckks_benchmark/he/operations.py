"""
Operaciones primitivas CKKS instrumentadas (Hito 4B).

Envuelve las operaciones de Pyfhel registrando el estado (nivel, escala, tamaño)
antes y después, y validando precondiciones ANTES de llamar a SEAL. Esto evita
los access violations de C++ que ocurren al operar sobre ciphertexts
incompatibles (distinto nivel o escala).

Semántica de nivel en Pyfhel: mod_level cuenta niveles CONSUMIDOS (0 = fresco,
sube con cada rescale). El límite es la profundidad del perfil.

Regla: la alineación de nivel/escala se hace explícita con align_mod_n_scale;
nunca se opera sobre operandos desalineados.
"""

from __future__ import annotations

import numpy as np

from ckks_benchmark.he.ciphertext_state import OperationRecorder


class CKKSOperationError(RuntimeError):
    """Error de precondición en una operación CKKS (antes de llamar a SEAL)."""


def _mod_level(ctxt) -> int | None:
    try:
        return ctxt.mod_level
    except Exception:
        return None


def _scale_bits(ctxt) -> float | None:
    try:
        return float(ctxt.scale_bits)
    except Exception:
        return None


def encrypt_vector(he, values: np.ndarray, recorder: OperationRecorder | None = None):
    """Cifra un vector float64."""
    arr = np.asarray(values, dtype=np.float64)
    ctxt = he.encryptFrac(arr)
    if recorder is not None:
        recorder.record(ctxt, "encrypt")
    return ctxt


def decrypt_vector(he, ctxt, n: int | None = None) -> np.ndarray:
    """Descifra a float64 (opcionalmente los primeros n valores)."""
    dec = he.decryptFrac(ctxt)
    return dec[:n] if n is not None else dec


def multiply_plain(
    he,
    ctxt,
    plain_values: np.ndarray,
    *,
    recorder: OperationRecorder | None = None,
    rescale: bool = True,
):
    """Multiplica un ciphertext por un vector plaintext (pesos/coeficientes en claro).

    ciphertext-plaintext: no crece el tamaño, pero sí la escala (se acumula).
    Se hace rescale para volver a la escala base y consumir un nivel.
    """
    arr = np.asarray(plain_values, dtype=np.float64)
    ptxt = he.encodeFrac(arr)
    # Alinear el plaintext al nivel del ciphertext antes de multiplicar.
    ctxt_a, ptxt_a = he.align_mod_n_scale(ctxt, ptxt, copy_this=False, copy_other=True)
    he.multiply_plain(ctxt_a, ptxt_a)
    if rescale:
        he.rescale_to_next(ctxt_a)
    if recorder is not None:
        recorder.record(ctxt_a, "multiply_plain" + ("+rescale" if rescale else ""))
    return ctxt_a


def multiply_ciphertexts(
    he,
    ct_a,
    ct_b,
    *,
    recorder: OperationRecorder | None = None,
    relinearize: bool = True,
    rescale: bool = True,
):
    """Multiplica dos ciphertexts, con relinearización y rescale.

    Valida que ambos estén alineados (mismo nivel/escala) ANTES de multiplicar,
    para evitar el access violation de SEAL. Si no lo están, los alinea.
    """
    la, lb = _mod_level(ct_a), _mod_level(ct_b)
    sa, sb = _scale_bits(ct_a), _scale_bits(ct_b)
    if la != lb or (sa is not None and sb is not None and abs(sa - sb) > 0.5):
        # Alinear antes de operar (evita segfault).
        ct_a, ct_b = he.align_mod_n_scale(ct_a, ct_b, copy_this=False, copy_other=False)

    he.multiply(ct_a, ct_b)
    if relinearize:
        he.relinearize(ct_a)
    if rescale:
        he.rescale_to_next(ct_a)
    if recorder is not None:
        name = "multiply_ct"
        if relinearize:
            name += "+relin"
        if rescale:
            name += "+rescale"
        recorder.record(ct_a, name)
    return ct_a


def square_ciphertext(
    he,
    ctxt,
    *,
    recorder: OperationRecorder | None = None,
    relinearize: bool = True,
    rescale: bool = True,
):
    """Eleva al cuadrado un ciphertext (x²), con relin y rescale."""
    he.square(ctxt)
    if relinearize:
        he.relinearize(ctxt)
    if rescale:
        he.rescale_to_next(ctxt)
    if recorder is not None:
        recorder.record(ctxt, "square+relin+rescale")
    return ctxt


def add_ciphertexts(he, ct_a, ct_b, *, recorder: OperationRecorder | None = None):
    """Suma dos ciphertexts, alineando nivel/escala antes."""
    la, lb = _mod_level(ct_a), _mod_level(ct_b)
    sa, sb = _scale_bits(ct_a), _scale_bits(ct_b)
    if la != lb or (sa is not None and sb is not None and abs(sa - sb) > 0.5):
        ct_a, ct_b = he.align_mod_n_scale(ct_a, ct_b, copy_this=False, copy_other=False)
    he.add(ct_a, ct_b)
    if recorder is not None:
        recorder.record(ct_a, "add_ct")
    return ct_a


def add_plain(he, ctxt, plain_values: np.ndarray, *, recorder: OperationRecorder | None = None):
    """Suma un vector plaintext a un ciphertext (bias en claro)."""
    arr = np.asarray(plain_values, dtype=np.float64)
    ptxt = he.encodeFrac(arr)
    ctxt_a, ptxt_a = he.align_mod_n_scale(ctxt, ptxt, copy_this=False, copy_other=True)
    he.add_plain(ctxt_a, ptxt_a)
    if recorder is not None:
        recorder.record(ctxt_a, "add_plain")
    return ctxt_a


def align_for_operation(he, ct_a, ct_b):
    """Alinea explícitamente dos operandos (nivel y escala). Devuelve la tupla."""
    return he.align_mod_n_scale(ct_a, ct_b, copy_this=True, copy_other=True)
