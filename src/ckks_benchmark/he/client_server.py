"""
Cliente y servidor CKKS para el bloque final (Hito 4F).

Formaliza el modelo de amenaza:
  - Cliente: tiene la clave secreta. Cifra z (preactivación de act3) y descifra
    los logits. NUNCA envía la clave secreta.
  - Servidor: tiene solo claves públicas (pública, relin, rotación) y los pesos
    en claro. Evalúa act3 -> fc2 bajo cifrado. NUNCA tiene la clave secreta.

El servidor no puede descifrar nada: opera enteramente sobre ciphertexts.
"""

from __future__ import annotations

import numpy as np

from ckks_benchmark.he.ciphertext_state import OperationRecorder
from ckks_benchmark.he.linear_layer import decrypt_logits, encrypted_linear_simd
from ckks_benchmark.he.polynomial_evaluator import evaluate_polynomial_ckks
from ckks_benchmark.he.simd_operations import pad_to_slots


class CKKSClient:
    """Cliente: posee la clave secreta. Cifra entradas, descifra salidas."""

    def __init__(self, context):
        self.ctx = context
        self.he = context.he

    def encrypt_input(self, z: np.ndarray):
        """Cifra la preactivación z (vector de 120) en un ciphertext SIMD."""
        packed = pad_to_slots(np.asarray(z, dtype=np.float64), self.he.get_nSlots())
        return self.he.encryptFrac(packed)

    def decrypt_logits(self, logit_ciphertexts: list) -> np.ndarray:
        """Descifra los 10 logits (slot 0 de cada ciphertext)."""
        return decrypt_logits(self.he, logit_ciphertexts)


class CKKSServer:
    """Servidor: sin clave secreta. Evalúa act3 -> fc2 sobre ciphertexts."""

    def __init__(
        self,
        he_server,
        act3_coefficients: np.ndarray,
        fc2_weights: np.ndarray,
        fc2_bias: np.ndarray,
        logical_size: int = 120,
    ):
        self.he = he_server
        self.act3_coeffs = np.asarray(act3_coefficients, dtype=np.float64)
        self.W = np.asarray(fc2_weights, dtype=np.float64)
        self.b = np.asarray(fc2_bias, dtype=np.float64)
        self.logical_size = logical_size
        # Salvaguarda del modelo de amenaza: el servidor no tiene clave secreta.
        assert self.he.is_secret_key_empty(), "El servidor NO debe tener clave secreta."

    def infer(self, encrypted_z, recorder: OperationRecorder | None = None) -> list:
        """Evalúa el bloque final cifrado: act3 -> fc2. Devuelve 10 ciphertexts."""
        # act3 polinómica (Horner) sobre los 120 valores empaquetados (SIMD).
        act3_ct = evaluate_polynomial_ckks(
            self.he, encrypted_z, self.act3_coeffs, strategy="horner", recorder=recorder
        )
        # fc2: producto matriz-vector cifrado -> 10 logits.
        logit_cts = encrypted_linear_simd(
            self.he, act3_ct, self.W, self.b, logical_size=self.logical_size, recorder=recorder
        )
        return logit_cts
