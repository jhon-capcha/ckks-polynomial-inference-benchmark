"""
Contexto CKKS del Hito 4 (Pyfhel 3.5.0).

Crea y valida un contexto CKKS a partir de un perfil oficial, generando las
claves y verificando que la seguridad realmente alcanzada sea >= 128 bits
(el piloto mostró que cadenas largas producen sec=0 silenciosamente).

Separa las claves de cliente (secreta) y servidor (pública, relin, rotación):
el servidor NUNCA debe recibir la clave secreta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ckks_benchmark.he.parameters import (
    MINIMUM_SECURITY_BITS,
    CKKSParameterProfile,
    get_profile,
)


@dataclass
class CKKSContext:
    """Envuelve un contexto Pyfhel CKKS validado, con su perfil."""

    profile: CKKSParameterProfile
    he: object  # Pyfhel instance
    security_verified: int

    @property
    def slot_count(self) -> int:
        return self.he.get_nSlots()

    def encrypt(self, values: np.ndarray):
        """Cifra un vector float64 (rellena/trunca a slot_count)."""
        arr = np.asarray(values, dtype=np.float64)
        return self.he.encryptFrac(_fit_to_slots(arr, self.slot_count))

    def decrypt(self, ciphertext, n: int | None = None) -> np.ndarray:
        """Descifra a un vector float64 (opcionalmente los primeros n valores)."""
        dec = self.he.decryptFrac(ciphertext)
        return dec[:n] if n is not None else dec

    def encode(self, values: np.ndarray):
        arr = np.asarray(values, dtype=np.float64)
        return self.he.encodeFrac(_fit_to_slots(arr, self.slot_count))


def _fit_to_slots(arr: np.ndarray, slots: int) -> np.ndarray:
    """Ajusta un array a exactamente `slots` elementos (rellena con ceros o trunca)."""
    if arr.size == slots:
        return arr
    if arr.size > slots:
        return arr[:slots]
    out = np.zeros(slots, dtype=np.float64)
    out[: arr.size] = arr
    return out


def create_context(
    profile_id: str,
    *,
    generate_rotation_keys: bool = False,
) -> CKKSContext:
    """Crea un contexto CKKS validado desde un perfil oficial.

    Genera clave secreta, pública y de relinearización. Las de rotación son
    opcionales (costosas; solo si se necesitan sumas de slots / convolución).
    Verifica que la seguridad alcanzada sea >= 128 bits.
    """
    from Pyfhel import Pyfhel

    profile = get_profile(profile_id)  # valida el perfil (incl. seguridad declarada)
    profile.validate()

    he = Pyfhel()
    he.contextGen(
        scheme="ckks",
        n=profile.poly_modulus_degree,
        scale=2**profile.scale_bits,
        qi_sizes=list(profile.coeff_mod_bit_sizes),
    )

    if he.is_context_empty():
        raise RuntimeError(f"El contexto del perfil {profile_id} quedó vacío.")

    # Verificación de seguridad REAL (no la declarada).
    try:
        actual_security = int(he.sec)
    except Exception:
        actual_security = -1
    if actual_security < MINIMUM_SECURITY_BITS:
        raise RuntimeError(
            f"Perfil {profile_id}: seguridad real {actual_security} < "
            f"{MINIMUM_SECURITY_BITS} bits. Contexto rechazado."
        )

    # Generación de claves.
    he.keyGen()  # secreta + pública
    he.relinKeyGen()  # relinearización (para mult ciphertext-ciphertext)
    if generate_rotation_keys:
        he.rotateKeyGen()

    return CKKSContext(profile=profile, he=he, security_verified=actual_security)


def split_client_server(context: CKKSContext) -> tuple[bytes, dict]:
    """Separa material de cliente (secreta) y servidor (pública/relin/rotación).

    Devuelve (secret_key_bytes, server_material). El servidor recibe SOLO
    server_material: NUNCA la clave secreta.
    """
    he = context.he
    secret_key = he.to_bytes_secret_key()
    server_material = {
        "context": he.to_bytes_context(),
        "public_key": he.to_bytes_public_key(),
        "relin_key": he.to_bytes_relin_key(),
    }
    if not he.is_rotate_key_empty():
        server_material["rotate_key"] = he.to_bytes_rotate_key()
    return secret_key, server_material


def load_server_context(server_material: dict) -> object:
    """Reconstruye un contexto de SERVIDOR (sin clave secreta) desde bytes."""
    from Pyfhel import Pyfhel

    he = Pyfhel()
    he.from_bytes_context(server_material["context"])
    he.from_bytes_public_key(server_material["public_key"])
    he.from_bytes_relin_key(server_material["relin_key"])
    if "rotate_key" in server_material:
        he.from_bytes_rotate_key(server_material["rotate_key"])
    return he
