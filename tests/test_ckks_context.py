"""
Pruebas del contexto CKKS (Hito 4A).
"""

from __future__ import annotations

import numpy as np
import pytest

from ckks_benchmark.he.ckks_context import (
    create_context,
    load_server_context,
    split_client_server,
)
from ckks_benchmark.he.parameters import (
    CKKSParameterProfile,
    get_profile,
    profile_for_degree,
)


# --- Perfiles ---
def test_official_profiles_valid():
    for pid in ("ckks_n16384_d3", "ckks_n16384_d5"):
        p = get_profile(pid)
        p.validate()  # no debe lanzar
        assert p.security_bits >= 128


def test_profile_depth_computation():
    p = get_profile("ckks_n16384_d5")
    # cadena (60,40,40,40,40,40,40,60): 6 primos intermedios.
    assert p.multiplicative_depth == 6
    assert p.total_coeff_modulus_bits == 360


def test_profile_rejects_insecure():
    """Un perfil con cadena excesiva debe fallar la validación."""
    insecure = CKKSParameterProfile(
        profile_id="insecure_test",
        poly_modulus_degree=16384,
        coeff_mod_bit_sizes=(60,) + (40,) * 8 + (60,),  # 440 bits
        scale_bits=40,
        security_bits=128,
    )
    with pytest.raises(ValueError):
        insecure.validate()


def test_profile_for_degree():
    assert profile_for_degree(3).profile_id == "ckks_n16384_d3"
    assert profile_for_degree(5).profile_id == "ckks_n16384_d5"
    with pytest.raises(ValueError):
        profile_for_degree(7)  # grado 7 no tiene perfil oficial


def test_unknown_profile_rejected():
    with pytest.raises(KeyError):
        get_profile("nonexistent_profile")


# --- Contexto (requieren Pyfhel; más lentos) ---
def test_context_creation_and_security():
    ctx = create_context("ckks_n16384_d5")
    assert ctx.security_verified >= 128
    assert ctx.slot_count == 8192


def test_encrypt_decrypt_roundtrip():
    ctx = create_context("ckks_n16384_d3")
    x = np.array([1.5, -2.0, 0.3, 0.0, -0.001])
    ct = ctx.encrypt(x)
    dec = ctx.decrypt(ct, n=len(x))
    assert np.allclose(dec, x, atol=1e-4)


def test_client_server_split_no_secret():
    """El material del servidor NO debe contener la clave secreta."""
    ctx = create_context("ckks_n16384_d5")
    _secret, server_material = split_client_server(ctx)
    he_server = load_server_context(server_material)
    assert he_server.is_secret_key_empty()
    assert not he_server.is_public_key_empty()
    assert not he_server.is_relin_key_empty()


def test_server_context_lacks_decryption_capability():
    """El servidor no tiene clave secreta, por lo que no puede descifrar.

    Verificamos la garantía por el estado de las claves, no intentando descifrar:
    llamar decryptFrac sin clave secreta causa un fallo a nivel C++ (no una
    excepción Python capturable), así que comprobamos is_secret_key_empty().
    """
    ctx = create_context("ckks_n16384_d3")
    _secret, server_material = split_client_server(ctx)
    he_server = load_server_context(server_material)

    # La garantía del modelo de amenaza: sin clave secreta, no hay descifrado posible.
    assert he_server.is_secret_key_empty()
    # El servidor sí tiene lo necesario para computar (pública, relin).
    assert not he_server.is_public_key_empty()
    assert not he_server.is_relin_key_empty()


def test_slot_fitting():
    """Vectores más pequeños que slots se rellenan; más grandes se truncan."""
    ctx = create_context("ckks_n16384_d3")
    small = np.array([1.0, 2.0])
    ct = ctx.encrypt(small)
    dec = ctx.decrypt(ct, n=4)
    assert np.allclose(dec[:2], [1.0, 2.0], atol=1e-4)
    assert np.allclose(dec[2:4], [0.0, 0.0], atol=1e-4)  # relleno con ceros
