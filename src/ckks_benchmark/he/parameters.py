"""
Perfiles de parámetros CKKS del Hito 4 (Pyfhel 3.5.0 / SEAL 4.1).

Perfiles congelados tras el piloto 4-P0, que estableció empíricamente:
    - N=16384 con escala 2^40 y primos intermedios de 40 bits
    - profundidad = número de primos intermedios (rescales disponibles)
    - grado 3 viable con margen, grado 5 viable (Horner), grado 7 no (Horner)
    - cadena de 440 bits rompe la seguridad de 128 bits (sec=0) → prohibida

Regla de seguridad: solo se aceptan perfiles con seguridad >= 128 bits.
El módulo total se calcula como suma de bits (la API no lo expone fiablemente).
"""

from __future__ import annotations

from dataclasses import dataclass

MINIMUM_SECURITY_BITS = 128


@dataclass(frozen=True)
class CKKSParameterProfile:
    """Perfil de parámetros CKKS. profundidad = nº de primos intermedios."""

    profile_id: str
    poly_modulus_degree: int
    coeff_mod_bit_sizes: tuple[int, ...]
    scale_bits: int
    security_bits: int = 128
    description: str = ""

    @property
    def total_coeff_modulus_bits(self) -> int:
        """Suma de bits de la cadena (la API de Pyfhel no lo expone fiablemente)."""
        return sum(self.coeff_mod_bit_sizes)

    @property
    def multiplicative_depth(self) -> int:
        """Profundidad = primos intermedios = len(cadena) - 2 (primero y último especiales)."""
        return len(self.coeff_mod_bit_sizes) - 2

    @property
    def slot_count(self) -> int:
        return self.poly_modulus_degree // 2

    def validate(self) -> None:
        """Valida coherencia y seguridad mínima."""
        if self.security_bits < MINIMUM_SECURITY_BITS:
            raise ValueError(
                f"Perfil {self.profile_id}: seguridad {self.security_bits} < "
                f"{MINIMUM_SECURITY_BITS} bits. No permitido para resultados oficiales."
            )
        if len(self.coeff_mod_bit_sizes) < 3:
            raise ValueError(f"Perfil {self.profile_id}: la cadena necesita al menos 3 primos.")
        # Límite empírico de seguridad para N=16384 (128-bit): ~438 bits.
        if self.poly_modulus_degree == 16384 and self.total_coeff_modulus_bits > 438:
            raise ValueError(
                f"Perfil {self.profile_id}: {self.total_coeff_modulus_bits} bits excede "
                f"el límite de ~438 para 128-bit en N=16384 (rompería la seguridad)."
            )


# --- Perfiles oficiales congelados ---

# Perfil shallow (grado 3): cadena de profundidad 4, margen holgado.
PROFILE_D3 = CKKSParameterProfile(
    profile_id="ckks_n16384_d3",
    poly_modulus_degree=16384,
    coeff_mod_bit_sizes=(60, 40, 40, 40, 40, 60),
    scale_bits=40,
    security_bits=128,
    description="Perfil para polinomios grado 3 (profundidad 4, margen para lineales).",
)

# Perfil medium (grado 5 + bloque final): cadena de profundidad 6.
PROFILE_D5 = CKKSParameterProfile(
    profile_id="ckks_n16384_d5",
    poly_modulus_degree=16384,
    coeff_mod_bit_sizes=(60, 40, 40, 40, 40, 40, 40, 60),
    scale_bits=40,
    security_bits=128,
    description="Perfil para polinomios grado 5 y bloque lineal final (profundidad 6).",
)

OFFICIAL_PROFILES = {
    PROFILE_D3.profile_id: PROFILE_D3,
    PROFILE_D5.profile_id: PROFILE_D5,
}


def get_profile(profile_id: str) -> CKKSParameterProfile:
    if profile_id not in OFFICIAL_PROFILES:
        raise KeyError(
            f"Perfil '{profile_id}' no es oficial. Disponibles: {list(OFFICIAL_PROFILES)}."
        )
    profile = OFFICIAL_PROFILES[profile_id]
    profile.validate()
    return profile


def profile_for_degree(degree: int) -> CKKSParameterProfile:
    """Perfil mínimo seguro para un grado dado (por Horner)."""
    if degree <= 3:
        return PROFILE_D3
    if degree <= 5:
        return PROFILE_D5
    raise ValueError(
        f"Grado {degree} no tiene perfil oficial (grado 7+ no factible con Horner "
        f"en N=16384/128-bit sin bootstrapping; ver Hito 4C para estrategias alternativas)."
    )
