"""
Prueba mínima de Pyfhel con CKKS.

Objetivo:
    Confirmar que el entorno puede crear un contexto CKKS, generar claves,
    cifrar vectores, ejecutar operaciones homomórficas y descifrar resultados.

Esta prueba valida el entorno técnico del Hito 0.
No representa todavía la configuración criptográfica definitiva del experimento.
"""

from __future__ import annotations

import sys

import numpy as np
from Pyfhel import Pyfhel


def main() -> int:
    print("=" * 60)
    print("PRUEBA DE ENTORNO — Pyfhel + CKKS")
    print("=" * 60)

    # Parámetros de validación inicial.
    # Todavía no son los parámetros definitivos del experimento.
    ckks_params = {
        "scheme": "CKKS",
        "n": 2**14,
        "scale": 2**30,
        "qi_sizes": [60, 30, 30, 30, 60],
    }

    # 1. Crear contexto y claves.
    he = Pyfhel()

    context_status = he.contextGen(**ckks_params)
    print(f"\n[INFO] Estado del contexto: {context_status}")

    he.keyGen()
    he.relinKeyGen()

    print("[OK] Contexto CKKS y claves generados.")

    # 2. Datos de prueba.
    a = np.array([1.5, 2.0, 3.25], dtype=np.float64)
    b = np.array([0.5, 1.0, 0.75], dtype=np.float64)

    expected_sum = a + b
    expected_product = a * b

    print(f"\nDato a en claro: {a}")
    print(f"Dato b en claro: {b}")

    # 3. Cifrado.
    ctxt_a = he.encryptFrac(a)
    ctxt_b = he.encryptFrac(b)

    print("\n[OK] Datos cifrados.")

    # 4. Operaciones homomórficas.
    ctxt_sum = ctxt_a + ctxt_b
    ctxt_product = ctxt_a * ctxt_b

    print("[OK] Suma y multiplicación homomórficas realizadas.")

    # 5. Descifrado.
    result_sum = he.decryptFrac(ctxt_sum)[: len(a)]
    result_product = he.decryptFrac(ctxt_product)[: len(a)]

    # 6. Cálculo de errores.
    max_error_sum = float(np.max(np.abs(result_sum - expected_sum)))
    max_error_product = float(np.max(np.abs(result_product - expected_product)))

    print("\n--- RESULTADOS ---")
    print(f"Suma descifrada:          {np.round(result_sum, 8)}")
    print(f"Suma esperada:            {expected_sum}")
    print(f"Multiplicación descifrada:{np.round(result_product, 8)}")
    print(f"Multiplicación esperada:  {expected_product}")

    print(f"\nError máximo de suma:          {max_error_sum:.6e}")
    print(f"Error máximo de multiplicación:{max_error_product:.6e}")

    tolerance = 1e-3

    sum_ok = np.allclose(
        result_sum,
        expected_sum,
        rtol=0.0,
        atol=tolerance,
    )
    product_ok = np.allclose(
        result_product,
        expected_product,
        rtol=0.0,
        atol=tolerance,
    )

    if sum_ok and product_ok:
        print("\n[ÉXITO] CKKS funciona correctamente.")
        print("[ÉXITO] Criterio técnico del Hito 0 satisfecho.")
        return 0

    print("\n[REVISAR] Uno o más errores superan la tolerancia.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
