"""
Inferencia por etapas para el bloque final cifrado (Hito 4F).

Separa la CNN polinómica en:
  - prefijo (claro): imagen -> conv1 -> act1 -> pool -> conv2 -> act2 -> pool
                     -> flatten -> fc1 -> z (preactivación de act3, vector 120)
  - bloque final: z -> act3 -> fc2 -> logits

z es la salida de fc1, ANTES de aplicar act3. El bloque final se ejecutará
bajo CKKS; el prefijo siempre en claro (alcance oficial del Hito 4).

Incluye verificación de equivalencia estructural: el flujo por etapas
(prefijo + bloque claro) debe dar los mismos logits que el modelo completo,
para garantizar que el benchmark compara la misma implementación clara.
"""

from __future__ import annotations

import torch
from torch import nn


def compute_act3_preactivation(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Ejecuta el prefijo en claro y devuelve z (preactivación de act3).

    z = fc1(flatten(pool2(act2(conv2(pool1(act1(conv1(image)))))))).
    Es la entrada a act3, antes de aplicar el polinomio.
    """
    with torch.no_grad():
        x = model.conv1(image)
        x = model.act1(x)
        x = model.pool1(x)
        x = model.conv2(x)
        x = model.act2(x)
        x = model.pool2(x)
        x = model.flatten(x)
        z = model.fc1(x)  # preactivación de act3
    return z


def compute_final_block_clear(model: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """Ejecuta el bloque final en claro: z -> act3 -> fc2 -> logits."""
    with torch.no_grad():
        a = model.act3(z)
        logits = model.fc2(a)
    return logits


def staged_forward(model: nn.Module, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward por etapas. Devuelve (z, logits) con logits = bloque(z)."""
    z = compute_act3_preactivation(model, image)
    logits = compute_final_block_clear(model, z)
    return z, logits


def verify_staged_equivalence(
    model: nn.Module, image: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-6
) -> dict:
    """Verifica que el flujo por etapas == modelo completo.

    Garantiza que compute_act3_preactivation + compute_final_block_clear
    reproduce exactamente model(image). Si no, el benchmark compararía dos
    implementaciones claras distintas.
    """
    with torch.no_grad():
        logits_full = model(image)
    _z, logits_staged = staged_forward(model, image)

    equal = torch.allclose(logits_full, logits_staged, rtol=rtol, atol=atol)
    max_diff = float(torch.max(torch.abs(logits_full - logits_staged)))
    return {
        "equivalent": bool(equal),
        "max_abs_diff": max_diff,
        "logits_full": logits_full.numpy(),
        "logits_staged": logits_staged.numpy(),
    }
