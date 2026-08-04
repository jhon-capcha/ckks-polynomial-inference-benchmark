"""
Representación de estado de un ciphertext CKKS (Hito 4B).

Captura una instantánea del estado de un ciphertext (nivel, escala, tamaño)
en un punto del cómputo, para instrumentar cómo evolucionan a lo largo de las
operaciones. No guarda el ciphertext en sí, solo sus metadatos.

En CKKS medimos el consumo de la cadena de módulos (mod_level) y la escala
(scale_bits), NO un "noise budget" en bits (eso es de BFV/BGV).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CiphertextState:
    """Instantánea de metadatos de un ciphertext en un punto del cómputo."""

    operation_index: int
    operation_name: str
    mod_level: int | None
    scale_bits: float | None
    ciphertext_size: int | None
    serialized_bytes: int | None = None

    @classmethod
    def capture(
        cls, ctxt, operation_index: int, operation_name: str, measure_bytes: bool = False
    ) -> CiphertextState:
        """Captura el estado de un ciphertext Pyfhel (PyCtxt)."""
        mod_level = _safe(lambda: ctxt.mod_level)
        scale_bits = _safe(lambda: float(ctxt.scale_bits))
        size = _safe(lambda: ctxt.size())
        sbytes = None
        if measure_bytes:
            sbytes = _safe(lambda: len(ctxt.to_bytes()))
        return cls(
            operation_index=operation_index,
            operation_name=operation_name,
            mod_level=mod_level,
            scale_bits=scale_bits,
            ciphertext_size=size,
            serialized_bytes=sbytes,
        )

    def as_dict(self) -> dict:
        return {
            "operation_index": self.operation_index,
            "operation_name": self.operation_name,
            "mod_level": self.mod_level,
            "scale_bits": self.scale_bits,
            "ciphertext_size": self.ciphertext_size,
            "serialized_bytes": self.serialized_bytes,
        }


def _safe(fn):
    try:
        return fn()
    except Exception:
        return None


@dataclass
class OperationRecorder:
    """Acumula estados de ciphertext a lo largo de una secuencia de operaciones."""

    states: list[CiphertextState] = field(default_factory=list)
    _counter: int = 0

    def record(self, ctxt, operation_name: str, measure_bytes: bool = False) -> None:
        state = CiphertextState.capture(ctxt, self._counter, operation_name, measure_bytes)
        self.states.append(state)
        self._counter += 1

    def reset(self) -> None:
        self.states.clear()
        self._counter = 0

    def as_list(self) -> list[dict]:
        return [s.as_dict() for s in self.states]

    def summary(self) -> dict:
        """Resumen: niveles consumidos, escala final, nº de operaciones."""
        if not self.states:
            return {"n_operations": 0}
        levels = [s.mod_level for s in self.states if s.mod_level is not None]
        return {
            "n_operations": len(self.states),
            "initial_mod_level": levels[0] if levels else None,
            "final_mod_level": levels[-1] if levels else None,
            "levels_consumed": (levels[-1] - levels[0]) if len(levels) >= 2 else 0,
            "final_scale_bits": self.states[-1].scale_bits,
        }
