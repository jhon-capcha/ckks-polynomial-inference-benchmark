# Decisión de librería y validación del entorno — Hito 0

## 1. Propósito del documento

Este documento registra la decisión técnica adoptada para implementar el componente de
cifrado homomórfico del proyecto, así como el proceso de instalación y validación del
entorno sobre Windows 11.

Su objetivo es dejar evidencia reproducible de:

- La librería seleccionada.
- La justificación de la elección.
- Las versiones utilizadas.
- Las dificultades encontradas durante la instalación.
- La solución aplicada.
- El resultado del smoke test con CKKS.
- El criterio de cierre del Hito 0.

---

## 2. Decisión adoptada

Se adopta **Pyfhel 3.5.0** como interfaz Python para trabajar con el esquema CKKS,
utilizando Microsoft SEAL como backend criptográfico.

Esta elección permite desarrollar el laboratorio experimental íntegramente en Python,
manteniendo compatibilidad con el enfoque metodológico definido en la sinopsis del
proyecto.

Pyfhel proporciona una API de alto nivel para:

- Crear contextos CKKS.
- Generar claves.
- Codificar y cifrar vectores de números reales.
- Ejecutar operaciones homomórficas.
- Descifrar y decodificar resultados.
- Acceder a información relevante del ciphertext durante la experimentación.

La adopción de Pyfhel evita tener que desarrollar una integración propia entre Python y
Microsoft SEAL en C++, lo que reduciría la productividad y aumentaría la complejidad del
laboratorio sin aportar valor directo a las preguntas de investigación.

---

## 3. Relación con el alcance del proyecto

La elección de Pyfhel está alineada con las decisiones metodológicas del proyecto:

- Esquema criptográfico: CKKS.
- Lenguaje principal: Python.
- Sistema operativo: Windows 11.
- Entorno de desarrollo: Visual Studio Code.
- Hardware: CPU, sin GPU.
- Entrenamiento del modelo: en texto plano.
- Inferencia: sobre datos cifrados.
- Backend: Microsoft SEAL.
- Bootstrapping: no ejecutado ni medido.

La librería seleccionada no modifica el título, el problema, las preguntas de
investigación ni el diseño experimental aprobado.

---

## 4. Entorno validado

| Componente                | Versión / configuración               |
| ------------------------- | ------------------------------------- |
| Sistema operativo         | Windows 11 x64                        |
| Build del sistema         | Windows 10.0.26200                    |
| Python                    | 3.11.9                                |
| Arquitectura de Python    | 64 bits                               |
| Entorno virtual           | `.venv`                               |
| pip                       | 26.1.2                                |
| setuptools                | 83.0.0                                |
| wheel                     | 0.47.0                                |
| packaging                 | 26.2                                  |
| Pyfhel                    | 3.5.0                                 |
| NumPy                     | 2.4.6                                 |
| Visual Studio Build Tools | 2022, versión 17.14                   |
| MSVC                      | 19.44                                 |
| Toolset                   | 14.44                                 |
| CMake                     | 3.31.6                                |
| NMake                     | 14.44                                 |
| Windows SDK               | 10.0.26100.0                          |
| Backend compilado         | Microsoft SEAL 4.1                    |
| Repositorio               | `ckks-polynomial-inference-benchmark` |
| Commit validado           | `f608a1a`                             |

---

## 5. Diagnóstico inicial

El equipo tenía varias versiones de Python instaladas:

| Comando            | Resultado                     |
| ------------------ | ----------------------------- |
| `python --version` | Python 3.11.9                 |
| `py --version`     | Python 3.14.3                 |
| `py -0p`           | Python 3.14, 3.13, 3.11 y 3.9 |

Se seleccionó **Python 3.11.9 x64** para el proyecto por ser una versión estable y
compatible con el stack científico y con Pyfhel.

El entorno virtual se creó mediante:

```powershell
py -3.11 -m venv .venv
```
