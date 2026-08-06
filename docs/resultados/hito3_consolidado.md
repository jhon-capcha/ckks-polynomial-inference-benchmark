# Hito 3 — Aproximación polinómica de activaciones (documentación consolidada)

## 1. Objetivo del Hito 3

Construir, evaluar y seleccionar aproximaciones polinómicas de la función de
activación ReLU, aptas para su evaluación bajo cifrado homomórfico CKKS. El
bloque abarca tres sub-hitos:

- **3A** — Construcción matemática de los polinomios (tres métodos).
- **3B** — Evaluación funcional del error de aproximación.
- **3C** — Integración en la CNN clara, selección congelada y evaluación final.

Este documento consolida el arco completo; el detalle por sub-hito permanece en
`hito3b_error_funcional.md` y `hito3c_integracion_cnn.md`.

## 2. Contexto y entradas

- **Modelo base**: ReducedLeNet entrenada con ReLU sobre MNIST (checkpoint
  época 8, accuracy de test 0.9901, F1 macro 0.9900).
- **Activaciones a aproximar**: act1, act2, act3 (las tres ReLU de la red).
- **Intervalos de aproximación** (del Hito 2): dos políticas por activación, I1
  (99% central) e I2 (99.9% central), asimétricas según la distribución de
  preactivaciones observada.

## 3. Construcción matemática (3A)

Se construyeron polinomios con tres métodos, exportados en base monomial para
comparabilidad:

- **Chebyshev**: proyección L² ponderada con peso 1/√(1−t²), distinta de un
  ajuste por mínimos cuadrados.
- **Mínimos cuadrados (LSQ)**: ajuste L² discreto uniforme sobre malla de 1000
  puntos, vía `numpy.linalg.lstsq` (SVD estable), sin ponderación.
- **Taylor**: derivación simbólica de la Softplus (SymPy) como aproximación
  suave de ReLU, con β=1 y punto de expansión x₀=0.

Grados evaluados: 3, 5, 7, 9. Se registró la distinción entre grado nominal y
grado efectivo (coeficientes de orden alto nulos afectan la profundidad
multiplicativa real bajo CKKS).

## 4. Evaluación funcional del error (3B)

Se evaluaron **72 configuraciones** (3 métodos × 4 grados × 3 activaciones × 2
intervalos, agrupadas por las combinaciones válidas), midiendo el error de
aproximación con jerarquía de métricas:

- **Métrica principal**: error empírico (sobre la distribución real de
  preactivaciones de validación).
- **Diagnóstico**: error uniforme (malla uniforme sobre todo el intervalo).

### Hallazgos del 3B

- **Bias_ratio bidireccional**: Chebyshev y LSQ presentan bias<1 (datos
  concentrados cerca del origen); Taylor presenta bias>1 (falla en las colas,
  casi vacías de datos reales).
- **Taylor grado alto es prácticamente inviable** en act2 y act3 (intervalo
  amplio + grado alto produce salidas >10⁶); en act1 (rango estrecho) sigue
  viable.
- Una configuración se invalida **solo por no-finitud**; un error alto pero
  finito es un resultado válido que permanece como evidencia.
- La viabilidad práctica se registró en columna separada (umbral operativo de
  salida máxima 10⁶), distinta de la validez matemática.

## 5. Integración en la CNN clara (3C)

Las aproximaciones se integraron en la CNN sustituyendo las ReLU por los
polinomios (los pesos del backbone se verificaron idénticos bit a bit tras la
sustitución; los 51 902 parámetros entrenables se conservaron, con los
coeficientes como buffers).

### Precisión de evaluación

Los pilotos 3C-P1/P2 fijaron el dtype de evaluación en float32: no desborda, no
altera predicciones, y su error numérico es ~10⁴ veces menor que el de
aproximación. El modo float64 quedó como diagnóstico.

### Los tres estados de una configuración

Se distinguieron tres estados, con umbrales operativos:

- **valid** (salidas finitas): 18 de 24 configuraciones.
- **practically_viable** (accuracy ≥ 0.50, no colapsada): 14.
- **eligible_for_ckks** (accuracy ≥ 0.90): 10.

### Hallazgo: cascada de error

Las magnitudes crecen multiplicativamente act1 → act2 → act3; las
configuraciones inválidas colapsan en act3. La accuracy no es monótona en el
grado (en I1, grados altos se degradan; I2 estabiliza).

### Correlación error funcional ↔ pérdida de accuracy

En el análisis principal (Chebyshev y LSQ, válidas/viables/elegibles, n=10):
Pearson 0.818, Spearman 0.467. El error funcional predice parcialmente la
pérdida de accuracy; la cascada y la estabilidad numérica importan igual. Taylor,
como outlier, infla la correlación cuando se incluye.

## 6. Selección congelada (3C-C)

La selección se realizó sobre **validación** (test reservado), por reglas
categóricas generales, sin IDs predefinidos:

- Mejor por método (máxima accuracy).
- Trade-off de grado 5.
- Grado 3 de baja profundidad.
- Comparación de intervalos.

La shortlist congelada contiene **8 configuraciones** (test_used=false):

| Configuración | Método | Grado | Intervalo |
|---|---|---|---|
| chebyshev_d7_I2 | Chebyshev | 7 | I2 |
| least_squares_d7_I2 | LSQ | 7 | I2 |
| chebyshev_d5_I1 | Chebyshev | 5 | I1 |
| least_squares_d5_I1 | LSQ | 5 | I1 |
| chebyshev_d5_I2 | Chebyshev | 5 | I2 |
| least_squares_d5_I2 | LSQ | 5 | I2 |
| chebyshev_d3_I1 | Chebyshev | 3 | I1 |
| least_squares_d3_I1 | LSQ | 3 | I1 |

Las dos configuraciones de grado 7 resultaron después **no factibles bajo CKKS**
con Horner en el perfil seguro (Hito 4); las seis de grados 3 y 5 constituyeron
el conjunto oficial evaluado en los Hitos 4-6.

## 7. Evaluación final sobre test (3C-D)

Aplicada solo tras congelar la selección. Sobre el test completo (10 000
imágenes, base ReLU 0.9901):

- **Mejor configuración clara**: least_squares_d7_I2 (0.9864, −0.37 pp), seguida
  de chebyshev_d7_I2 y chebyshev_d5_I1 (ambas 0.9862, −0.39 pp).
- **Grado 5** es el compromiso favorable entre precisión y profundidad.
- **Taylor colapsa** (pérdida de hasta −83.5 pp en las peores configuraciones).

### Consistencia validación ↔ test

La máxima brecha de generalización fue 0.61 pp (chebyshev_d3_I1); la selección
sobre validación generalizó a test sin degradación inesperada, sin evidencia
material de sobreajuste.

## 8. Disciplina anti-leakage

- La selección se congeló sobre validación con test_used=false; los hashes de
  las entradas se registraron antes de tocar el test.
- El test se usó únicamente para la evaluación final (3C-D), no para seleccionar.
- Esta disciplina se mantuvo en los Hitos 4-6: la selección del Hito 3C no se
  reajustó con resultados posteriores.

## 9. Conexión con los Hitos 4-6

- El Hito 4 evaluó las seis configuraciones factibles de grados 3 y 5 bajo CKKS,
  y determinó que grado 7 no es factible con Horner en el perfil seguro.
- El Hito 5 midió su latencia y huella.
- El Hito 6 respondió las preguntas de investigación y construyó la guía de
  selección, usando la accuracy del test completo de este Hito 3 como fuente
  oficial de precisión.

## 10. Evidencia y trazabilidad

- **Documentos por sub-hito**: `hito3b_error_funcional.md`,
  `hito3c_integracion_cnn.md`.
- **Tablas**: `hito3b_functional_metrics.csv` (72 configs),
  `hito3c_cnn_validation_metrics.csv`, `hito3c_cnn_test_metrics.csv`,
  `hito3c_combined_analysis.csv`, `hito3c_validation_test_comparison.csv`.
- **Evidencia publicada**: `hito3b_functional_metrics.json`, `hito3b_manifest.json`,
  `hito3c_analysis_summary.json`, `hito3c_frozen_selection.json` (shortlist
  congelada con hashes), `hito3c_selection_manifest.json`,
  `hito3c_validation_results.json`, `hito3c_test_results.json`,
  `hito3c_validation_manifest.json`.
- **Figuras**: 13 en `results/figures/hito3b/`, 11 en `results/figures/hito3c/`.
- **Manifiesto consolidado**: `results/published/hito3_manifest.json`.

## 11. Limitaciones

- La aproximación se evaluó sobre ReducedLeNet y MNIST; la generalización a otras
  arquitecturas y datos no se estudió.
- La selección usó umbrales operativos (0.50 para viabilidad, 0.90 para
  elegibilidad) definidos para este proyecto.
- El error funcional predice solo parcialmente la pérdida de accuracy; la
  correlación no es un predictor completo.
- Taylor se incluyó como baseline desfavorable (ReLU no es diferenciable en cero,
  sin serie de Taylor clásica alrededor del origen) y actúa como outlier.

## 12. Criterio de cierre

Polinomios construidos con tres métodos y exportados en base monomial (3A);
error funcional evaluado sobre 72 configuraciones con jerarquía de métricas (3B);
integración estructural verificada (pesos idénticos bit a bit), tres estados
distinguidos, cascada de error caracterizada, selección congelada de 8
configuraciones sobre validación con disciplina anti-leakage, y evaluación final
sobre test sin evidencia material de sobreajuste (3C). Evidencia reproducible y
versionada.

**Estado: Hito 3 completo (incluyendo la documentación consolidada 3D).**
