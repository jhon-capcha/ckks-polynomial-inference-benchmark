# Hito 3C — Integración de las aproximaciones polinómicas en la CNN clara

## Objetivo

Sustituir las tres activaciones ReLU de ReducedLeNet por aproximaciones
polinómicas y cuantificar la pérdida de clasificación atribuible
exclusivamente a la aproximación, en texto plano, antes de introducir CKKS:

Δ_aproximación = accuracy(ReLU) − accuracy(polinomio)

No se mide todavía error criptográfico, consumo de niveles, rescale ni
latencia homomórfica; eso corresponde a los Hitos 4 y 5.

## Diseño experimental

Los 72 polinomios del Hito 3A se agrupan en 24 configuraciones de red
(3 métodos × 4 grados × 2 intervalos), cada una con una terna coherente
act1/act2/act3 del mismo método, grado e intervalo lógico. No son 72 redes.

## Precisión de evaluación (pilotos 3C-P1, 3C-P2)

Se evaluaron tres modos de dtype. El piloto sobre la activación aislada mostró
que float32 no desborda (ni siquiera con salidas ~10⁷) y que su error relativo
frente a float64 es ~10⁻⁵, unas 10⁴ veces menor que el error de aproximación
del polinomio. El piloto sobre la CNN completa confirmó que el dtype de
evaluación no altera ninguna predicción ni la accuracy en las configuraciones
finitas. Se congeló float32 (dtype natural del modelo) como modo principal.

Nota metodológica: en la configuración extrema (Taylor grado 9), ambos modos
produjeron logits no finitos; la igualdad de predicciones bajo NaN no implica
equivalencia clasificatoria. float32 se adoptó por la evidencia de las
configuraciones finitas, no por el comportamiento de la configuración colapsada.

## Integración estructural (3C-A)

Se implementó PolynomialActivation (coeficientes como buffer no entrenable,
evaluación por Horner) y una fábrica que carga el checkpoint ReLU, valida la
terna, clona el modelo, sustituye las tres activaciones y verifica que los pesos
del backbone (conv1, conv2, fc1, fc2) permanezcan idénticos bit a bit. El número
de parámetros entrenables se conserva (51 902); los coeficientes son buffers.

## Benchmark de validación (3C-B)

Línea base ReLU sobre validación: accuracy 0.9887, F1 0.9886. (Difiere del
0.9901 de test porque es otra partición.)

De las 24 configuraciones: 18 válidas, 6 inválidas (no finitas). Se instrumentó
cada capa en streaming para registrar magnitudes y localizar el colapso.

### Hallazgos del benchmark

1. **La aproximación conserva la accuracy en las mejores configuraciones.**
   Chebyshev y LSQ de grado 5-7 en I2 quedan a menos de 0.3% de ReLU.

2. **La accuracy no es monótona en el grado.** En el intervalo estrecho I1,
   grado 7 es peor que grado 5 (chebyshev: 0.9852 → 0.7913) y grado 9 colapsa.
   En el intervalo ancho I2, los grados altos se mantienen estables. El
   intervalo estrecho con grado alto produce coeficientes de mayor magnitud
   (peor condicionamiento) que se amplifican en cascada.

3. **El error se propaga en cascada y colapsa en act3.** Las magnitudes crecen
   multiplicativamente entre activaciones. En una configuración estable
   (chebyshev d5 I2) van de 9 a 13 a 17; en una que colapsa (chebyshev d9 I1)
   van de 70 a 10⁷ a 10³⁸, desbordando float32 en act3. En todas las
   configuraciones inválidas, la primera capa no finita es act3 (la última
   activación, que recibe el error acumulado).

4. **Taylor es inservible en la CNN.** Colapsa a nivel de azar incluso en grado
   3 (accuracy ~0.09), y produce valores no finitos en grado 7 y superiores.

## Análisis y selección (3C-C)

Se unieron las tablas del Hito 3B (error funcional) y 3C-B (clasificación),
agregando las tres activaciones por configuración y calculando la amplificación
en cascada. Se distinguieron tres estados: válido (finito), prácticamente viable
(finito y no colapsado, accuracy ≥ 0.50) y elegible para CKKS (viable y
accuracy ≥ 0.90). Resultado: 18 válidas, 14 viables, 10 elegibles.

### Correlación error funcional ↔ pérdida de accuracy

La correlación principal (Chebyshev y LSQ elegibles) entre MAE funcional y
Δaccuracy es positiva pero moderada. Al incluir Taylor, la correlación aumenta
artificialmente porque Taylor actúa como outlier dominante (MAE y Δaccuracy
ambos extremos). Entre configuraciones competitivas, el MAE funcional predice
solo parcialmente la pérdida de accuracy: la propagación en cascada y la
estabilidad numérica importan tanto como el error funcional. Esto justifica
medir la accuracy integrada en la CNN y no solo el error funcional.

### Shortlist congelada

Se aplicaron reglas categóricas generales (mejor por método, mejor grado 5,
mejor grado 3, comparación de intervalo) sobre las configuraciones elegibles,
sin identificadores predefinidos. Las reglas produjeron 8 configuraciones
diversas (ambos métodos, grados 3/5/7, ambos intervalos), más Taylor grado 5
como baseline diagnóstico. La selección se congeló con hashes de las entradas y
test_used=false, antes de evaluar test.

## Evaluación final sobre test (3C-D)

Línea base ReLU sobre test: accuracy 0.9901, F1 0.9900.

Antes de tocar test se verificó la integridad de la selección congelada (hashes
de entradas coincidentes, test_used=false). Se evaluaron las 8 seleccionadas más
el baseline diagnóstico, sin re-seleccionar.

### Δ_aproximación final (test)

| Configuración              | test accuracy | Δ accuracy |
| -------------------------- | ------------- | ---------- |
| least_squares_d7_I2        | 0.9864        | +0.0037    |
| chebyshev_d5_I1            | 0.9862        | +0.0039    |
| chebyshev_d7_I2            | 0.9862        | +0.0039    |
| least_squares_d5_I1        | 0.9843        | +0.0058    |
| least_squares_d5_I2        | 0.9836        | +0.0065    |
| chebyshev_d5_I2            | 0.9817        | +0.0084    |
| least_squares_d3_I1        | 0.9647        | +0.0254    |
| chebyshev_d3_I1            | 0.9443        | +0.0458    |
| taylor_d5_I1 (diagnóstico) | 0.1548        | +0.8353    |

### Hallazgos finales

1. **La aproximación polinómica de ReLU es viable para inferencia.** La mejor
   configuración pierde solo ~0.37% de accuracy respecto a ReLU sobre datos
   nunca vistos.

2. **El salto de precisión ocurre entre grado 3 y grado 5.** Grado 3 pierde
   ~2.5-4.6%; grado 5 recupera casi toda la accuracy (pérdida ~0.4-0.8%); de
   grado 5 a 7 la mejora es marginal. Grado 5 aparece como el compromiso
   favorable entre precisión y profundidad multiplicativa.

3. **Chebyshev y LSQ son competitivos.** Ninguno domina uniformemente; LSQ es
   algo mejor en grado bajo. Preservar ambos permite comparar su comportamiento
   homomórfico en el Hito 4.

4. **El método de construcción es determinante.** Taylor pierde 83.5% de
   accuracy: un grado bajo no garantiza utilidad; la aproximación debe ser
   global, no local.

### Consistencia validación ↔ test

La diferencia entre accuracy de validación y test fue pequeña en las 8
configuraciones (máxima brecha absoluta 0.61%, en chebyshev_d3_I1). No se
observó evidencia material de sobreajuste a la partición de validación dentro de
las configuraciones evaluadas, lo que respalda la robustez de la shortlist
congelada.

## Disciplina anti-leakage

Los intervalos provienen de entrenamiento (Hito 2). La selección de
configuraciones se realizó sobre validación. El conjunto de test se utilizó una
sola vez, tras congelar la selección, sin modificar ninguna decisión. La
integridad se verificó mediante hashes antes de evaluar test.

## Criterio de cierre

Las 24 configuraciones se integraron en la CNN base sin modificar los pesos, se
evaluaron primero sobre validación, se cuantificó la pérdida de accuracy y F1
atribuible a la sustitución de ReLU, se congeló una selección sin utilizar test,
y se reportaron los resultados finales sobre test de forma reproducible.

**Estado: Hito 3C completo.**
