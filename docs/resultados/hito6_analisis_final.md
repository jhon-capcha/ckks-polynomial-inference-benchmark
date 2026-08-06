# Hito 6 — Análisis final

## 1. Objetivo del Hito 6

Interpretar los resultados de los Hitos 3 (aproximación polinómica), 4
(integración CKKS) y 5 (benchmarking), y responder las tres preguntas de
investigación del proyecto, convirtiendo la evidencia experimental en una guía
de selección de configuración. El Hito 6 no genera datos primarios nuevos:
integra, verifica y sintetiza los resultados ya obtenidos.

## 2. Evidencia experimental consolidada

Fuentes oficiales por dimensión:

- **Precisión de aproximación**: `hito3c_validation_test_comparison.csv` (test
  completo, 10 000 imágenes). Base ReLU = 0.9901.
- **Transparencia del cifrado**: `hito4_ckks_validation_summary.csv` (100
  imágenes estratificadas, 600 inferencias) y `hito4_polynomial_ckks_error.csv`
  (error CKKS por activación).
- **Latencia**: `hito5_latency_by_config.csv` (corrida oficial, 1800
  inferencias con desglose por etapa).
- **Huella de almacenamiento**: `hito5_resource_consumption.csv` (tamaños
  serializados).
- **Trade-off consolidado**: `hito6_master_results.csv` (tabla maestra que une
  las anteriores por configuration_id).

Tabla maestra (accuracy sobre test completo, latencia oficial, niveles, huella):

| Configuración | Accuracy | Δaprox (pp) | Niveles | Latencia (ms) | Claves rot. (MB) |
|---|---|---|---|---|---|
| chebyshev_d5_I1 | 0.9862 | 0.39 | 6 | 244.8 | 312.2 |
| least_squares_d5_I1 | 0.9843 | 0.58 | 6 | 244.8 | 312.2 |
| least_squares_d5_I2 | 0.9836 | 0.65 | 6 | 244.8 | 312.2 |
| chebyshev_d5_I2 | 0.9817 | 0.84 | 6 | 244.4 | 312.2 |
| least_squares_d3_I1 | 0.9647 | 2.54 | 4 | 189.0 | 171.6 |
| chebyshev_d3_I1 | 0.9443 | 4.58 | 4 | 188.7 | 171.6 |

## 3. Respuesta a RQ1

*¿De qué manera el método, el grado y el intervalo de aproximación polinómica
impactan en el consumo de niveles de la cadena de módulos y en la evolución del
error numérico de CKKS durante la evaluación de las funciones de activación?*

**Respuesta directa.** El grado fue el factor determinante del consumo de
niveles. En la implementación Horner utilizada, la evaluación de una activación
de grado 3 consumió tres niveles y la de grado 5 consumió cinco. Para un mismo
grado, el método de construcción y el intervalo no alteraron la estructura del
circuito homomórfico —misma secuencia de multiplicaciones, rescale y niveles
consumidos—, pero sí produjeron diferencias en el error numérico CKKS
observado, debido a los distintos coeficientes, magnitudes intermedias y rangos
de entrada. El error también dependió de la activación: act3 presentó el mayor
MAE CKKS en las seis configuraciones evaluadas.

**Evidencia cuantitativa.**
- Consumo por grado: en la evaluación aislada de act3, grado 3 consumió tres
  niveles y grado 5 consumió cinco (uniforme en las seis configuraciones).
- Invariancia estructural: `levels_consumed` idéntico para configuraciones del
  mismo grado, independientemente del método y el intervalo.
- Variación del error numérico: para grado 5 y el mismo circuito, el MAE CKKS
  de act3 varió desde 6.73×10⁻⁵ (chebyshev_d5_I2) hasta 1.57×10⁻⁴
  (chebyshev_d5_I1), una diferencia aproximada de 2.3 veces.
- Diferencia entre activaciones: act3 presentó el mayor error entre las tres
  activaciones en las seis configuraciones.
- Rango global: el MAE CKKS estuvo entre 3.0×10⁻⁶ y 1.57×10⁻⁴ en las 18
  mediciones (seis configuraciones × tres activaciones).

**Explicación técnica.** El consumo de niveles estuvo gobernado por la
profundidad multiplicativa. En Horner, cada etapa incorpora una multiplicación
seguida de un rescale; por ello los grados 3 y 5 consumieron tres y cinco
niveles. Como el método y el intervalo solo modifican los coeficientes
manteniendo el grado y la secuencia de operaciones, no se observaron diferencias
en el consumo de niveles. El error numérico CKKS sí varió entre configuraciones
estructuralmente equivalentes, porque los coeficientes distintos modifican los
valores intermedios y la propagación del error. Dado que las activaciones se
evaluaron aisladamente sobre muestras de preactivación, el mayor error de act3
no demuestra por sí solo una acumulación homomórfica en cascada entre capas.

**Alcance.** Evaluación aislada de activaciones con Horner bajo CKKS,
poly_modulus_degree = 16384, seguridad configurada de 128 bits, escala 2⁴⁰,
Pyfhel 3.5.0, 200 muestras de preactivación por activación y configuración.

**Limitaciones.** La invariancia entre método e intervalo se refiere al costo
estructural (niveles y operaciones), no al error numérico, que sí varió. Los
tres y cinco niveles corresponden a la activación aislada; el bloque cifrado
act3→fc2 consumió cuatro y seis. El predominio del error de act3 se verificó en
ReducedLeNet con estas configuraciones; no debe generalizarse. El error CKKS
medido cuantifica la diferencia entre evaluación polinómica clara y cifrada, no
el error funcional frente a ReLU.

## 4. Respuesta a RQ2

*¿Cuál es la relación cuantitativa entre el grado del polinomio, la profundidad
multiplicativa y la pérdida de precisión bajo inferencia segura, y qué trade-off
define?*

**Respuesta directa.** El grado determinó directamente la profundidad
multiplicativa y mostró una relación clara con la precisión. Bajo Horner, el
bloque cifrado act3→fc2 consumió cuatro niveles con grado 3 y seis con grado 5.
La evaluación aislada de grado 7 ya excedió la profundidad disponible del perfil
seguro. En precisión, grado 3 perdió entre 2.54 y 4.58 puntos porcentuales
frente a ReLU, mientras grado 5 redujo esa pérdida a entre 0.39 y 0.84 puntos
porcentuales. El trade-off sitúa a grado 5 como el punto de mayor precisión
viable dentro del perfil: preservó casi toda la accuracy, operando en el límite
útil de profundidad del bloque. El resultado no dependió únicamente del grado;
el método y el intervalo también influyeron en la precisión a igual profundidad.

**Evidencia cuantitativa.**
- Profundidad: grado 3 → 4 niveles, grado 5 → 6 niveles (bloque act3→fc2). La
  activación aislada de grado 7 no completó dentro de la cadena segura; el
  bloque requeriría aún más profundidad.
- Pérdida de precisión (base ReLU test = 0.9901): grado 5 — chebyshev_d5_I1
  0.9862 (−0.39pp, mejor de grado 5), least_squares_d5_I1 0.9843 (−0.58pp),
  least_squares_d5_I2 0.9836 (−0.65pp), chebyshev_d5_I2 0.9817 (−0.84pp); grado
  3 — least_squares_d3_I1 0.9647 (−2.54pp), chebyshev_d3_I1 0.9443 (−4.58pp).
- Método según grado: en grado 5, Chebyshev > LSQ (0.9862 vs 0.9843); en grado
  3, LSQ > Chebyshev (0.9647 vs 0.9443). Ningún método dominó uniformemente.
- Efecto del cifrado: en las 100 imágenes del Hito 4F, concordancia poly↔CKKS =
  1.000 y cero cambios de predicción; el MAE de logits (no nulo, 4.7×10⁻⁵ a
  4.0×10⁻⁴) no alteró el argmax.

**Explicación técnica.** Horner evalúa un polinomio de grado d con d
multiplicaciones encadenadas, cada una con su rescale; por eso grado 5 consumió
dos niveles más que grado 3, más el nivel de fc2. En las configuraciones
estables, pasar de grado 3 a grado 5 incrementó la capacidad de representar ReLU
y redujo la pérdida de clasificación. Los dos efectos operan en sentidos
opuestos sobre el mismo parámetro: mayor grado implica menor pérdida de
aproximación, pero mayor profundidad y menor margen criptográfico. El error CKKS
permaneció varios órdenes por debajo de las diferencias funcionales relevantes y
no modificó las predicciones en la muestra evaluada; por ello, la pérdida de
accuracy estuvo dominada por la aproximación, no por el cifrado. Un grado mayor
no garantiza universalmente mejor precisión: configuraciones de grados
superiores evaluadas en el Hito 3 presentaron extrapolación e inestabilidad.

**Alcance.** Las accuracies de aproximación proceden del test completo (10 000
imágenes, Hito 3C). La transparencia del cifrado, del Hito 4F (100 imágenes, 600
inferencias). Profundidad y factibilidad bajo N=16384, cadena oficial por grado,
escala 2⁴⁰, 128 bits, Horner, sin bootstrapping, Pyfhel 3.5.0.

**Limitaciones.** Grado 5 preservó casi toda la precisión, pero no igualó a
ReLU (la mejor perdió 0.39pp). ΔAcc_CKKS = 0 significa que no hubo cambios de
accuracy en las 100 imágenes, no que la evaluación cifrada fuera exacta (hubo
error de logits no nulo). La no factibilidad de grado 7 es específica de Horner,
N=16384, la cadena evaluada y la ausencia de bootstrapping. Las diferencias
entre métodos fueron descriptivas, sin prueba inferencial. chebyshev_d5_I1 fue
la mejor de grado 5, no la mejor global: la configuración polinómica clara con
mayor accuracy fue least_squares_d7_I2 (0.9864), no factible bajo CKKS.

## 5. Respuesta a RQ3

*¿Cómo varía la latencia de la inferencia cifrada según el método y el grado, y
en qué medida las configuraciones de menor consumo de niveles amplían la
profundidad evaluable dentro de una cadena fija antes de requerir un refresco
del ciphertext?*

**Respuesta directa.** La latencia absoluta del bloque cifrado final estuvo
dominada por la capa fc2, debido al costo de las rotaciones de los productos
matriz-vector. Sin embargo, la diferencia de latencia entre grado 3 y grado 5 la
explicó principalmente la mayor profundidad de act3. El paso de grado 3 a grado
5 incrementó la mediana de latencia de aproximadamente 189 ms a 245 ms (~30%).
Para un mismo grado, no se observaron diferencias materialmente relevantes entre
métodos ni intervalos (medianas con variación < 0.5 ms). Respecto a la
profundidad evaluable, cada grado operó bajo un perfil dimensionado a su medida
y consumió toda la cadena disponible; el estudio no evaluó una cadena fija
compartida, por lo que la relación entre menor grado y mayor margen se plantea
como inferencia estructural, no como resultado medido.

**Evidencia cuantitativa.**
- Latencia por grado: grado 3 = 188.7–189.0 ms (P95 203–204 ms); grado 5 =
  244.4–244.8 ms (P95 257–263 ms); incremento ≈29.5%.
- Método/intervalo: en grado 5, act3 varió de 81.24 a 81.49 ms entre las cuatro
  configuraciones (rango 0.25 ms); latencia total 244.4–244.8 ms.
- Origen del incremento (chebyshev_d3 → chebyshev_d5): act3 +42.4 ms (76% del
  total +56 ms), fc2 +10.1 ms, encrypt +3.2 ms, decrypt 0 ms.
- Distribución: fc2 = 69.9% (grado 3) → 58.0% (grado 5); act3 = 20.7% → 33.3%.
- Perfiles: d3 = [60,40,40,40,40,60] (profundidad 4), d5 =
  [60,40,40,40,40,40,40,60] (profundidad 6); ambos consumidos por completo.

**Explicación técnica.** La evaluación Horner de grado 5 contiene dos etapas
multiplicativas adicionales respecto de grado 3, lo que alarga el camino crítico
de act3 y explica la mayor parte de la diferencia entre grados. fc2 mantuvo la
misma estructura (70 rotaciones = 7 por logit × 10 logits) en todas las
configuraciones; fue la principal fuente de latencia absoluta, pero no la
principal causa del incremento entre grados. Se observó un aumento de ~10 ms en
fc2 bajo el perfil de grado 5, que puede asociarse al distinto perfil
criptográfico y al estado de los ciphertexts, aunque el experimento no aisló
causalmente ese efecto. Método e intervalo no afectan la latencia porque solo
cambian los coeficientes, no la secuencia de operaciones.

**Alcance.** Corrida oficial del Hito 5B: 1800 inferencias (6 × 10 × 30 +
warm-up), orden aleatorizado, temporización con perf_counter separando encrypt,
act3, fc2 y decrypt. Bloque act3→fc2 (del cifrado de la preactivación de act3 al
descifrado de los diez logits); el prefijo convolucional se ejecutó en claro.
CPU, N=16384, 128 bits, Pyfhel 3.5.0. Mediana como estadística principal, P95
para la cola.

**Limitaciones.** Las cifras oficiales son 189/245 ms; no deben mezclarse con la
corrida preliminar (185/240 ms) archivada. La medición cubre el bloque final,
no una CNN completamente cifrada. El estudio no evaluó una cadena fija
compartida entre grados; el margen adicional de un menor grado es inferencia
estructural condicional, no comparación experimental directa. El bootstrapping
no se implementó, por lo que no se midió el punto de refresco del ciphertext.
Las equivalencias de latencia se establecieron por estadística descriptiva.

## 6. Síntesis precisión–profundidad–latencia

El proyecto caracteriza empíricamente el acoplamiento entre la aproximación
polinómica de las activaciones y el presupuesto de la cadena de módulos:

- **El grado es el parámetro que acopla las tres dimensiones.** Determina la
  profundidad (niveles), gobierna la pérdida de precisión por aproximación y, a
  través de act3, el incremento de latencia. Un grado mayor mejora la precisión
  pero consume más profundidad, más latencia y más huella de claves.
- **El cifrado es funcionalmente transparente.** En la muestra evaluada, CKKS no
  cambió ninguna predicción; el error de clasificación proviene de la
  aproximación, no del esquema criptográfico.
- **El costo absoluto y el costo marginal tienen orígenes distintos.** fc2
  (rotaciones) domina el tiempo total; act3 (multiplicaciones de Horner) domina
  el incremento entre grados.
- **Método e intervalo afectan la precisión, no el costo.** A igual grado, la
  estructura del circuito —y por tanto latencia y huella— es la misma; método e
  intervalo solo cambian la calidad de aproximación (y el error numérico CKKS).

## 7. Frontera de Pareto

Se calcularon dos fronteras (accuracy sobre test completo vs latencia oficial):

- **Frontera estricta** (medianas exactas): chebyshev_d5_I1, chebyshev_d5_I2,
  least_squares_d3_I1, chebyshev_d3_I1. Dos configuraciones (chebyshev_d5_I2,
  chebyshev_d3_I1) ingresan por ventajas de latencia de 0.3–0.4 ms.

- **Frontera material** (tolerancia ε_latencia = 0.5 ms, coherente con el
  umbral de materialidad de RQ3; ε_accuracy = 0.001): **least_squares_d3_I1** y
  **chebyshev_d5_I1**. Las diferencias submilisegundo, no materiales según el
  análisis de latencia, se descartan: least_squares_d3_I1 domina materialmente a
  chebyshev_d3_I1 (+2.04pp, latencia equivalente), y chebyshev_d5_I1 domina a
  las demás de grado 5 (mayor accuracy, latencia equivalente).

La frontera material es la oficial para la guía de selección. La estricta se
conserva como resultado matemático auditable.

## 8. Guía de selección

La frontera material identifica dos regímenes operativos dentro de las seis
configuraciones factibles.

| Prioridad | Configuración | Justificación |
|---|---|---|
| Menor costo criptográfico | least_squares_d3_I1 | Accuracy 0.9647; mediana 189.0 ms; 4 niveles; claves de rotación 171.6 MB. Mejor accuracy de grado 3; domina materialmente a chebyshev_d3_I1. |
| Máxima precisión viable | chebyshev_d5_I1 | Accuracy 0.9862 (−0.39pp vs ReLU); mediana 244.8 ms; 6 niveles; claves 312.2 MB. Mayor accuracy factible; domina materialmente a las demás de grado 5. |

**Naturaleza del trade-off.** Bajo los objetivos de maximizar accuracy y
minimizar latencia, y con la tolerancia material de 0.5 ms, no apareció una
tercera configuración no dominada como compromiso intermedio. La decisión queda
reducida a dos alternativas:

- Elegir least_squares_d3_I1 sobre chebyshev_d5_I1: reduce la latencia 55.8 ms
  (≈22.8%), usa dos niveles menos y reduce las claves de rotación 140.6 MB
  (≈45%), a cambio de 2.15 pp menos de accuracy.
- Elegir chebyshev_d5_I1 sobre least_squares_d3_I1: recupera 2.15 pp de
  accuracy, con ≈29.5% más de latencia, dos niveles adicionales y ≈81.9% más de
  claves de rotación.

**Guía por restricción específica.**

| Restricción dominante | Elección | Justificación |
|---|---|---|
| Menor latencia | least_squares_d3_I1 | 189.0 ms vs 244.8 ms |
| Menor profundidad | least_squares_d3_I1 | 4 niveles vs 6 |
| Menor huella de claves | least_squares_d3_I1 | 171.6 MB vs 312.2 MB |
| Máxima precisión factible | chebyshev_d5_I1 | 0.9862 (−0.39pp) |
| Método dentro de grado 3 | Mínimos cuadrados | Mayor accuracy, costo equivalente |
| Método dentro de grado 5, I1 | Chebyshev | Mayor accuracy, costo equivalente |
| Mayor margen bajo cadena común | Grado 3 (condicional) | Consume dos niveles menos; no medido bajo perfil común |

**Sobre grado 7.** La configuración polinómica clara con mayor accuracy fue
least_squares_d7_I2 (0.9864), no factible mediante Horner bajo el perfil seguro.
Su ventaja sobre chebyshev_d5_I1 fue de solo 0.02 pp; grado 5 obtuvo
prácticamente la misma precisión con factibilidad criptográfica demostrada.

**Transparencia del cifrado.** En las seis configuraciones factibles y sobre las
100 imágenes del Hito 4F, la concordancia poly↔CKKS fue 1.000 y no hubo cambios
de predicción. El error de logits fue no nulo pero no alteró el argmax en esa
muestra. Este resultado no garantiza concordancia perfecta fuera de la muestra.

## 9. Limitaciones

Las siguientes restricciones corresponden al alcance deliberadamente fijado para
centrar el estudio en las activaciones polinómicas y su trade-off:

1. **Cifrado parcial de la red.** Solo se cifró el bloque final act3→fc2; el
   prefijo convolucional se ejecutó en claro. Las latencias no representan una
   CNN completamente cifrada de extremo a extremo.
2. **Ausencia de bootstrapping.** Se caracterizó el consumo de niveles sin
   refresco; no se midió el punto operativo de refresco del ciphertext.
3. **Factibilidad condicionada de grado 7.** No fue factible mediante Horner con
   N=16384, la cadena segura evaluada, escala 2⁴⁰ y sin bootstrapping.
   least_squares_d7_I2 (mayor accuracy clara entre las evaluadas) quedó fuera del
   bloque cifrado. No implica inviabilidad universal de grado 7.
4. **Empaquetado de salida no optimizado.** Cada logit se devolvió en un
   ciphertext independiente; se priorizó claridad y validación sobre eficiencia.
5. **Única plataforma de ejecución.** CPU, un único entorno de hardware y
   software; las latencias absolutas no deben extrapolarse a otras plataformas,
   bibliotecas o versiones.
6. **Perfiles distintos por grado.** Cada grado usó una cadena dimensionada a su
   profundidad; no se comparó el margen restante bajo una cadena común.
7. **Muestra cifrada limitada.** La transparencia se validó sobre 100 imágenes;
   la pérdida de aproximación sobre las 10 000 del test.
8. **Seguridad dependiente de la biblioteca.** No se realizó estimación
   criptográfica independiente. Se usó la configuración de 128 bits aceptada por
   Pyfhel/SEAL, descartando perfiles que exigían desactivar esa garantía.
9. **Comparación limitada de implementaciones.** No se compararon otras
   bibliotecas CKKS ni estrategias alternativas de backend.
10. **Análisis estadístico descriptivo.** Las similitudes de latencia se
    determinaron descriptivamente, sin pruebas de equivalencia. Las diferencias
    de accuracy no se sometieron a pruebas pareadas ni intervalos de confianza.
11. **Huella serializada, no memoria residente.** Se midieron tamaños
    serializados de claves y ciphertexts, no memoria RAM máxima ni asignaciones
    nativas del proceso.
12. **Modelo de amenaza acotado.** El estudio se centró en la confidencialidad
    durante la evaluación; no abordó integridad verificable, servidores
    maliciosos, canales laterales ni gestión de claves en producción.

## 10. Trabajo futuro

1. **Optimizar el empaquetado de logits** en uno o menos ciphertexts, con
   potencial de reducir la huella de salida cerca de un orden de magnitud, sujeto
   al costo de rotaciones y máscaras.
2. **Reducir las rotaciones de fc2** mediante esquemas diagonales,
   baby-step/giant-step u otras técnicas de producto matriz-vector.
3. **Evaluar N=32768**, analizando si permite ejecutar grado 7 manteniendo
   seguridad, y cuantificando el incremento de latencia y tamaños.
4. **Evaluar circuitos polinómicos balanceados** (Paterson–Stockmeyer, árboles
   de potencias) y comparar su factibilidad frente a Horner.
5. **Incorporar bootstrapping** y medir cuánta profundidad adicional permite
   antes y después del refresco.
6. **Extender el cifrado a la CNN completa** (convoluciones y pooling
   homomórficos) para medir la latencia extremo a extremo.
7. **Ampliar arquitecturas y datasets** para evaluar la generalización.
8. **Evaluar aceleración especializada** con backends HE sobre GPU o FPGA,
   reconociendo que requeriría una plataforma o adaptación distinta.
9. **Explorar otros perfiles CKKS** (escalas, cadenas, niveles de seguridad).
10. **Comparar bibliotecas** (OpenFHE, TenSEAL u otras compatibles con CKKS).
11. **Incorporar inferencia estadística**: pruebas pareadas (McNemar),
    intervalos de confianza por bootstrap, pruebas de equivalencia para latencia.
12. **Medir recursos de ejecución**: memoria residente, uso de CPU, energía,
    transferencia de datos.
13. **Ampliar el modelo de amenaza**: gestión de claves, integridad de
    resultados, servidores maliciosos y canales laterales.

## 11. Conclusiones

El proyecto caracterizó empíricamente el trade-off precisión–profundidad–
latencia en la inferencia segura del bloque final de una CNN con CKKS, evaluando
seis configuraciones de aproximación polinómica (Chebyshev y mínimos cuadrados,
grados 3 y 5, intervalos I1 e I2).

Los hallazgos centrales:

1. **El cifrado CKKS es funcionalmente transparente**: en la muestra evaluada no
   cambió ninguna predicción; el error de clasificación proviene de la
   aproximación polinómica, no del esquema criptográfico.
2. **El grado acopla las tres dimensiones**: determina la profundidad, gobierna
   la pérdida de precisión y, vía act3, el incremento de latencia.
3. **Grado 5 es el punto de mayor precisión viable**: preserva casi toda la
   accuracy de ReLU (pérdida de 0.39 pp la mejor configuración), operando en el
   límite de profundidad del perfil seguro; grado 7 no fue factible.
4. **El costo tiene dos orígenes distintos**: fc2 (rotaciones) domina la latencia
   absoluta; act3 (multiplicaciones de Horner) domina el incremento con el grado.
5. **Método e intervalo afectan la precisión, no el costo estructural**.
6. **La decisión práctica es binaria**: least_squares_d3_I1 (menor costo) o
   chebyshev_d5_I1 (máxima precisión viable), con un trade-off de 2.15 pp de
   accuracy por ≈30% de latencia, dos niveles y ≈82% de huella de claves.

Estos resultados responden las tres preguntas de investigación con evidencia
reproducible y cuantificada, y ofrecen una guía de selección fundamentada en la
frontera de Pareto material.

## 12. Trazabilidad de resultados

- **Matriz de evidencia**: `results/tables/hito6_evidence_matrix.csv` (20
  afirmaciones, cada una con fuente, columnas, configuraciones, valor, alcance y
  limitación).
- **Tabla maestra**: `results/tables/hito6_master_results.csv` (6
  configuraciones, ambas fronteras de Pareto).
- **Fuentes primarias**: Hito 3C (precisión test), Hito 4 (transparencia CKKS,
  error por activación), Hito 5 (latencia, huella).
- **Figuras**: `results/figures/hito6/` (frontera material, trade-off por
  régimen).
- **Análisis y manifiesto**: `results/published/hito6_final_analysis.json`,
  `hito6_manifest.json`.

**Estado: Hito 6 completo.**
