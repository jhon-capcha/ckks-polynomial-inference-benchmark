# Hito 3B — Evaluación funcional de las aproximaciones

## Objetivo

Evaluar los 72 polinomios generados en el Hito 3A y cuantificar su error
matemático frente a ReLU, en texto plano, sin integrarlos aún en la CNN y sin
usar CKKS. El hito responde qué método aproxima mejor a ReLU, cómo influyen el
grado, el intervalo y la activación, cuánto difieren el error uniforme y el
empírico, y qué configuraciones son prácticamente inviables.

## Entradas congeladas

- **Polinomios:** `results/approximations/coefficients/polynomials.json`
  (72 configuraciones del Hito 3A, con trazabilidad SHA-256).
- **Preactivaciones:** `data/processed/preactivations_validation_sample.npz`
  (100 000 muestras por activación, conjunto de **validación**).
- Los intervalos I1/I2 provienen de **entrenamiento** (Hito 2). El error
  empírico se mide en **validación**. El conjunto de **prueba no se utiliza**.

## Metodología

Cada polinomio se evalúa de dos formas:

- **Uniforme:** malla de 10 000 puntos equiespaciados sobre el intervalo. Da
  igual peso a todo el dominio declarado. Métrica **diagnóstica**.
- **Empírica:** sobre las preactivaciones reales de validación de su misma
  activación. Refleja la distribución real de operación. Métrica **principal**.

Se calculan MAE, RMSE y error máximo (uniforme y empírico), percentiles del
error empírico (P95, P99), error dentro y fuera del intervalo, contribución de
las colas, error cerca del origen (ventana [-0.5, 0.5], malla de 2001 puntos
como principal y muestras empíricas como complemento), y diagnósticos de
estabilidad numérica.

### Jerarquía de métricas

1. MAE empírico (criterio principal)
2. RMSE empírico
3. Percentiles del error empírico
4. Error uniforme (diagnóstico)
5. Error máximo

### Validez vs. viabilidad práctica

Una configuración se marca **inválida** solo por no-finitud (NaN, infinito,
desbordamiento, forma incompatible, arreglo vacío). Un error alto pero finito
es un resultado **válido**. La **viabilidad práctica** se registra en columna
separada: una configuración es prácticamente no viable si su salida máxima
supera 10^6 (umbral operativo provisional, no un límite CKKS definitivo).

## Resultados

De las 72 configuraciones: **72 válidas, 0 inválidas, 4 no viables** en la
práctica. Las cuatro no viables son Taylor grado 9 en las activaciones de
rango amplio: `act2_taylor_d9_I1`, `act2_taylor_d9_I2`, `act3_taylor_d9_I1`,
`act3_taylor_d9_I2`.

### Comparación de métodos (grado 5, I1)

| Configuración      | MAE empírico | MAE uniforme | bias_ratio |
| ------------------ | ------------ | ------------ | ---------- |
| act1 chebyshev     | 0.2047       | 0.1013       | 0.50       |
| act1 least_squares | 0.1877       | 0.1002       | 0.53       |
| act1 taylor        | 0.7374       | 1.6818       | 2.28       |
| act2 chebyshev     | 0.2618       | 0.1500       | 0.57       |
| act2 least_squares | 0.2559       | 0.1493       | 0.58       |
| act2 taylor        | 15.5838      | 76.2554      | 4.89       |
| act3 chebyshev     | 0.2507       | 0.1852       | 0.74       |
| act3 least_squares | 0.2395       | 0.1808       | 0.75       |
| act3 taylor        | 11.6943      | 41.6725      | 3.56       |

### Hallazgos

1. **Chebyshev y mínimos cuadrados aproximan mucho mejor que Taylor.** En
   error empírico, los métodos globales dan ~0.2 mientras Taylor da 11–15
   (~50–75× peor). Taylor cumple su rol de baseline local desfavorable.

2. **Mínimos cuadrados es ligeramente mejor que Chebyshev en error empírico**
   en las tres activaciones. Como las preactivaciones reales se concentran
   cerca del origen y no en los extremos que la proyección de Chebyshev
   prioriza, el ajuste uniforme resulta marginalmente más ajustado sobre la
   distribución real. La diferencia es pequeña pero consistente; los métodos
   producen coeficientes distintos.

3. **El bias_ratio revela dos regímenes opuestos según el método:**
   - Chebyshev/LSQ: bias_ratio < 1 (empírico > uniforme). Estos métodos
     aproximan bien casi todo el intervalo, pero los datos reales se
     concentran cerca del origen, la región de mayor error (el codo de ReLU),
     elevando el error empírico.
   - Taylor: bias_ratio > 1 (uniforme > empírico). Taylor falla gravemente en
     los extremos del intervalo, pero los datos reales rara vez llegan tan
     lejos; la malla uniforme penaliza esas colas casi vacías.

   El signo de la diferencia entre uniforme y empírico depende de dónde falla
   cada método frente a dónde se concentran los datos. Esto justifica medir
   ambas métricas y priorizar la empírica.

4. **Taylor grado 9 se vuelve prácticamente inviable en act2 y act3.** La
   combinación de intervalo amplio y grado alto produce salidas que superan
   10^6. En act1 (rango estrecho) Taylor sigue siendo viable incluso en grado 9. Ninguna de las cuatro configuraciones es numéricamente inválida: todas
   producen valores finitos y permanecen en la tabla como evidencia.

## Configuraciones inválidas

Ninguna. Las 72 configuraciones producen salidas finitas. Las cuatro
marcadas como no viables (Taylor grado 9 en act2/act3) son válidas pero
superan el umbral operativo de magnitud.

## Limitaciones

- El error cerca del origen usa la malla uniforme como métrica principal para
  garantizar comparabilidad entre las 72 configuraciones; la variante empírica
  depende de la masa muestral de cada activación en la ventana [-0.5, 0.5].
- El umbral de viabilidad práctica (10^6) es operativo y provisional. La
  viabilidad real bajo CKKS se determinará en los Hitos 4 y 5.

## Criterio de cierre

Los 72 polinomios han sido evaluados frente a ReLU sobre malla uniforme y
sobre preactivaciones reales de validación; se calcularon MAE, RMSE, error
máximo, error dentro y fuera del intervalo y error cerca del origen; los
resultados están exportados de forma reproducible (CSV, JSON, manifiesto); y
las configuraciones prácticamente inviables están identificadas sin utilizar
el conjunto de prueba.

**Estado: Hito 3B completo.**
