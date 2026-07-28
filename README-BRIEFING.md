# Briefing del Proyecto — Documento Rector

> Documento vivo. Se actualiza al cerrar cada hito y constituye el punto de reentrada al proyecto.

---

## 1. Identidad del proyecto

**Nombre del repositorio:**  
`ckks-polynomial-inference-benchmark`

**Título aprobado (no modificable):**  
**Acoplamiento entre aproximación polinómica de activaciones y presupuesto de ruido RLWE: caracterización empírica del trade-off precisión–profundidad–latencia en inferencia segura con CKKS.**

**Curso:**  
Criptografía — Maestría en Ciberseguridad, Universidad Nacional Mayor de San Marcos.

**Autor:**  
Jhon Capcha

**Repositorio:**  
https://github.com/jhon-capcha/ckks-polynomial-inference-benchmark

**Resumen en una frase:**  
Estudio experimental de benchmarking que compara tres métodos de aproximación polinómica de la activación ReLU, midiendo cómo cada uno afecta a la precisión, la profundidad multiplicativa y la latencia durante la inferencia sobre datos cifrados con CKKS.

**Estado general:**  
🟡 En desarrollo

---

## 2. Resumen ejecutivo

El proyecto implementa un entorno experimental reproducible para estudiar el acoplamiento entre la aproximación polinómica de funciones de activación y la capacidad operativa del esquema CKKS durante la inferencia segura de una red neuronal convolucional.

La red base será una CNN pequeña tipo LeNet reducida, entrenada previamente en texto plano sobre MNIST. La activación original ReLU será sustituida por aproximaciones polinómicas obtenidas mediante tres métodos:

- Taylor.
- Chebyshev.
- Mínimos cuadrados.

Cada método será evaluado para los grados 3, 5, 7 y 9, sobre dos intervalos derivados de los percentiles de las preactivaciones del modelo. El diseño factorial principal comprende:

`3 métodos × 4 grados × 2 intervalos = 24 configuraciones`

Para cada configuración se medirán tres dimensiones:

1. **Precisión:** accuracy y F1 en tres niveles de evaluación.
2. **Profundidad:** profundidad multiplicativa teórica y niveles de la cadena consumidos.
3. **Latencia:** tiempo de evaluación de la activación y tiempo de inferencia completa.

El objetivo final es caracterizar empíricamente el trade-off precisión–profundidad–latencia y construir una guía de configuraciones eficientes mediante análisis de Pareto.

---

## 3. Objetivo del laboratorio

Construir un entorno reproducible que permita ejecutar, instrumentar y comparar cada configuración de aproximación polinómica definida por:

`método × grado × intervalo`

El entorno deberá medir, de forma separada y trazable:

- La pérdida debida a la aproximación matemática de ReLU.
- La degradación adicional introducida por la aritmética aproximada de CKKS.
- La profundidad multiplicativa teórica de la evaluación.
- Los niveles de la cadena de módulos efectivamente consumidos.
- La evolución de la escala y del estado del ciphertext.
- La latencia de la activación y de la inferencia completa.

**Entregable final:**  
Una guía de configuraciones eficientes, sustentada por resultados reproducibles y por un frente de Pareto precisión–profundidad–latencia.

---

## 4. Alcance

### 4.1 Incluido

- Dataset MNIST.
- CNN pequeña tipo LeNet reducida.
- Entrenamiento del modelo en texto plano.
- Activación original ReLU.
- Sustitución de ReLU por aproximaciones polinómicas.
- Métodos Taylor, Chebyshev y mínimos cuadrados.
- Grados 3, 5, 7 y 9.
- Dos intervalos de aproximación, I1 e I2.
- Intervalos derivados de percentiles de preactivaciones.
- Evaluación polinómica con Paterson–Stockmeyer.
- Entrada del cliente cifrada mediante CKKS.
- Pesos y sesgos del modelo mantenidos en claro.
- Pyfhel como interfaz Python sobre Microsoft SEAL.
- Medición de accuracy y F1.
- Medición del error funcional de aproximación.
- Medición del error numérico introducido por CKKS.
- Cálculo de profundidad multiplicativa teórica.
- Instrumentación de niveles consumidos, niveles restantes, escala y `parms_id`.
- Medición de latencia por activación e inferencia completa.
- Repeticiones para análisis estadístico.
- Construcción de un frente de Pareto.
- Ejecución sobre Windows 11 físico, sin GPU y sin máquinas virtuales.

### 4.2 Fuera de alcance

- Entrenamiento homomórfico.
- Bootstrapping ejecutado o medido.
- Comparación con BFV, BGV o TFHE.
- Comparación entre distintas estrategias de evaluación polinómica.
- GPU, FPGA, ASIC o aceleradores especializados.
- Entornos de Ejecución Confiables (TEE).
- Aprendizaje federado.
- Redes neuronales profundas.
- Datasets distintos de MNIST.
- Optimización extrema de bajo nivel.
- Implementación directa sobre Microsoft SEAL en C++.
- Comparación de múltiples librerías FHE.
- Inferencia distribuida o multiusuario.

---

## 5. Preguntas de investigación

### RQ1

¿De qué manera el método, el grado y el intervalo de aproximación polinómica impactan en el consumo de niveles de la cadena de módulos y en la evolución del error numérico de CKKS durante la evaluación de las funciones de activación?

### RQ2

¿Cuál es la relación cuantitativa entre el grado del polinomio de aproximación, la profundidad multiplicativa del circuito homomórfico y la pérdida de precisión de clasificación del modelo bajo inferencia segura, y qué trade-off define entre estas dimensiones?

### RQ3

¿Cómo varía la latencia de la inferencia cifrada —evaluación de la activación e inferencia completa— según el método y el grado de aproximación, y en qué medida las configuraciones de menor consumo de niveles amplían la profundidad evaluable dentro de una cadena de módulos fija antes de requerir un refresco del ciphertext?

---

## 6. Decisiones cerradas

Estas decisiones no deben reabrirse salvo que aparezca un error técnico grave, una incompatibilidad demostrada o una observación académica que obligue a corregir el diseño.

| Elemento                 | Decisión                                                      |
| ------------------------ | ------------------------------------------------------------- |
| Dataset                  | MNIST                                                         |
| Modelo                   | CNN pequeña tipo LeNet reducida                               |
| Entrenamiento            | En texto plano                                                |
| Activación original      | ReLU                                                          |
| Métodos                  | Taylor, Chebyshev y mínimos cuadrados                         |
| Grados                   | 3, 5, 7 y 9                                                   |
| Intervalos               | Dos: I1 e I2                                                  |
| Criterio de intervalos   | Percentiles de preactivaciones, aproximadamente 99 % y 99.9 % |
| Configuraciones          | 3 × 4 × 2 = 24                                                |
| Repeticiones             | Sí, número por definir después de pruebas piloto              |
| Evaluación polinómica    | Paterson–Stockmeyer                                           |
| Estrategia de evaluación | Fija para todas las configuraciones                           |
| Bootstrapping            | No se ejecuta ni se mide                                      |
| Capacidad restante       | Se estima mediante niveles restantes                          |
| Esquema                  | CKKS                                                          |
| Librería                 | Pyfhel                                                        |
| Backend                  | Microsoft SEAL                                                |
| Sistema operativo        | Windows 11 físico                                             |
| Editor                   | Visual Studio Code                                            |
| GPU                      | No                                                            |
| Máquina virtual          | No                                                            |
| Datos cifrados           | Entrada del cliente                                           |
| Parámetros del modelo    | Pesos y sesgos en claro                                       |
| Reproducibilidad         | Configuración externa, manifests y commits trazables          |
| Fuente de configuración  | `configs/*.yaml`                                              |
| Fuente de dependencias   | `pyproject.toml` después de validar versiones                 |
| Estructura Python        | `src/ckks_benchmark/`                                         |

---

## 7. Marco conceptual crítico

### 7.1 CKKS no expone un noise budget directo en bits

CKKS no debe tratarse como BFV o BGV respecto al presupuesto de ruido.

No se utilizará:

```python
invariant_noise_budget()
```

El denominado presupuesto de ruido RLWE se operacionalizará indirectamente mediante indicadores observables:

- Nivel actual del ciphertext.
- `parms_id`.
- Posición dentro de la cadena de módulos.
- Número de primos restantes.
- Bits disponibles en el `coeff_modulus`.
- Escala del ciphertext.
- Error numérico después de descifrar y decodificar.
- Viabilidad de decodificación respecto a un umbral definido.

### 7.2 Tres errores distintos

No deben mezclarse:

1. **Error de aproximación polinómica**  
   Diferencia entre ReLU y el polinomio en texto plano.

2. **Error numérico de CKKS**  
   Diferencia entre la evaluación polinómica en claro y la evaluación homomórfica después de descifrar.

3. **Consumo de niveles**  
   Reducción de la capacidad operativa de la cadena de módulos.

### 7.3 Precisión en tres niveles

La precisión se medirá en:

1. Red original con ReLU en claro.
2. Red con activación polinómica en claro.
3. Red con activación polinómica bajo CKKS.

Esto permite separar:

- `Δ_aproximación`
- `Δ_CKKS`

### 7.4 Profundidad no equivale a número de operaciones

La profundidad multiplicativa es la longitud de la ruta crítica de multiplicaciones dependientes.

No es:

- El número total de multiplicaciones.
- El número total de operaciones.
- La suma de todas las ramas del circuito.

También debe distinguirse de los niveles consumidos observados.

### 7.5 Taylor es un baseline deliberadamente desfavorable

ReLU no es diferenciable en `x = 0`. Por tanto, no existe una serie de Taylor clásica de ReLU alrededor del origen.

Taylor se implementará mediante una definición operacional reproducible, que deberá:

- Estar documentada en `docs/metodologia/definicion_taylor.md`.
- Incluir explícitamente el procedimiento de construcción.
- Mantener el origen dentro del dominio evaluado.
- Preservar la limitación local característica del método.
- No presentarse como una serie de Taylor clásica de ReLU en cero.

### 7.6 La relación grado–precisión no es monotónica

Un grado mayor no garantiza automáticamente:

- Menor error de aproximación.
- Mayor accuracy.
- Menor error numérico.
- Mayor profundidad.
- Menor latencia.

La interacción entre método, grado, intervalo y estructura algebraica puede producir comportamientos irregulares. Esa irregularidad constituye parte de la justificación experimental.

### 7.7 Paterson–Stockmeyer es una variable controlada

La estrategia de evaluación polinómica se mantiene fija para evitar confundir:

- La calidad del método de aproximación.
- La estructura del polinomio.
- La estrategia de reutilización de potencias.
- La política de relinearización.
- La política de `rescale`.
- El algoritmo de evaluación.

---

## 8. Diseño experimental

### 8.1 Factores principales

| Factor    | Niveles                              |
| --------- | ------------------------------------ |
| Método    | Taylor, Chebyshev, mínimos cuadrados |
| Grado     | 3, 5, 7, 9                           |
| Intervalo | I1, I2                               |

### 8.2 Número de configuraciones

`3 × 4 × 2 = 24 configuraciones`

### 8.3 Unidad experimental

Una unidad experimental queda definida por:

`método + grado + intervalo + parámetros CKKS + repetición`

### 8.4 Repeticiones

Cada configuración se ejecutará con un número fijo de repeticiones para estimar la distribución de la latencia.

El número definitivo se fijará después de las pruebas piloto del Hito 4.

### 8.5 Warm-up

Antes de registrar latencias se ejecutarán iteraciones de calentamiento para reducir el efecto de:

- Carga inicial de módulos.
- Creación de objetos internos.
- Cachés.
- Inicialización del entorno.
- Primeras asignaciones de memoria.

El número de iteraciones de warm-up se definirá durante las pruebas piloto.

### 8.6 Orden de ejecución

El orden de las configuraciones deberá ser controlado o aleatorizado para reducir sesgos temporales provocados por:

- Variaciones de carga del sistema.
- Procesos en segundo plano.
- Estado térmico del equipo.
- Cachés.
- Fragmentación de memoria.

### 8.7 Manifest por ejecución

Cada ejecución deberá registrar como mínimo:

- `run_id`
- Fecha y hora.
- Commit de Git.
- Rama.
- Versión de Python.
- Versión de Pyfhel.
- Versión de PyTorch.
- Sistema operativo.
- CPU.
- Memoria.
- Seed.
- Método.
- Grado.
- Intervalo.
- Repetición.
- Parámetros CKKS.
- Estado de ejecución.
- Ruta de salida.
- Hash o identificador de configuración.

---

## 9. Variables experimentales

### 9.1 Variables independientes

- Método de aproximación.
- Grado polinómico.
- Intervalo de aproximación.

### 9.2 Variables dependientes

- Accuracy con ReLU en claro.
- F1 con ReLU en claro.
- Accuracy con polinomio en claro.
- F1 con polinomio en claro.
- Accuracy con polinomio bajo CKKS.
- F1 con polinomio bajo CKKS.
- MAE de aproximación.
- RMSE de aproximación.
- Error absoluto máximo.
- Error numérico CKKS.
- Profundidad multiplicativa teórica.
- Niveles consumidos.
- Niveles restantes.
- Escala del ciphertext.
- Latencia de evaluación de la activación.
- Latencia de inferencia completa.
- Tamaño del ciphertext, si la API permite medirlo de forma estable.

### 9.3 Variables controladas

- Dataset.
- Partición de entrenamiento, validación y prueba.
- Arquitectura de la CNN.
- Pesos y sesgos del modelo.
- Activación original.
- Estrategia Paterson–Stockmeyer.
- Política de cálculo y reutilización de potencias.
- Política de relinearización.
- Política de `rescale`.
- Política de alineación de escalas.
- Política de `mod_switch`.
- Parámetros CKKS dentro de cada comparación.
- Hardware.
- Sistema operativo.
- Entorno Python.
- Seeds.
- Tamaño de muestra.
- Procedimiento de medición.
- Número de warm-ups.
- Número de repeticiones.
- Carga del sistema, en la medida de lo posible.

---

## 10. Métricas

### 10.1 Precisión de clasificación

Se medirá accuracy y F1 en tres niveles:

1. `ReLU_claro`
2. `Polinomio_claro`
3. `Polinomio_CKKS`

### 10.2 Pérdida por aproximación

```text
Δ_aproximación = métrica(ReLU_claro) − métrica(Polinomio_claro)
```

### 10.3 Pérdida adicional por CKKS

```text
Δ_CKKS = métrica(Polinomio_claro) − métrica(Polinomio_CKKS)
```

### 10.4 Error funcional de aproximación

Se calcularán:

- MAE.
- RMSE.
- Error absoluto máximo.

Comparación:

```text
ReLU(x) vs. P(x)
```

### 10.5 Error numérico de CKKS

Se comparará:

```text
P(x) en claro vs. Decodificar(Descifrar(Evaluar_CKKS(P, x)))
```

### 10.6 Profundidad

Se registrarán por separado:

- Profundidad multiplicativa teórica.
- Ruta crítica de multiplicaciones dependientes.
- Niveles consumidos observados.
- Niveles restantes.
- Diferencia entre profundidad prevista y comportamiento observado.

No se sumarán en una única métrica.

### 10.7 Estado criptográfico

Cuando la API lo permita, se registrarán:

- `parms_id`.
- Nivel actual.
- Número de módulos primos restantes.
- Bits disponibles del `coeff_modulus`.
- Escala.
- Tamaño del ciphertext.
- Estado antes y después de cada operación relevante.

### 10.8 Latencia

Se medirá:

- Latencia de la activación.
- Latencia de la inferencia completa.
- Tiempo por repetición.
- Media.
- Mediana.
- Desviación estándar.
- Mínimo.
- Máximo.
- Percentiles relevantes.

---

## 11. Hitos técnicos del desarrollo

### Hito previo — Infraestructura del repositorio

**Objetivo:**  
Preparar el repositorio, la estructura de carpetas, Visual Studio Code y GitHub.

**Actividades:**

- Crear estructura profesional del proyecto.
- Configurar `.vscode/settings.json`.
- Instalar extensiones necesarias.
- Inicializar Git.
- Crear repositorio remoto.
- Definir `.gitignore`.
- Establecer el documento rector.

**Criterio de cierre:**

- Repositorio local y remoto operativos.
- Estructura versionada.
- Working tree limpio.
- Briefing creado y aprobado.

**Estado:**  
✅ Completo

---

### Hito 0 — Verificación del entorno

**Objetivo:**  
Validar que el entorno Windows 11 puede ejecutar el stack requerido, especialmente Pyfhel.

**Actividades:**

- Confirmar versión de Python.
- Confirmar arquitectura de Python.
- Identificar ejecutables disponibles.
- Crear `.venv`.
- Instalar dependencias mínimas.
- Instalar Pyfhel mediante el mecanismo compatible disponible.
- Ejecutar un smoke test de CKKS.
- Cifrar y descifrar un valor.
- Validar codificación y decodificación.
- Documentar el procedimiento.
- Fijar versiones compatibles.

**Archivos principales:**

- `docs/instalacion/decision_libreria.md`
- `pyproject.toml`
- `requirements.txt`
- `tests/test_homomorphic.py`

**Criterio de cierre:**

Pyfhel cifra y descifra correctamente un valor de prueba en el entorno del proyecto.

**Estado:**  
⬜ Pendiente

---

### Hito 1 — Modelo base sobre MNIST

**Objetivo:**  
Entrenar y validar la CNN base con ReLU en texto plano.

**Actividades:**

- Definir LeNet reducida.
- Descargar MNIST.
- Preparar particiones.
- Fijar seeds.
- Entrenar el modelo.
- Guardar pesos.
- Medir accuracy y F1.
- Documentar arquitectura.
- Registrar configuración de entrenamiento.

**Archivos principales:**

- `src/ckks_benchmark/model/architecture.py`
- `src/ckks_benchmark/model/train.py`
- `models/`
- `configs/experiment.yaml`

**Criterio de cierre:**

Modelo entrenado, pesos guardados y línea base ReLU registrada de manera reproducible.

**Estado:**  
⬜ Pendiente

---

### Hito 2 — Preactivaciones e intervalos

**Objetivo:**  
Obtener la distribución de preactivaciones y fijar I1 e I2.

**Actividades:**

- Insertar hooks en las capas relevantes.
- Extraer preactivaciones.
- Generar estadísticas descriptivas.
- Calcular percentiles.
- Definir I1.
- Definir I2.
- Visualizar distribuciones.
- Registrar valores definitivos.

**Archivos principales:**

- `src/ckks_benchmark/model/preactivations.py`
- `notebooks/01_explore_preactivations.ipynb`
- `configs/experiment.yaml`
- `results/figures/`

**Criterio de cierre:**

Valores concretos de I1 e I2 documentados, reproducibles y derivados de la distribución observada.

**Estado:**  
⬜ Pendiente

---

### Hito 3 — Métodos de aproximación

**Objetivo:**  
Implementar y validar Taylor, Chebyshev y mínimos cuadrados.

**Actividades:**

- Definir interfaz común.
- Implementar Taylor operacional.
- Documentar su definición.
- Implementar Chebyshev.
- Implementar mínimos cuadrados.
- Convertir coeficientes a una base común.
- Evaluar grados 3, 5, 7 y 9.
- Evaluar ambos intervalos.
- Calcular MAE, RMSE y error máximo.
- Generar gráficas.

**Archivos principales:**

- `src/ckks_benchmark/approximation/base.py`
- `src/ckks_benchmark/approximation/taylor.py`
- `src/ckks_benchmark/approximation/chebyshev.py`
- `src/ckks_benchmark/approximation/least_squares.py`
- `docs/metodologia/definicion_taylor.md`
- `tests/test_approximation.py`
- `notebooks/02_test_approximations.ipynb`

**Criterio de cierre:**

Los tres métodos generan coeficientes reproducibles para los grados 3, 5, 7 y 9 en ambos intervalos, y sus errores funcionales quedan registrados.

**Estado:**  
⬜ Pendiente

---

### Hito 4 — Evaluación homomórfica

**Objetivo:**  
Configurar CKKS e implementar la evaluación cifrada de polinomios.

**Actividades:**

- Definir parámetros CKKS preliminares.
- Estimar profundidad requerida.
- Configurar contexto.
- Generar claves.
- Implementar Paterson–Stockmeyer.
- Definir política de potencias.
- Definir política de relinearización.
- Definir política de `rescale`.
- Implementar instrumentación.
- Implementar cálculo teórico de profundidad.
- Comparar evaluación clara y cifrada.
- Ejecutar pruebas piloto.
- Fijar warm-ups y repeticiones.
- Fijar tamaño de muestra.
- Fijar umbral de viabilidad numérica.

**Archivos principales:**

- `configs/ckks.yaml`
- `src/ckks_benchmark/homomorphic/context.py`
- `src/ckks_benchmark/homomorphic/evaluator.py`
- `src/ckks_benchmark/homomorphic/instrumentation.py`
- `src/ckks_benchmark/homomorphic/depth.py`
- `tests/test_homomorphic.py`
- `tests/test_depth.py`

**Criterio de cierre:**

Un polinomio se evalúa correctamente bajo CKKS y se registran estado criptográfico, error numérico, profundidad y latencia.

**Estado:**  
⬜ Pendiente

---

### Hito 5 — Benchmarking

**Objetivo:**  
Ejecutar el diseño experimental completo.

**Actividades:**

- Implementar runner.
- Generar matriz de 24 configuraciones.
- Ejecutar warm-ups.
- Ejecutar repeticiones.
- Registrar métricas.
- Registrar manifests.
- Detectar ejecuciones fallidas.
- Validar completitud.
- Consolidar resultados crudos.
- Generar resultados procesados.

**Archivos principales:**

- `src/ckks_benchmark/experiment/runner.py`
- `src/ckks_benchmark/experiment/metrics.py`
- `results/raw/`
- `results/processed/`
- `results/manifests/`

**Criterio de cierre:**

Las 24 configuraciones se ejecutan con el número de repeticiones definido, sin registros incompletos, y cada ejecución dispone de métricas y manifest de trazabilidad.

**Estado:**  
⬜ Pendiente

---

### Hito 6 — Análisis y trade-off

**Objetivo:**  
Interpretar los resultados y responder las preguntas de investigación.

**Actividades:**

- Construir frente de Pareto.
- Generar figuras finales.
- Comparar métodos.
- Comparar grados.
- Comparar intervalos.
- Analizar precisión.
- Analizar profundidad.
- Analizar niveles.
- Analizar latencia.
- Analizar errores.
- Responder RQ1.
- Responder RQ2.
- Responder RQ3.
- Elaborar guía de selección.

**Archivos principales:**

- `src/ckks_benchmark/experiment/pareto.py`
- `notebooks/03_analyze_results.ipynb`
- `results/figures/`
- `results/processed/`

**Criterio de cierre:**

Frente de Pareto generado, configuraciones eficientes identificadas y preguntas de investigación respondidas con evidencia reproducible.

**Estado:**  
⬜ Pendiente

---

## 12. Estructura del repositorio

```text
CKKS/
│
├── README.md
├── README-BRIEFING.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── estructura_proyecto.ps1
│
├── .vscode/
│   └── settings.json
│
├── configs/
│   ├── experiment.yaml
│   └── ckks.yaml
│
├── src/
│   └── ckks_benchmark/
│       ├── __init__.py
│       ├── config.py
│       │
│       ├── model/
│       │   ├── __init__.py
│       │   ├── architecture.py
│       │   ├── train.py
│       │   └── preactivations.py
│       │
│       ├── approximation/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── taylor.py
│       │   ├── chebyshev.py
│       │   └── least_squares.py
│       │
│       ├── homomorphic/
│       │   ├── __init__.py
│       │   ├── context.py
│       │   ├── evaluator.py
│       │   ├── instrumentation.py
│       │   └── depth.py
│       │
│       └── experiment/
│           ├── __init__.py
│           ├── runner.py
│           ├── metrics.py
│           └── pareto.py
│
├── notebooks/
│   └── .gitkeep
│
├── data/
│   └── .gitkeep
│
├── models/
│   └── .gitkeep
│
├── results/
│   ├── raw/
│   ├── processed/
│   ├── figures/
│   └── manifests/
│
├── tests/
│   ├── test_approximation.py
│   ├── test_homomorphic.py
│   └── test_depth.py
│
└── docs/
    ├── articulo/
    ├── figuras/
    ├── metodologia/
    │   └── definicion_taylor.md
    └── instalacion/
        └── decision_libreria.md
```

---

## 13. Forma de trabajo

- Se trabaja por hitos.
- No se avanza al siguiente hito sin cumplir el criterio de cierre del actual.
- Se explica primero el porqué y después el cómo.
- Toda decisión metodológica queda documentada.
- Los parámetros científicos viven en `configs/*.yaml`.
- El código fuente vive en `src/ckks_benchmark/`.
- Los notebooks se usan para exploración y análisis, no como única implementación.
- Las pruebas se incorporan progresivamente.
- No se fijan versiones ni parámetros criptográficos sin validación.
- Los resultados no se consideran válidos sin manifest.
- La claridad tiene prioridad sobre la optimización prematura.
- Toda desviación del alcance debe documentarse y justificarse.
- El repositorio debe poder reconstruirse desde cero.

---

## 14. Estrategia de commits

### 14.1 Principio general

Los commits deben ser pequeños, coherentes y trazables.

Cada commit debe representar una unidad lógica de trabajo y dejar el proyecto en un estado consistente.

### 14.2 Convención

Formato:

```text
tipo: descripción breve
```

### 14.3 Tipos

| Tipo       | Uso                                        |
| ---------- | ------------------------------------------ |
| `feat`     | Nueva funcionalidad                        |
| `fix`      | Corrección de errores                      |
| `docs`     | Documentación                              |
| `test`     | Pruebas                                    |
| `refactor` | Reorganización sin cambio funcional        |
| `build`    | Dependencias, empaquetado y entorno        |
| `config`   | Configuración experimental o criptográfica |
| `chore`    | Mantenimiento                              |
| `results`  | Resultados experimentales validados        |

### 14.4 Reglas

1. No mezclar trabajo no relacionado en un mismo commit.
2. No subir `.venv`, datasets, cachés ni modelos regenerables.
3. No esperar al cierre completo de un hito para realizar commits.
4. Cada unidad lógica debe versionarse.
5. El cierre de un hito debe actualizar este briefing.
6. Los resultados solo se versionan cuando son válidos.
7. Cada ejecución publicada debe tener manifest.
8. Los mensajes deben describir lo que cambia.
9. No usar mensajes genéricos como `cambios`, `update` o `fix`.
10. Antes de cada commit se ejecutará:
    - Ruff.
    - Pruebas aplicables.
    - Revisión de `git status`.
    - Revisión del diff.

### 14.5 Ejemplos por fase

#### Infraestructura

```text
chore: initialize CKKS project structure
config: add project-specific VS Code settings
docs: add project briefing and milestone roadmap
```

#### Hito 0

```text
build: add validated Python environment
build: add compatible Pyfhel dependency
test: add CKKS encryption smoke test
docs: record Pyfhel installation decision
docs: close environment validation milestone
```

#### Hito 1

```text
feat: implement reduced LeNet architecture
feat: add MNIST training pipeline
test: validate model output dimensions
results: record baseline ReLU accuracy
docs: close baseline model milestone
```

#### Hito 2

```text
feat: add preactivation extraction hooks
feat: compute activation percentile intervals
config: register I1 and I2 intervals
docs: close preactivation analysis milestone
```

#### Hito 3

```text
feat: implement approximation interface
feat: implement Chebyshev approximation
feat: implement least-squares approximation
feat: implement operational Taylor baseline
test: validate polynomial approximation metrics
docs: document Taylor baseline definition
docs: close approximation milestone
```

#### Hito 4

```text
feat: configure CKKS context
feat: implement Paterson-Stockmeyer evaluator
feat: add ciphertext instrumentation
feat: add multiplicative depth analysis
test: compare plaintext and CKKS polynomial evaluation
docs: close homomorphic evaluation milestone
```

#### Hito 5

```text
feat: implement experimental runner
feat: add execution manifests
results: add validated benchmark dataset
docs: close benchmarking milestone
```

#### Hito 6

```text
feat: compute Pareto frontier
results: add final trade-off figures
docs: document answers to research questions
docs: close final analysis milestone
```

### 14.6 Flujo de trabajo Git

```text
editar
→ revisar
→ probar
→ git status
→ git diff
→ git add
→ git commit
→ git push
```

---

## 15. Riesgos técnicos

| Riesgo                                 | Impacto | Tratamiento                                     |
| -------------------------------------- | ------- | ----------------------------------------------- |
| Incompatibilidad entre Python y Pyfhel | Alto    | Validar versiones en el Hito 0                  |
| Ausencia de wheel compatible           | Alto    | Evaluar compilación local o versión alternativa |
| Fallo de compilación en Windows        | Alto    | Verificar Build Tools y documentación           |
| Cadena de módulos insuficiente         | Alto    | Estimar profundidad antes de fijar parámetros   |
| Desalineación de niveles               | Alto    | Aplicar política uniforme                       |
| Escalas incompatibles                  | Alto    | Instrumentar y validar cada etapa               |
| Error numérico excesivo                | Alto    | Fijar umbral de viabilidad                      |
| Taylor demasiado competitivo           | Medio   | Controlar suavizado y validar el origen         |
| Taylor matemáticamente mal definido    | Alto    | Documentar definición operacional               |
| Latencia excesiva en CPU               | Medio   | Realizar pruebas piloto                         |
| Variabilidad de tiempos                | Medio   | Warm-up, repeticiones y control de carga        |
| Tamaño de muestra demasiado grande     | Medio   | Definir muestra tras prueba piloto              |
| Resultados incompletos                 | Medio   | Validación automática y manifests               |
| Duplicación de configuración           | Medio   | Usar YAML como fuente de verdad                 |
| Divergencia entre dependencias         | Medio   | Consolidar en `pyproject.toml`                  |
| Sobre-ingeniería                       | Medio   | Mantener alcance de proyecto individual         |
| Subida accidental de datos pesados     | Bajo    | Mantener `.gitignore`                           |
| Pérdida de trazabilidad                | Alto    | Registrar commit y configuración por ejecución  |

---

## 16. Decisiones pendientes

| Decisión                                    | Hito límite | Estado                                                                |
| ------------------------------------------- | ----------- | --------------------------------------------------------------------- |
| Versión definitiva de Python                | Hito 0      | Resuelto: Python 3.11.9 x64                                           |
| Versión compatible de Pyfhel                | Hito 0      | Resuelto: Pyfhel 3.5.0                                                |
| Mecanismo de instalación de Pyfhel          | Hito 0      | Resuelto: compilación desde fuente con Visual Studio Build Tools 2022 |
| Versiones de PyTorch y torchvision          | Hito 1      | Pendiente                                                             |
| Arquitectura exacta de LeNet reducida       | Hito 1      | Pendiente                                                             |
| Hiperparámetros de entrenamiento            | Hito 1      | Pendiente                                                             |
| Seed definitiva                             | Hito 1      | Pendiente                                                             |
| Valores concretos de I1 e I2                | Hito 2      | Pendiente                                                             |
| Capas cuyas preactivaciones se medirán      | Hito 2      | Pendiente                                                             |
| Definición operacional de Taylor            | Hito 3      | Pendiente                                                             |
| Parámetro de suavizado o punto de expansión | Hito 3      | Pendiente                                                             |
| Base común de coeficientes                  | Hito 3      | Pendiente                                                             |
| Parámetros CKKS definitivos                 | Hito 4      | Pendiente                                                             |
| `poly_modulus_degree`                       | Hito 4      | Pendiente                                                             |
| `coeff_mod_bit_sizes`                       | Hito 4      | Pendiente                                                             |
| Escala inicial                              | Hito 4      | Pendiente                                                             |
| Política exacta de rescale                  | Hito 4      | Pendiente                                                             |
| Número de warm-ups                          | Hito 4      | Pendiente                                                             |
| Número de repeticiones                      | Hito 4      | Pendiente                                                             |
| Tamaño de muestra cifrada                   | Hito 4      | Pendiente                                                             |
| Umbral de viabilidad numérica               | Hito 4      | Pendiente                                                             |
| Criterio de ejecución inválida              | Hito 5      | Pendiente                                                             |
| Formato final de resultados                 | Hito 5      | Pendiente                                                             |
| Criterio de dominancia de Pareto            | Hito 6      | Pendiente                                                             |

---

## 17. Estado del proyecto

| Fase                                      | Estado       | Fecha de cierre |
| ----------------------------------------- | ------------ | --------------- |
| Infraestructura: estructura, Git y GitHub | ✅ Completo  | 2026-07-27      |
| Briefing del proyecto                     | ✅ Completo  | 2026-07-27      |
| Hito 0 — Entorno                          | ✅ Completo  | 2026-07-27      |
| Hito 1 — CNN base                         | ⬜ Pendiente | —               |
| Hito 2 — Preactivaciones                  | ⬜ Pendiente | —               |
| Hito 3 — Aproximaciones                   | ⬜ Pendiente | —               |
| Hito 4 — CKKS                             | ⬜ Pendiente | —               |
| Hito 5 — Benchmarking                     | ⬜ Pendiente | —               |
| Hito 6 — Análisis                         | ⬜ Pendiente | —               |

### Leyenda

- ✅ Completo
- 🔄 En curso
- ⬜ Pendiente
- ⛔ Bloqueado
- ⚠️ Requiere revisión

---

## 18. Bitácora de decisiones

| Fecha      | Decisión                                                     | Justificación                                                                                                  |
| ---------- | ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| 2026-07-27 | Se adopta el patrón `src/ckks_benchmark/`                    | Evita utilizar `src` como nombre del paquete                                                                   |
| 2026-07-27 | Se separan `depth.py` e `instrumentation.py`                 | Distingue profundidad teórica de niveles observados                                                            |
| 2026-07-27 | Se incorporan manifests por ejecución                        | Garantiza trazabilidad experimental                                                                            |
| 2026-07-27 | Se crea `configs/`                                           | Separa parámetros científicos y criptográficos del código                                                      |
| 2026-07-27 | Se evita duplicar parámetros entre YAML y `config.py`        | Mantiene una única fuente de verdad                                                                            |
| 2026-07-27 | Se adopta `pyproject.toml` como fuente principal             | Reduce divergencia de dependencias y configuración                                                             |
| 2026-07-27 | Se mantiene `requirements.txt` sin versiones definitivas     | Las versiones se fijarán después del Hito 0                                                                    |
| 2026-07-27 | Se adopta Ruff                                               | Unifica linting, formato e imports                                                                             |
| 2026-07-27 | Se crea configuración de VS Code por proyecto                | Evita afectar otros repositorios                                                                               |
| 2026-07-27 | Se inicializa Git en la rama `main`                          | Establece trazabilidad desde el inicio                                                                         |
| 2026-07-27 | Se crea el repositorio `ckks-polynomial-inference-benchmark` | Identifica el artefacto experimental                                                                           |
| 2026-07-27 | Se establece Paterson–Stockmeyer como variable controlada    | Evita confundir método de aproximación y estrategia de evaluación                                              |
| 2026-07-27 | No se ejecutará bootstrapping                                | Se medirá capacidad restante mediante niveles                                                                  |
| 2026-07-27 | Taylor se considera baseline desfavorable                    | ReLU no es diferenciable en cero                                                                               |
| 2026-07-27 | Se separan tres tipos de error                               | Evita atribuciones incorrectas                                                                                 |
| 2026-07-27 | Se medirán tres niveles de precisión                         | Permite separar `Δ_aproximación` y `Δ_CKKS`                                                                    |
| 2026-07-27 | Se valida Python 3.11.9 x64 para el proyecto                 | Es la versión utilizada para compilar e instalar Pyfhel 3.5.0                                                  |
| 2026-07-27 | Se valida Pyfhel 3.5.0 sobre Windows 11                      | El smoke test CKKS completó cifrado, suma, multiplicación y descifrado con errores máximos inferiores a `1e-5` |
| 2026-07-27 | Se adopta compilación desde fuente para Pyfhel               | No existía wheel compatible para Windows y CPython 3.11                                                        |
| 2026-07-27 | Se fijan NumPy 2.4.6 y Pyfhel 3.5.0                          | Son las versiones verificadas durante el Hito 0                                                                |

---

## 19. Principios del proyecto

- Reproducibilidad por encima del rendimiento.
- Claridad por encima de optimización prematura.
- Una única fuente de verdad para cada configuración.
- Ninguna decisión metodológica debe quedar implícita.
- Todo experimento debe poder repetirse.
- Todo resultado debe estar asociado a un commit.
- Todo resultado debe registrar sus parámetros.
- Todo resultado publicado debe tener manifest.
- Ningún resultado se incorpora al artículo sin poder reconstruirse.
- Las comparaciones deben mantener constantes las variables de control.
- Los errores de aproximación, CKKS y niveles deben analizarse por separado.
- La profundidad teórica y los niveles observados no deben confundirse.
- El baseline Taylor debe documentarse con honestidad matemática.
- No se amplía el alcance sin justificación explícita.
- No se introducen tecnologías adicionales sin necesidad experimental.
- Los notebooks apoyan el análisis, pero el código reproducible vive en `src/`.

---

## 20. Criterio de finalización del proyecto

El proyecto se considerará finalizado cuando:

1. Las 24 configuraciones hayan sido ejecutadas.
2. Todas las configuraciones tengan el número de repeticiones definido.
3. Cada ejecución tenga manifest.
4. Existan métricas completas de precisión, profundidad y latencia.
5. Se separen `Δ_aproximación` y `Δ_CKKS`.
6. Se documenten los niveles consumidos y restantes.
7. Se genere el frente de Pareto.
8. Se respondan RQ1, RQ2 y RQ3.
9. Se publiquen las figuras finales.
10. El repositorio pueda reconstruirse desde cero.
11. El briefing refleje todas las decisiones finales.
12. El README público explique instalación, ejecución y resultados.
13. El código pase las pruebas definidas.
14. El estado de Git quede limpio.
15. El commit final esté publicado en GitHub.

---

## 21. Próximo paso

Retomar:

**Paso 0.1 — Diagnóstico del entorno**

Comandos previstos:

```powershell
python --version
py --version
py -0p
python -c "import sys, platform; print('Executable:', sys.executable); print('Architecture:', platform.architecture()); print('Platform:', platform.platform())"
pip --version
```

No instalar Pyfhel hasta revisar la salida completa.

---

## 22. Control de actualización del documento

Este briefing debe actualizarse cuando ocurra cualquiera de estos eventos:

- Se cierre un hito.
- Se adopte una decisión metodológica.
- Se modifique una configuración aprobada.
- Se identifique un riesgo nuevo.
- Se resuelva una decisión pendiente.
- Se cambie una dependencia crítica.
- Se modifique el diseño experimental.
- Se publique un resultado validado.
- Se detecte una desviación del alcance.

Cada actualización debe registrarse mediante un commit de documentación.

Ejemplo:

```text
docs: update project briefing after environment validation
```
