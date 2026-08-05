# Hito 5 — Benchmarking

## Objetivo

Caracterizar sistematicamente el rendimiento de la inferencia cifrada del
bloque final (act3 -> fc2) validada en el Hito 4: latencia, huella de
almacenamiento y el trade-off precision-profundidad-latencia. El Hito 4
demostro correctitud y factibilidad; el Hito 5 mide el costo.

## Diseno experimental

- 6 configuraciones (Chebyshev/LSQ, grados 3 y 5, intervalos I1/I2).
- 10 imagenes (1 por clase MNIST, del subconjunto congelado del Hito 4).
- 30 repeticiones por imagen y configuracion + 3 de warm-up descartadas.
- Total: 1800 inferencias medidas.
- Orden de ejecucion aleatorizado (seed congelada) para evitar sesgo termico.
- Timer online AISLADO: solo encrypt/act3/fc2/decrypt. Fuera del timer: carga de
  datos, construccion de modelos, calculo del prefijo claro (z), serializacion.
- Contexto, claves y pesos reutilizados entre repeticiones (no regenerados).
- Setup criptografico medido por separado.

Estadistica jerarquizada: mediana y P95 como principales (robustas ante
interrupciones del SO); media, desviacion, P99 y coeficiente de variacion como
complementarias. El minimo no se usa como cifra principal. El residual temporal
(online_total - suma de etapas) fue 0, confirmando que no hay trabajo no
instrumentado entre etapas.

Taylor grado 5 se excluyo del benchmark: su estructura de circuito homomorfico
es identica a cualquier polinomio denso de grado 5, y su utilidad clasificatoria
ya se descarto en los Hitos 3 y 4.

## Latencia online

| Grado | Mediana | P95 |
|---|---|---|
| 3 | ~189 ms | ~204 ms |
| 5 | ~245 ms | ~260 ms |

El paso de grado 3 a grado 5 incremento la mediana de latencia online en
aproximadamente 30%. La varianza fue baja (P95 a ~3-8% de la mediana),
indicando mediciones estables.

### Desglose por etapa (mediana)

| Etapa | Grado 3 | Grado 5 | Fraccion (g3 / g5) |
|---|---|---|---|
| encrypt | 9.6 ms | 12.7 ms | 5.1% / 5.2% |
| act3 | 39.1 ms | 81.5 ms | 20.7% / 33.3% |
| fc2 | 131.9 ms | 142.0 ms | 69.9% / 58.0% |
| decrypt | 8.0 ms | 8.0 ms | 4.2% / 3.3% |

**fc2 domina la latencia** (58-70% del total), por las rotaciones del producto
matriz-vector (7 rotaciones por logit x 10 logits = 70 rotaciones,
independientes del grado). act3 es la segunda etapa.

### Origen del incremento de grado

El incremento grado 3 -> 5 (+56 ms) proviene casi enteramente de act3
(+42.4 ms, el 76%). fc2 apenas cambia (+10.1 ms), porque su circuito de
rotaciones es identico entre grados; el pequeno aumento se debe a operar sobre
ciphertexts en niveles mas profundos. encrypt y decrypt son practicamente
constantes.

Este es un resultado explicativo: el costo adicional de un polinomio de mayor
grado se concentra en la evaluacion de la activacion (Horner: grado 5 encadena
5 multiplicaciones vs 3 en grado 3), no en la capa lineal. Conecta directamente
con el eje profundidad-latencia del estudio.

### Independencia de metodo e intervalo

Para un mismo grado, Chebyshev y Least Squares presentaron latencias identicas
(diferencias < 0.5 ms, dentro del ruido), y lo mismo para I1 vs I2. El metodo y
el intervalo modifican la calidad de aproximacion, pero no la estructura del
circuito homomorfico ni su costo.

## Huella de almacenamiento y comunicacion

Se midieron tamanos serializados (no memoria RAM residente).

| Componente | Grado 3 (perfil d3) | Grado 5 (perfil d5) |
|---|---|---|
| Claves de rotacion | 171.6 MB | 312.2 MB |
| Clave de relinearizacion | 6.6 MB | 12.0 MB |
| Clave publica | 1.32 MB | 1.71 MB |
| Ciphertext de entrada | 1.31 MB | 1.84 MB |
| Ciphertexts de salida (10 logits) | 2.62 MB | 2.62 MB |

**Las claves de rotacion dominan la huella** (172-312 MB), muy por encima del
resto, lo que explica que su generacion domine el tiempo de setup. El perfil de
grado 5 requirio claves de rotacion aproximadamente 82% mas grandes y un
ciphertext de entrada 40% mayor (cadena de modulos mas larga).

La expansion de cifrado es notable: el vector de entrada de 120 valores
(960 bytes en claro) se expande ~1400-1900x al cifrarse; los 10 logits
(80 bytes) se expanden ~32800x, porque cada logit ocupa un ciphertext completo
(un polinomio de N=16384 para un solo valor util). Empaquetar los 10 logits en
un unico ciphertext reduciria la salida ~10x; queda como optimizacion futura.

Los tamanos dependen del perfil (grado), no del metodo ni el intervalo
(variacion < 0.01% entre configuraciones del mismo perfil, ruido de
serializacion aleatoria de las claves).

## Trade-off precision-profundidad-latencia

Uniendo precision (Hito 4), profundidad (niveles) y latencia (Hito 5):

| Configuracion | Accuracy | Latencia | Niveles | Pareto |
|---|---|---|---|---|
| chebyshev_d3_I1 | 0.950 | 188.7 ms | 4 | si |
| least_squares_d3_I1 | 0.960 | 189.0 ms | 4 | si |
| chebyshev_d5_I1 | 0.990 | 244.8 ms | 6 | no |
| least_squares_d5_I1 | 0.990 | 244.8 ms | 6 | si |
| chebyshev_d5_I2 | 0.970 | 244.4 ms | 6 | si |
| least_squares_d5_I2 | 0.980 | 244.8 ms | 6 | no |

### Frontera de Pareto

La frontera (maximizar accuracy, minimizar latencia) contiene cuatro
configuraciones no dominadas:

- **least_squares_d3_I1** (0.960, 189 ms): mejor accuracy entre las rapidas.
- **least_squares_d5_I1** (0.990, 245 ms): maxima accuracy, iguala a ReLU.
- **chebyshev_d3_I1** y **chebyshev_d5_I2**: puntos intermedios.

Anadir "niveles consumidos" como tercer objetivo no cambia la frontera: los
niveles estan perfectamente correlacionados con el grado, y el grado ya
determina la latencia, por lo que la latencia es un proxy de la profundidad.

### Lectura practica

- Maxima velocidad: least_squares_d3_I1 (0.960, 189 ms).
- Maxima precision: least_squares_d5_I1 (0.990, 245 ms), iguala a ReLU.
- Least Squares supera a Chebyshev en grado 3 (mejor accuracy a igual latencia)
  y empata en grado 5.

El costo de subir a grado 5 (~30% mas latencia, 82% mas claves de rotacion) se
paga por +4 puntos de accuracy (0.95 -> 0.99). La eleccion depende de si la
aplicacion prioriza latencia o precision.

## Criterio de cierre

1800 inferencias medidas con warm-up, repeticiones y orden aleatorizado;
desglose de latencia por etapa con residual cero; huella de almacenamiento de
claves y ciphertexts cuantificada; trade-off consolidado con frontera de Pareto
reproducible por codigo; sin reajuste de modelos ni parametros (congelados desde
Hitos 3 y 4).

**Estado: Hito 5 completo.**

El Hito 6 abordara el analisis final y la sintesis de resultados del proyecto.
