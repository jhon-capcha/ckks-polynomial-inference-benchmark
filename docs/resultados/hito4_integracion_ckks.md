# Hito 4 — Integracion CKKS

## Objetivo

Implementar y validar la ejecucion homomorfica del bloque final de la CNN
(activacion polinomica act3 seguida de la capa lineal fc2) bajo cifrado CKKS,
de modo que sea posible separar:

    Delta_total = Delta_aproximacion + Delta_CKKS

donde:
    Delta_aproximacion = CNN ReLU clara - CNN polinomica clara
    Delta_CKKS         = CNN polinomica clara - CNN polinomica con bloque cifrado

## Modelo de amenaza y alcance

- **Cliente**: posee la clave secreta. Cifra la preactivacion de act3 y
  descifra los logits.
- **Servidor**: solo claves publicas (publica, relin, rotacion) y los pesos en
  claro. Evalua act3 -> fc2 bajo cifrado, sin poder descifrar.

Alcance cifrado (oficial): bloque final act3 -> fc2. El prefijo de la red
(conv1, act1, pool, conv2, act2, pool, fc1) se ejecuta en claro. La CNN
completamente cifrada (con convoluciones homomorficas) y el bootstrapping
quedan fuera de alcance, por decision de enfoque hacia el eje de la tesis: las
activaciones polinomicas y su trade-off.

## Parametros CKKS

Perfiles congelados tras el piloto de factibilidad (4-P0):

- **d3** (`ckks_n16384_d3`): N=16384, cadena [60,40,40,40,40,60] (280 bits),
  escala 2^40, seguridad 128 bits, profundidad 4.
- **d5** (`ckks_n16384_d5`): N=16384, cadena [60,40,40,40,40,40,40,60]
  (360 bits), escala 2^40, seguridad 128 bits, profundidad 6.

La seguridad real (atributo `.sec`) se verifica >= 128 al crear cada contexto.
Una cadena de 440 bits produjo sec=0 (insegura) y quedo prohibida.

## Estrategia de evaluacion polinomica

Estrategia oficial: **Horner** (base monomial, sin conversion). Grados oficiales
soportados: 3 y 5. Horner consume el grado en niveles (3 y 5 respectivamente).

Se exploro una estrategia de potencias reutilizadas (power_basis), no adoptada:
no redujo el consumo de niveles en grado 3 y produjo incompatibilidades de
escala al alinear ramas en grados 5 y 7. Queda documentada como exploratoria.

## Empaquetado SIMD

La preactivacion de act3 (vector de 120) se empaqueta en los primeros 120 slots
de un unico ciphertext (padding a 128, potencia de 2, con ceros que no
contaminan). act3 se evalua sobre los 120 valores en paralelo con una sola
evaluacion Horner. fc2 usa producto matriz-vector: multiply_plain por cada fila
de pesos, reduccion de slots por rotate+add en arbol (7 rotaciones), y suma de
bias. Salida: un ciphertext por logit (10).

## Factibilidad del bloque final

El bloque act3 -> fc2 completa dentro de la cadena segura:
- grado 3: 3 (act3) + 1 (fc2) = 4 niveles, margen holgado.
- grado 5: 5 (act3) + 1 (fc2) = 6 niveles, usa toda la cadena, completa.

Las 6 configuraciones oficiales (Chebyshev/LSQ, grados 3 y 5, intervalos I1/I2)
resultaron factibles, sin valores no finitos.

El grado 7 no es factible con Horner bajo el perfil N=16384: requiere ~7 niveles
del polinomio mas fc2, y la cadena segura provee 6. No se afirma inviabilidad
universal; podria ser viable con N=32768, otra estrategia de menor profundidad,
o bootstrapping (fuera de alcance).

## Resultado principal: el cifrado es funcionalmente transparente

Validacion sobre 100 imagenes de test estratificadas (10 por clase, seed
congelada), tres rutas por imagen (ReLU clara, polinomica clara, bloque CKKS):

| Configuracion | ReLU | poly clara | CKKS | Delta_CKKS | concordancia |
|---|---|---|---|---|---|
| chebyshev_d5_I1 | 0.990 | 0.990 | 0.990 | 0.000 | 1.000 |
| least_squares_d5_I1 | 0.990 | 0.990 | 0.990 | 0.000 | 1.000 |
| least_squares_d5_I2 | 0.990 | 0.980 | 0.980 | 0.000 | 1.000 |
| chebyshev_d5_I2 | 0.990 | 0.970 | 0.970 | 0.000 | 1.000 |
| least_squares_d3_I1 | 0.990 | 0.960 | 0.960 | 0.000 | 1.000 |
| chebyshev_d3_I1 | 0.990 | 0.950 | 0.950 | 0.000 | 1.000 |

**Delta_CKKS = 0 en accuracy para las 6 configuraciones.** El cifrado del bloque
final no cambia ninguna prediccion en las 600 inferencias cifradas (concordancia
poly vs CKKS = 1.000). El error CKKS de logits es del orden de 5e-5 (grado 3) a
4e-4 (grado 5), muy por debajo del margen entre clases, por lo que no altera el
argmax.

Todo el error de clasificacion proviene de la aproximacion polinomica
(Delta_aproximacion, de 0 a 0.04 segun grado/metodo/intervalo), no del cifrado.
Las mejores configuraciones (grado 5, I1) igualan la accuracy de ReLU (0.990)
incluso bajo cifrado.

## Error CKKS por etapas (Hito 4C)

La evaluacion aislada de los 18 polinomios oficiales sobre muestras reales de
preactivacion mostro un error CKKS de 1e-6 a 1.6e-4 (MAE), aproximadamente 1000
veces menor que el error de aproximacion del Hito 3B (~1e-1). El error crece con
el grado y con la profundidad de la activacion en la red (act3 > act1).

## Disciplina anti-leakage

La seleccion de configuraciones se congelo en el Hito 3C (test_used=false). En
el Hito 4, test se usa solo para EVALUAR el bloque cifrado; no se reajustaron
polinomios, parametros CKKS, perfiles ni tolerancias con estos resultados. La
integridad se registra por hashes en `hito4_manifest.json`.

## Criterio de cierre

Contexto CKKS seguro (128 bits verificados); primitivas instrumentadas
(nivel/escala por operacion); evaluacion polinomica con Horner (grados 3 y 5);
bloque lineal final act3 -> fc2 cifrado y validado; flujo cliente-servidor con
el servidor sin clave secreta; Delta_CKKS cuantificado sobre imagenes reales
(= 0 en accuracy); configuraciones factibles identificadas; grado 7 documentado
como no factible bajo el perfil; estrategia power_basis documentada como
exploratoria; resultados reproducibles y versionados.

**Estado: Hito 4 completo.**

El Hito 5 abordara el benchmark sistematico de latencia, consumo y trade-offs.
