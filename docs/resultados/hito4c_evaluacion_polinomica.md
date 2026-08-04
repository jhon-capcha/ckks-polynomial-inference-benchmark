# Hito 4C — Evaluación homomórfica de polinomios

## Objetivo

Evaluar bajo CKKS los polinomios de aproximación de la shortlist congelada
(Hito 3A/3C), midiendo el error criptográfico adicional (Delta_CKKS) que
introduce el cifrado respecto de la evaluación polinómica clara.

Aquí se mide SOLO el error CKKS: p_claro(x) vs Dec(p(Enc(x))). El error de
aproximación (ReLU vs p) corresponde al Hito 3B.

## Estrategia oficial: Horner

Se congela **Horner** como estrategia oficial de evaluación:

    p(x) = a0 + x(a1 + x(a2 + ... ))

en base monomial, consumiendo directamente los coeficientes congelados del
Hito 3A sin conversion de base. Grados oficiales soportados: 3 y 5.

### Estrategia explorada y no adoptada: power_basis

Se exploro una evaluacion por potencias reutilizadas (base monomial, DAG de
baja profundidad). Resultados:

- Grado 3: funciono, pero consumio los mismos 3 niveles que Horner (sin ahorro).
- Grados 5 y 7: fallaron por incompatibilidad de escala al alinear ramas de
  distinta profundidad antes de sumarlas.

No se adopto porque no era necesaria para el alcance (grados 3 y 5), no
demostro reduccion de niveles, y su depuracion habria desviado el proyecto
hacia la optimizacion de circuitos CKKS. Queda registrada como exploratoria en
`polynomial_evaluator.py` (accesible solo con `allow_experimental=True`).

## Perfiles y profundidad

- Grado 3: perfil `ckks_n16384_d3` (profundidad 4), consume 3 niveles.
- Grado 5: perfil `ckks_n16384_d5` (profundidad 6), consume 5 niveles.

Horner consume exactamente el grado en niveles. El perfil d5 deja 1 nivel de
margen tras grado 5, suficiente para el bloque lineal final (fc2).

## Resultado principal: el error CKKS es despreciable

Se evaluaron los 18 polinomios oficiales (Chebyshev y Least Squares, grados 3
y 5, activaciones act1/act2/act3) sobre 200 muestras reales de preactivacion
por configuracion. Todas las evaluaciones fueron finitas.

| Metrica | Valor |
|---|---|
| MAE CKKS maximo | 1.57e-04 (act3_chebyshev_d5_I1) |
| MAE CKKS tipico grado 3 | ~3e-6 a 7e-6 |
| MAE CKKS tipico grado 5 | ~5e-6 a 1.6e-4 |
| Evaluaciones no finitas | 0 |

**El error CKKS es del orden de 1e-6 a 1e-4, aproximadamente 1000 veces menor
que el error de aproximacion del Hito 3B (~1e-1).** En la descomposicion
Delta_total = Delta_aproximacion + Delta_CKKS, el termino CKKS es marginal: la
perdida de precision en inferencia segura proviene casi enteramente de la
aproximacion polinomica, no del cifrado.

### Estructura del error CKKS

El error crece con el grado (mas multiplicaciones = mas rescales = mas error
acumulado) y con la profundidad de la activacion en la red (act3 > act2 >
act1), coherente con que act3 opera sobre rangos de preactivacion mas amplios,
donde los productos intermedios de Horner son mayores.

## Veredicto de grado 7

El grado 7 no resulto factible mediante Horner bajo el perfil seguro
`n16384_c8` (escala 2^40, seguridad de 128 bits, sin bootstrapping): requiere
~7 niveles y la cadena provee 6 (fallo "scale out of bounds").

No se afirma inviabilidad universal de grado 7 en CKKS. Podria llegar a ser
viable mediante N=32768, una estrategia de evaluacion de menor profundidad, una
cadena distinta compatible con la seguridad, o bootstrapping; todas fuera del
alcance principal del proyecto.

## Criterio de cierre

Horner congelado como estrategia oficial; grados 3 y 5 validados sobre muestras
reales; error CKKS cuantificado (~1e-4 maximo, todas finitas); grado 7
documentado como no factible bajo el perfil evaluado; power_basis documentado
como exploratorio; decision versionada en
`results/published/hito4_polynomial_strategy_decision.json`.

**Estado: Hito 4C completo.**
