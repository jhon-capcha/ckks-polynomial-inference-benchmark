# Definición operacional de Taylor (Hito 3)

## Contexto

ReLU no es diferenciable en x=0 y no posee serie de Taylor clásica alrededor
del origen. Por ello, Taylor se implementa como baseline local desfavorable,
construido sobre una versión suavizada de ReLU.

## Construcción

- **Función suavizada:** Softplus_β(x) = (1/β)·ln(1 + e^(βx))
- **Punto de expansión:** x₀ = 0 (punto de transición de ReLU)
- **Parámetro:** β = 1 (congelado, ver selección abajo)
- **Método:** serie de Taylor truncada al grado nominal, derivada
  simbólicamente con SymPy (evita errores algebraicos manuales).
- **Base de salida:** monomial [a₀, a₁, ..., a_d].

El error de Taylor se mide **contra ReLU**, no contra Softplus.

## Selección de β

Se evaluaron β ∈ {1, 3, 5, 10, 20} sobre act1-I1 (grados 5 y 9) y se verificó
β ∈ {1, 3} en el escenario extremo act2-I2. Se seleccionó **β = 1** porque
presentó el menor error empírico, coeficientes de menor magnitud y mejor
estabilidad numérica para los grados evaluados.

El polinomio truncado de Taylor con β = 1 presenta menor error sobre los
intervalos evaluados que las variantes con β mayor. Esto es un resultado
empírico sobre los intervalos considerados, no una afirmación de convergencia
global. Los valores mayores de β incrementaron el error fuera del entorno local
del punto de expansión y produjeron crecimiento acelerado de los coeficientes
de orden alto (p. ej. C_max pasó de 0.69 con β=1 a ~3·10⁴ con β=20 en grado 9).

Evidencia: results/pilots/taylor_beta_metrics.csv

## Notas

- Grados nominales: 3, 5, 7, 9.
- El grado efectivo puede ser menor que el nominal porque ciertos coeficientes
  de orden alto se anulan (p. ej. grado nominal 5 → efectivo 4, con a₅ = 0).
  Esto afecta la profundidad multiplicativa real en CKKS.
- Coeficientes conocidos para β=1, grado 5: [ln2, 1/2, 1/8, 0, -1/192, 0].
- SymPy se usa solo para construcción offline, no en inferencia ni benchmarking.
