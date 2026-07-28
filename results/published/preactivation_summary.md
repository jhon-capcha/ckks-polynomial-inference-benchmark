# Resumen de preactivaciones — Hito 2

Caracterización de las preactivaciones de la CNN base (ReducedLeNet + ReLU), entrenada sobre MNIST, para definir los intervalos de aproximación polinómica. Checkpoint: mejor época de validación (época 8). Todo en texto plano.

## Estadísticas exactas globales

| Activación | Count | Min | Max | Mean | Std |
|---|---|---|---|---|---|
| act1 | 254,016,000 | -9.849 | 10.658 | 0.032 | 1.987 |
| act2 | 86,400,000 | -32.066 | 18.482 | -2.097 | 4.547 |
| act3 | 6,480,000 | -38.089 | 25.043 | -1.005 | 5.136 |

## Intervalos definitivos (asimétricos, por activación)

| Activación | I1 (99% central) | I2 (99.9% central) |
|---|---|---|
| act1 | [-6.610, 7.235] | [-8.450, 8.756] |
| act2 | [-18.473, 9.093] | [-24.834, 12.234] |
| act3 | [-15.780, 12.519] | [-22.149, 16.397] |

## Estabilidad del muestreo (seeds 42 vs 123)

- Límites comparados: 12
- Límites estables: 12
- Máxima diferencia absoluta: 0.1565
- Máxima diferencia relativa: 0.945%
- Estado: stable

## Nota metodológica

Los intervalos se mantienen asimétricos porque reflejan la distribución empírica de cada punto de activación. La simetrización habría ampliado innecesariamente el dominio, especialmente en act2, reduciendo la capacidad efectiva de los polinomios de grado fijo alrededor del origen.

Los intervalos son globales por activación, no por canal. La cobertura central global del 99% o 99.9% no implica idéntica cobertura dentro de cada canal; el canal 4 de act1 (unilateral, con cola negativa dominante) es evidencia concreta de esa heterogeneidad. Se aplica un único polinomio por activación de forma deliberada.
