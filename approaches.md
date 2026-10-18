# Diferentes enfoques probados

El objetivo es predecir la probabilidad de que un sistema (subject) resuelva un ítem de un benchmark que no ha visto nunca, y mejorar esa predicción a medida que se revelan etiquetas durante el test. La métrica es el **Brier ALC**: el Brier score medio sobre presupuestos de 0, 1, 3, 7, 15 y 31 etiquetas reveladas por sistema (menor es mejor; un predictor constante de 0.5 obtiene 0.25).

## Componentes de la predicción
La predicción consta de dos partes:

1. Una parte a *priori (offline)*: básicamente todo lo que no utilice etiquetas reveladas en tiempo de test-adaptation
2. Una parte a *posteriori (test-time)*: la corrección en base a datos etiquetados del conjunto de test otorgado en tiempo real

Cualquier enfoque probado es una combinación de una parte a priori y otra parte a posteriori. Ambas se suman en la escala log-odds:

> **A = P + T**   (predictor = prior + test-time)

Los baselines se denotan **B** y el techo (ceiling, prdicción muy optimista y tramposa al usar datos de test) **G**. Cada predictor se escribe como **prior | test-time**.

### Componentes del prior (P)

El prior puede usar tres tipos de información:

- **R (resultados):** información sacada de los aciertos y fallos observados en entrenamiento, por ejemplo para obtener el rendimiento medio de un modelo
- **S (sistema):** metadatos del sistema (familia, tamaño, fecha de publicación...), no su comportamiento.
- **I (ítem):** características del ítem.

| Código | Qué es |
|---|---|
| **R1** | Tasa media de acierto en los datos de entrenamiento. Es el punto de partida de todos los priors. |
| **R2** | Fuerza de cada modelo base en comparación con otros, estimada con sus resultados en benchmarks. No es su tasa media de acierto. |
| **RG** | Dificultad real de cada ítem, medida con los resultados de los *demás* sistemas en el propio benchmark de test. Imposible en la competición: solo sirve para el techo. |
| **S1** | Metadatos del sistema, a partir de los campos que da la competición (nombre, proveedor, fechas, harness, esfuerzo de razonamiento). Se convierten en 11 variables:<br>• proveedor (OpenAI, Anthropic, Google…);<br>• fecha de lanzamiento y fecha de acceso (en años desde 2023);<br>• esfuerzo de razonamiento (0 si no se indica; de 1 = *low* a 5 = *max*);<br>• si es un modelo de razonamiento (por el nombre, p. ej. "think", "R1", "o1", o si tiene esfuerzo indicado);<br>• si usa harness, y si es un agente o *scaffold*;<br>• tamaño en parámetros, cuando el nombre lo dice (p. ej. "32B");<br>• gama pequeña ("mini", "flash", "haiku"…) o grande ("pro", "opus", "ultra"…), según el nombre;<br>• si falta el nombre del modelo. |
| **I0** | Estadísticas del texto: longitud, proporción de dígitos, matemáticas, código… |
| **I1** | Embeddings del texto (bge-base, reducidos a 48 componentes). |
| **I2** | Niveles de demanda ADeLe 1.0 (18 dimensiones). |
| **I3** | Niveles de demanda ADeLe 2.0 (8 dimensiones válidas). |
| **I4** | TF-IDF del texto (reducido a 48 componentes). |
| **I5** | Una única "puntuación de dificultad" que resume I1 + I2, aprendida con resultados (híbrido de I y R). |

El prior (assessor) es un modelo LightGBM que parte de R1 y aprende a desviarse de él usando los demás componentes.


### Componentes del test-time (T)

| Código | Qué es |
|---|---|
| **T0** | Sin adaptación: solo el prior. |
| **Tm** | La media de aciertos en el conjunto de test del sistema hasta el momento (el baseline oficial). |
| **T1** | Tres correcciones que se reajustan con cada etiqueta revelada: una del benchmark (¿es más fácil o difícil de lo esperado?), una por sistema (¿rinde mejor o peor de lo esperado?) y una por ítem (¿qué tal les ha ido a otros sistemas con este ítem?). Las correcciones se contraen hacia cero cuando hay pocas etiquetas. |
| **T2** | T1 + un factor λ que decide cuánto fiarse del prior en ese benchmark. Con λ = 1 se usa el prior tal cual; con λ = 0 se ignora y se vuelve a T1 puro. λ se estima con las etiquetas reveladas. |
| T1-s, T1-sb | T1 reducido: solo la corrección por sistema, o por sistema y benchmark. |
| T1-u | T1, pero eligiendo qué etiquetas revelar según la incertidumbre del modelo en vez de al azar. |

## Protocolos de evaluación

En ambos protocolos se simula la competición: se aparta un benchmark entero, se entrena con el resto y se evalúa en 30 sorteos (draws) con las reglas oficiales (grupos de 80–1.000 ítems, presupuestos de 0 a 31 etiquetas).

**Protocolo v1 (todos los resultados hasta ahora).** Solo measurement-db. Cada benchmark se aparta por turnos, pero solo 4 cumplen las reglas de tamaño: MathArena, Multi-SWE-bench, REAL y ResearchCodeBench. Toda conclusión sobre generalización descansa, por tanto, en cuatro casos.

**Protocolo v2 (en curso).** Añade la batería original de ADeLe: 15 modelos y 20 sub-benchmarks, de los que 13 son evaluables. Las decisiones principales, detalladas en [evaluation_protocol.md](NeurIPS-assessor-competition/docs/evaluation_protocol.md), son:

- **Dos conjuntos, reportados por separado:** measurement-db (principal) y la batería (secundario). La puntuación global es la media de los dos.
- **La batería se aparta por colección de origen** (AGIEval, TimeBench, LiveBench, MedCalcBench, SciBench), no por sub-benchmark. Así el benchmark evaluado no tiene "hermanos" en entrenamiento, como ocurrirá en la competición.
- **Variante con sistemas fríos:** se quitan de entrenamiento los modelos evaluados, porque en la batería siempre responden los mismos 15 y en la competición los sistemas serán mayoritariamente nuevos.
- **Peso de entrenamiento igual por colección,** para que la batería no domine solo por su número de benchmarks.
- **Los hiperparámetros se ajustan dentro del mismo conjunto,** fuera del benchmark evaluado.

## Baselines y techo

| | Definición | v1: measurement-db | v2: measurement-db | v2: batería | v2: global |
|---|---|---|---|---|---|
| **B1** constante | 0.5 \| T0 | 0.2500 | 0.2500 | 0.2500 | 0.2500 |
| **B2** oficial | 0.5 \| Tm | 0.2635 | 0.2635 | 0.2139 | 0.2387 |
| **B3** solo etiquetas | R1 \| T1 | 0.1984 | 0.1979 | 0.1892 | 0.1936 |
| **G** techo | RG \| T1 | 0.1264 | 0.1264 | 0.1534 | 0.1399 |

- **B3 no usa ningún modelo entrenado:** solo la tasa media de entrenamiento como punto de partida y las correcciones de T1. Aun así, mejora mucho el baseline oficial.
- **El hueco entre B3 y G** es el margen que podría ganar un buen prior: 0.072 en measurement-db y 0.036 en la batería.
- **En la variante fría, B3 obtiene 0.1893 en la batería,** prácticamente igual.

## Enfoques probados (protocolo v1, measurement-db)

Todos los priors entrenados hasta ahora comparten la misma **base = R1 + R2 + S1 + I0**, y se diferencian en la representación del ítem que se añade.

| Predictor | Prior | Con T1 | Con T2 |
|---|---|---|---|
| A1 / A2 | base + I4 (TF-IDF) | 0.2020 | 0.1981 |
| A3 / A4 | base + I1 (embeddings) | 0.1993 | **0.1966** |
| A5 / A6 | base + I1 + I4 | 0.2009 | 0.1976 |
| A7 / A8 | base + I2 (ADeLe 1.0, primera anotación) | 0.2018 | 0.1973 |
| A9 / A10 | base + I1 + I2 | 0.1999 | **0.1962** (mejor) |
| A11 / A12 | base + I5 (puntuación de dificultad) | 0.2047 | 0.1973 |
| A13–A20 | base + I2 limpio, ± I1, ± I3 | no ejecutados (cancelados) | — |

Como referencia, B3 obtiene 0.1984. A1 era el predictor pre-registrado; el resto se probaron después.

### Variantes y ablaciones

- **Sin información del sistema** (R1 + I0 + ítem \| T2, sin R2 ni S1): 0.1975–0.1996. Siempre peor que con ellos.
- **Solo el prior, sin adaptación** (\| T0): 0.256–0.268, peor que la constante. El prior por sí solo está mal calibrado en un benchmark nuevo.
- **Test-time reducido,** con el prior de A1 fijo:
  - T1-sb: 0.2259;
  - T1-s: 0.2309;
  - T1-u: 0.2019.

  Quitar la corrección por ítem es lo que más daña. Elegir las etiquetas por incertidumbre no aporta nada.
- **Cambios en cómo se entrena el prior de A1:**
  - quitar la información del ítem (I0 + I4): 0.2007;
  - quitar la del sistema (R2 + S1): 0.2058;
  - cambiar los pesos entre benchmarks: 0.1987–0.2024;
  - tratar de otra forma las puntuaciones continuas de MMDocRAG: 0.2027–0.2029;
  - **quitar R1**, sin punto de partida: 0.2062, de los peores.

## Conclusiones preliminares

1. **Casi toda la mejora viene del test-time.** B3 (sin modelo entrenado) consigue 0.065 de los 0.072 puntos de mejora sobre el baseline oficial que logra el mejor predictor.
2. **Los priors entrenados aportan poco sobre B3:** como mucho 0.002, y solo con T2, que puede rebajar la confianza en el prior cuando no transfiere bien. Con T1, ningún prior entrenado mejoró a B3.
3. **Lo que sí funciona en el prior son los componentes de resultados:** R1 (el punto de partida) y R2 (la fuerza del modelo). Quitarlos siempre empeora.
4. **Las características del ítem (I) son el eslabón débil:** ninguna se acerca a lo que daría conocer la dificultad real (RG). Además, ADeLe 2.0 resulta bastante redundante con ADeLe 1.0. Pero todo esto se ha medido en solo cuatro benchmarks; la batería (13 benchmarks más) debe confirmarlo o desmentirlo.

## En curso (protocolo v2)

Scripts preparados para ejecutar en el clúster o en un servidor de los nuestros. Todos usan T1, y también se reporta el prior solo (T0):

| Ejecución | Prior |
|---|---|
| Embeddings | R1 + I1 |
| ADeLe 1.0 | R1 + I2; R1 + I2 + S1 + R2 |
| ADeLe 2.0 | R1 + I3; R1 + I3 + S1 + R2 |
| Combinaciones | R1 + I1 + I2; R1 + I1 + I3; R1 + I2 + I3; R1 + I1 + I2 + I3 |

A diferencia de los enfoques anteriores, estos priors no incluyen I0 ni, por defecto, R2 + S1. Así se aísla qué aporta cada representación del ítem por sí sola.
