# Semana 3 - Actividad 2: Implementación de Backpropagation y Funciones de Activación

## Objetivo
El objetivo de esta actividad es demostrar la implementación técnica de una red neuronal multicapa desde cero utilizando NumPy, con un enfoque particular en el algoritmo de **backpropagation** y el análisis comparativo de funciones de activación.

## Implementación Técnica
Se ha programado una arquitectura de red neuronal que incluye:
* **Forward Pass:** Implementación matricial para la propagación de señales.
* **Backpropagation:** Cálculo manual de gradientes utilizando la regla de la cadena para actualizar pesos ($W$) y sesgos ($b$).
* **Optimizador:** Gradiente Descendente con ajuste de hiperparámetros.
* **Dataset:** El problema no lineal **XOR**, que requiere necesariamente una capa oculta para su resolución.

## Comparación de Resultados
Se evaluaron dos funciones de activación bajo las mismas condiciones de entrenamiento:

1. **Sigmoide:** Presentó una convergencia progresiva. Aunque efectiva, requiere más iteraciones para estabilizar el error debido a la naturaleza de su derivada en los extremos.
2. **ReLU (Rectified Linear Unit):** Mostró una eficiencia superior, logrando una reducción del error casi instantánea (convergencia acelerada) en comparación con la Sigmoide.

### Métricas de Evaluación Final
Tras el entrenamiento, se validaron los modelos obteniendo los siguientes resultados:

| Activación | Loss Final | Accuracy (Precisión) |
| :--- | :--- | :--- |
| **Sigmoid** | ~0.0150 | 100.0% |
| **ReLU** | ~0.0001 | 100.0% |

## Conclusiones del Experimento
* **Eficacia del Algoritmo:** El descenso de la curva de pérdida (Loss) de 0.25 a valores cercanos a cero confirma que el algoritmo de backpropagation está ajustando los pesos correctamente.
* **Superioridad de ReLU:** En este escenario, ReLU demostró ser más robusta frente al estancamiento de gradientes, permitiendo que la red aprenda la lógica XOR en menos épocas.
* **Precisión:** Ambos modelos alcanzaron un 100% de precisión, clasificando correctamente todas las combinaciones de la tabla de verdad XOR.

## Cómo ejecutar el notebook
1.  Abrir `notebook.ipynb` en **Google Colab**.
2.  Ejecutar todas las celdas (`Runtime > Run all`).
3.  La última celda imprimirá la tabla de resultados finales y mostrará la gráfica comparativa de convergencia.
