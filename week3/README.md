# 🧠 Implementación de Backpropagation y Funciones de Activación - CADI Deep Learning

## 🎯 Objetivo de la Actividad
Demostrar la implementación técnica de una **red neuronal multicapa** desde sus fundamentos matemáticos utilizando únicamente **NumPy**. El enfoque principal reside en el algoritmo de **backpropagation** y el análisis comparativo del impacto de diferentes funciones de activación en el aprendizaje.

---

## 🏗️ Arquitectura de la Solución
Se ha desarrollado un modelo capaz de resolver el problema no lineal **XOR**, el cual requiere necesariamente una capa oculta para su resolución.

### 🛠️ Componentes Técnicos:
* **Forward Pass:** Implementación matricial para la propagación de señales a través de la red.
* **Backpropagation:** Cálculo manual de gradientes mediante la **regla de la cadena** para la actualización de pesos ($W$) y sesgos ($b$).
* **Optimizador:** Algoritmo de Gradiente Descendente con ajuste fino de hiperparámetros.
* **Dataset:** Tabla de verdad XOR (entradas no linealmente separables).

---

## 🧪 Comparación de Resultados
Se evaluaron dos funciones de activación bajo condiciones de entrenamiento idénticas para medir su eficiencia y velocidad de convergencia:

1.  **Sigmoide:** Presentó una convergencia progresiva. Aunque efectiva, sufre de saturación de gradientes en los extremos, lo que requiere más iteraciones para estabilizar el error.
2.  **ReLU (Rectified Linear Unit):** Mostró una eficiencia superior con una **convergencia acelerada**, logrando reducir el error drásticamente en las primeras épocas.

### 📊 Métricas de Evaluación Final

| Activación | Loss Final | Accuracy (Precisión) | Estado |
| :--- | :--- | :--- | :--- |
| **Sigmoid** | `~0.0150` | `100.0%` | ✅ Éxito |
| **ReLU** | `~0.0001` | `100.0%` | ✅ Éxito |

---

## 📈 Conclusiones 

* **Eficacia del Algoritmo:** El descenso de la curva de pérdida (MSE) desde **0.25** hasta valores cercanos a cero confirma que el motor de backpropagation está ajustando los parámetros correctamente.
* **Superioridad de ReLU:** En este escenario, ReLU demostró ser más robusta frente al estancamiento de gradientes, permitiendo que la red aprenda la lógica XOR en menos tiempo de cómputo.
* **Precisión Total:** Ambos modelos alcanzaron el **100% de precisión**, logrando una clasificación perfecta de todas las combinaciones lógicas de la tabla de verdad.



---
*Este proyecto forma parte de la especialización en Machine Learning.*
