
# 🧠 Análisis de Regularización en Redes Neuronales

**Universidad de Cundinamarca** | Especialización en Inteligencia Artificial  
**Módulo:** Deep Learning (Semana 4)

---

## 📌 Descripción del Proyecto
Este proyecto implementa y compara diversas técnicas de regularización matemática y estructural sobre una Red Neuronal Artificial (ANN) para mitigar el problema de sobreentrenamiento (overfitting). Se utiliza el **Dataset de Diabetes** (OpenML) como caso de estudio clínico.

Se construyó un modelo "Baseline" intencionalmente propenso al overfitting y se comparó su rendimiento al aplicar L2, Dropout y Early Stopping.

---

## ⚙️ Justificación y Comparación de Técnicas Implementadas

La siguiente tabla resume el impacto documentado en el código:

| Técnica | Descripción Técnica | Impacto Observado en el Modelo | Justificación de Selección |
| :--- | :--- | :--- | :--- |
| **Baseline (Base)** | Red profunda sin restricciones. | Alta varianza. El *Loss* de validación diverge drásticamente del *Loss* de entrenamiento tras ~30 épocas. | Punto de referencia estricto para medir el impacto de la regularización. |
| **L2 (Weight Decay)** | Añade penalización $\lambda \sum w_i^2$ a la función de coste. | Suaviza la curva de validación, restringiendo el crecimiento excesivo de los pesos. | Ideal para manejar la multicolinealidad y evitar que unas pocas características dominen el modelo. |
| **Dropout (40%)** | Desactiva aleatoriamente el 40% de las neuronas en cada iteración. | Retrasa el inicio del overfitting y mejora ligeramente la precisión de prueba. | Previene la co-adaptación de neuronas, forzando a la red a distribuir el aprendizaje. |
| **Early Stopping** | Detiene el entrenamiento si el *val_loss* no mejora tras 15 épocas. | Interrumpe el entrenamiento en el punto óptimo (menor varianza) y restaura pesos. | Método computacionalmente eficiente para capturar el modelo en su punto de máxima generalización temporal. |

---

## 📊 Conclusiones Principales

1. **Efectividad en el Dataset de Diabetes:** Dado que el dataset de diabetes tiene un número relativamente pequeño de observaciones, el modelo *Baseline* memoriza rápidamente los datos. 
2. **Mejor Método Individual:** **Early Stopping** demostró ser el método más pragmático. Logró capturar los mejores pesos antes de la divergencia del error, ahorrando tiempo computacional y asegurando un modelo con la varianza óptima.
3. **Estabilidad:** La técnica de **Dropout** ofreció la curva de validación más estable a lo largo de las épocas, demostrando ser esencial en arquitecturas densas para forzar características robustas e independientes.
4. **Reducción de Varianza:** Todas las técnicas de regularización implementadas cumplieron su objetivo matemático: restringir la complejidad del espacio de hipótesis, cambiando un ligero incremento en el sesgo (bias) por una reducción masiva en la varianza frente a datos de prueba.

---
