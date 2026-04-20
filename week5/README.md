# Semana 5 – Técnicas de Optimización e Hiperparámetros
## CADI Deep Learning | Actividad 5

---

### Objetivo
Comparar el efecto de distintos hiperparámetros y optimizadores en el entrenamiento de una red neuronal densa (sin convolución) sobre el dataset **Diabetes (OpenML ID 37)**. El análisis se basa en la modificación de un solo elemento por prueba para asegurar que la comparación sea válida e interpretable.

**Variables analizadas:**
* **Capacidad de la red:** Número de neuronas en capa oculta (32 vs 64 units).
* **Tasa de aprendizaje:** Learning rate (1e-3 vs 1e-2).
* **Tamaño de lote:** Batch size (32 vs 16).
* **Algoritmo de optimización:** Adam vs SGD con momentum.

---

### Evidencia Principal

* **Configuración óptima:** El modelo base utilizando **Adam, lr=1e-3, batch=32 y 32 neuronas** obtuvo una convergencia estable y un balance adecuado entre la precisión de validación y la de prueba.
* **Comportamiento de SGD:** Este optimizador requirió mayor tiempo para converger y presentó una varianza superior en las épocas finales del entrenamiento.
* **Análisis de estabilidad:** Se observó que un `lr=1e-2` fue el factor principal de inestabilidad durante el proceso de aprendizaje.

---

### Especificaciones del Modelo

**Arquitectura:**
Red neuronal densa (MLP):
`Input(8) -> Dense(units, activation='relu') -> Dense(1, activation='sigmoid')`

**Parámetros de entrenamiento:**

| Concepto | Detalle |
| :--- | :--- |
| **Función de pérdida** | binary_crossentropy |
| **Épocas totales** | 50 |
| **Validación** | 20% del conjunto de entrenamiento |

---

### Conclusiones

1. **Sensibilidad al Learning Rate:** Una tasa de aprendizaje alta (1e-2) provocó oscilaciones constantes en la curva de validación, demostrando que el exceso de velocidad en el ajuste impide alcanzar los mínimos óptimos.
2. **Influencia del Batch Size:** El uso de lotes pequeños (16) generó mayor ruido en las actualizaciones de los pesos, aunque la precisión final no se vio comprometida drásticamente.
3. **Eficacia de los Optimizadores:** Adam demostró ser superior a SGD en términos de velocidad y estabilidad inicial, confirmando su idoneidad para el desarrollo de prototipos rápidos y el manejo de datasets de escala pequeña.
