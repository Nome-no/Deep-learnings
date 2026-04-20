Semana 5 – Técnicas de Optimización e Hiperparámetros
CADI Deep Learning | Actividad 5

Objetivo
Comparar el efecto de distintos hiperparámetros y optimizadores en el entrenamiento de una red neuronal densa (sin convolución) sobre el dataset Diabetes (OpenML ID 37), cambiando un elemento a la vez para que la comparación sea válida e interpretable.
Elementos comparados:

Número de neuronas en capa oculta (units: 32 vs 64)
Tasa de aprendizaje (lr: 1e-3 vs 1e-2)
Tamaño de lote (batch_size: 32 vs 16)
Optimizador (Adam vs SGD con momentum)


Evidencia Principal

La configuración base con Adam, lr=1e-3, batch=32, 32 neuronas logró una convergencia estable con buen balance entre precisión de validación y prueba.
SGD tardó más en converger y mostró mayor varianza en las últimas épocas, mientras que lr=1e-2 generó la mayor inestabilidad en el entrenamiento según la gráfica de estabilidad.


Modelo
Red neuronal densa (sin capas convolucionales):
Input(8) → Dense(units, ReLU) → Dense(1, Sigmoid)

Loss: binary_crossentropy
Epochs: 50
Validation split: 20% del set de entrenamiento


Conclusiones

Learning rate alto (1e-2) causó la mayor inestabilidad durante el entrenamiento, con oscilaciones frecuentes en la curva de validación, evidenciando que saltar sobre mínimos óptimos perjudica la convergencia.
Batch size pequeño (16) introdujo más ruido por actualización, pero no afectó significativamente la precisión final, mostrando un balance entre velocidad de convergencia y estabilidad.
Adam superó a SGD en velocidad de convergencia y estabilidad desde las primeras épocas, confirmando su ventaja para datasets pequeños y prototipos rápidos.
