# Semana 13 – Implementación de un Autoencoder en una Red Denoising usando el dataset MNIST

## Descripción

En esta actividad se implementa un **Autoencoder Denoising** utilizando el dataset **MNIST** en Google Colab.  
El modelo aprende a reconstruir imágenes de dígitos manuscritos que han sido corrompidas con ruido gaussiano, sin utilizar las etiquetas de clase (aprendizaje no supervisado).

## Estructura del notebook

| Sección | Contenido |
|---|---|
| 1. Importación de librerías | TensorFlow, NumPy, Matplotlib, scikit-learn (PCA) |
| 2. Carga y preprocesamiento | Normalización, aplanado, adición de ruido gaussiano (σ = 0.5) |
| 3. Implementación del Autoencoder | Arquitectura Encoder–Decoder con capas Dense |
| 4. Modelos auxiliares | Encoder y Decoder como sub-modelos separados |
| 5. Entrenamiento | 50 épocas máx., batch=256, EarlyStopping (patience=5) |
| 6. Curvas de entrenamiento | Loss de entrenamiento y validación por época |
| 7. Evaluación de resultados | Loss de prueba, MSE, visualización comparativa, mapa de error |
| 8. Espacio latente | Proyección PCA 2D del espacio latente (64 dimensiones) |
| 9. Conclusiones | Análisis del desempeño y limitaciones del modelo |

## Arquitectura del modelo

```
Input (784)
    │
   Dense(256, ReLU)   ← Encoder
    │
   Dense(128, ReLU)
    │
   Dense(64, ReLU)    ← Espacio latente
    │
   Dense(128, ReLU)   ← Decoder
    │
   Dense(256, ReLU)
    │
   Dense(784, Sigmoid) ← Reconstrucción
```

**Parámetros totales:** ~534 000  
**Espacio latente:** 64 dimensiones (compresión 12.25×)

## Dataset

- **MNIST**: 70 000 imágenes de dígitos escritos a mano (0–9)
  - 60 000 entrenamiento | 10 000 prueba
  - Tamaño: 28×28 píxeles, escala de grises
  - Preprocesamiento: normalización [0,1] → aplanado (784) → ruido gaussiano (σ=0.5)

## Hiperparámetros

| Parámetro | Valor |
|---|---|
| Épocas máximas | 50 |
| Batch size | 256 |
| Optimizador | Adam |
| Función de pérdida | Binary Crossentropy |
| Validación | 15 % del conjunto de entrenamiento |
| EarlyStopping patience | 5 épocas |
| Factor de ruido | 0.5 |

## Librerías utilizadas

| Librería | Versión mínima | Uso |
|---|---|---|
| `tensorflow` | 2.x | Modelo, entrenamiento, Keras API |
| `numpy` | 1.x | Manipulación de arreglos, adición de ruido |
| `matplotlib` | 3.x | Visualización de imágenes y curvas |
| `scikit-learn` | 1.x | PCA para reducción del espacio latente |

> Todas las librerías están preinstaladas en Google Colab. No se requieren instalaciones adicionales.

## Resultados esperados

- **Loss de prueba** (Binary CE): < 0.10
- **MSE**: < 0.02
- Imágenes reconstruidas visualmente coherentes con las originales
- Agrupaciones por dígito visibles en la proyección PCA del espacio latente

## Instrucciones de ejecución

1. Abrir el archivo `.ipynb` en [Google Colab](https://colab.research.google.com/)
2. Ejecutar las celdas en orden secuencial (`Ctrl+F9` para ejecutar todo)
3. No se requiere GPU (el modelo es ligero con capas Dense)
4. Tiempo estimado de entrenamiento: **2–5 minutos** en CPU

## Criterios de rúbrica cubiertos

| Criterio | Implementación |
|---|---|
| Carga y preprocesamiento del dataset | Sección 2: normalización, aplanado y ruido documentados línea a línea |
| Implementación del autoencoder | Sección 3–4: encoder, decoder y modelos auxiliares con comentarios |
| Entrenamiento del autoencoder | Sección 5: hiperparámetros justificados, EarlyStopping, convergencia |
| Evaluación y visualización | Sección 6–8: curvas, comparación visual, MSE, mapa de error, PCA |
| Documentación y organización | Código comentado línea a línea, celdas Markdown explicativas, estructura clara |

---

**Actividad:** Semana 13 – CADI Deep Learning | Universidad de Cundinamarca  
**Entorno:** Google Colab (Python 3.10, TensorFlow 2.x)
