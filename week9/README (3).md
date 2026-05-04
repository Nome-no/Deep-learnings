# Semana 9 – Data Augmentation y Transfer Learning
**CADI Deep Learning | Universidad de Cundinamarca**

---

## Descripción general

En esta actividad se implementan y comparan técnicas de **Data Augmentation** y **Transfer Learning** para mejorar el desempeño de modelos de clasificación de imágenes sobre **Fashion-MNIST**.

Se comparan cuatro escenarios:

| Modelo | Data Augmentation | Transfer Learning |
|--------|:-----------------:|:-----------------:|
| A – CNN base           | ✗ | ✗ |
| B – CNN + Aug          | ✓ | ✗ |
| C – MobileNetV2        | ✗ | ✓ |
| D – MobileNetV2 + Aug  | ✓ | ✓ |

El estudiante compara el desempeño de los modelos en los cuatro escenarios y analiza el impacto de ambas técnicas en la capacidad de generalización.

---

## Criterios de rúbrica cubiertos

### 1. Implementación de Data Augmentation *(1 punto)*
Se implementan las siguientes técnicas sobre el dataset de Fashion-MNIST:
- `RandomFlip('horizontal')` — volteo horizontal aleatorio
- `RandomRotation(0.1)` — rotación aleatoria ±10%
- `RandomZoom(0.1)` — zoom aleatorio ±10%
- `RandomTranslation(0.05, 0.05)` — traslación ±5% en ambos ejes

El pipeline de aumento se inserta directamente en la arquitectura del modelo como una capa de Keras, de modo que **solo se activa durante el entrenamiento** (`training=True`) y nunca durante la evaluación. El código es eficiente y cada transformación está documentada línea a línea en la sección §3 del notebook.

### 2. Evaluación del impacto de Data Augmentation *(1 punto)*
La comparación entre modelos con y sin Data Augmentation se realiza en dos escenarios controlados:
- **A vs B** — misma CNN base, diferente estrategia de datos
- **C vs D** — mismo Transfer Learning, diferente estrategia de datos

Se presentan tres gráficos de apoyo (§6, §8, §10) que muestran curvas de entrenamiento, tabla resumen con valores numéricos exactos (Δ accuracy) y barras comparativas del impacto aislado de cada técnica. Las conclusiones de §12 se generan automáticamente con los valores reales de la ejecución.

### 3. Implementación de Transfer Learning *(1 punto)*
Se utiliza **MobileNetV2** preentrenado en ImageNet como extractor de características:
- `include_top=False` — se excluye la cabeza original de 1000 clases
- `weights='imagenet'` — se cargan los pesos preentrenados en 1.2M imágenes
- `base_model.trainable = False` — **la base convolucional se congela completamente**; solo la cabeza densa nueva es entrenable
- `base_model(x, training=False)` — las capas de BatchNorm usan estadísticas de ImageNet, no las del lote actual

La cabeza nueva consiste en `GlobalAveragePooling2D → Dense(128) + ReLU → Dropout(0.3) → Softmax(10)`, adaptada a las 10 clases de Fashion-MNIST.

### 4. Entrenamiento y evaluación del modelo con Transfer Learning *(1 punto)*
Los modelos C y D se entrenan con los hiperparámetros correctos:
- `Adam(lr=1e-3)` como optimizador
- `EarlyStopping(patience=4, restore_best_weights=True)` para evitar sobreajuste
- `ReduceLROnPlateau(factor=0.5, patience=2)` para ajuste fino del LR

Los resultados se presentan con: curvas de entrenamiento (§6), `classification_report` con precision, recall y F1 por clase (§7), tabla resumen comparativa (§8), matrices de confusión por modelo (§9) y accuracy individual por clase (§11).

### 5. Documentación y organización del código *(1 punto)*
- Cada línea no trivial tiene comentario explicativo en el mismo bloque de código
- Cada sección inicia con una celda Markdown que explica el propósito, el criterio de rúbrica cubierto y los conceptos clave
- El mapa de navegación en la portada permite ubicar cualquier sección de un vistazo
- Las funciones `build_cnn()` y `build_transfer()` están documentadas con docstring y comentarios internos
- La organización sigue el flujo lógico: datos → modelo → entrenamiento → evaluación → análisis → conclusiones

---

## Arquitecturas

### CNN base (Modelos A y B)
```
Input(28×28×1) → [Aug – solo B]
→ Conv2D(32) + BN + MaxPool  →  14×14
→ Conv2D(64) + BN + MaxPool  →  7×7
→ Conv2D(128) + BN
→ GlobalAveragePooling
→ Dense(128, ReLU) + Dropout(0.4)
→ Softmax(10)
```

### Transfer Learning (Modelos C y D)
```
Input(64×64×3) → [Aug – solo D]
→ MobileNetV2 [CONGELADO, pesos ImageNet]
→ GlobalAveragePooling
→ Dense(128, ReLU) + Dropout(0.3)
→ Softmax(10)  ← única parte entrenable
```

---

## Consideraciones de memoria para Colab gratuito

- **10 000 train / 2 000 test** en vez de los 60 000 completos → evita OOM
- **Entrada 64×64** para MobileNetV2 (mínimo permitido: 32×32) → 4× menos RAM que 96×96
- `GlobalAveragePooling2D` en vez de `Flatten` → menos parámetros en cabeza densa
- `EarlyStopping` reduce el número real de épocas ejecutadas

---

## Cómo ejecutar

1. Abrir `Semana9_DataAug_TransferLearning.ipynb` en [Google Colab](https://colab.research.google.com/)
2. Activar GPU: *Entorno de ejecución → Cambiar tipo de entorno → T4 GPU*
3. Ejecutar todo con `Ctrl + F9`
4. Tiempo estimado con GPU: **8–12 minutos**

---

## Librerías utilizadas

| Librería | Uso |
|----------|-----|
| `tensorflow / keras` | Construcción y entrenamiento de modelos |
| `numpy` | Operaciones numéricas y preprocesamiento |
| `matplotlib` | Curvas de entrenamiento y gráficos comparativos |
| `seaborn` | Matrices de confusión (heatmap) |
| `scikit-learn` | `classification_report`, `confusion_matrix` |
| `pandas` | Tabla resumen de resultados |
