# Semana 10 – Red Neuronal Siamesa para Reconocimiento Facial
**CADI Deep Learning | Universidad de Cundinamarca**

---

## Descripción

En esta actividad se implementa una **red neuronal siamesa** orientada al reconocimiento de similitud entre imágenes, aplicada al dataset **Olivetti Faces** (AT&T). La red recibe pares de imágenes y aprende a determinar si pertenecen a la misma persona, generando representaciones en un espacio latente y comparándolas mediante distancia euclidiana.

---

## Dataset

**Olivetti Faces** (AT&T Laboratories Cambridge)  
400 imágenes en escala de grises de 64×64 píxeles · 40 personas · 10 fotos por persona  
Variaciones: iluminación, expresión facial, presencia de lentes.

---

## Arquitectura

```
Imagen A ──┐
           ├─► SubRed compartida (Conv2D × 3 + Dense) ──► Embedding A (64-d) ──┐
Imagen B ──┘                                                                    ├─► ||A-B||₂ ──► Dense(1, sigmoid) ──► P(distinto)
           ├─► SubRed compartida (mismos pesos)        ──► Embedding B (64-d) ──┘
```

### Sub-red gemela (pesos compartidos)
| Capa | Filtros / Unidades | Salida |
|------|--------------------|--------|
| Conv2D + BN + MaxPool | 32 | 32×32 |
| Conv2D + BN + MaxPool | 64 | 16×16 |
| Conv2D + BN + GAP | 128 | 128 |
| Dense + Dropout(0.3) | 128 | 128 |
| Dense (espacio latente) | 64 | **64** |

---

## Criterios de rúbrica cubiertos

### Implementación de Data Augmentation — *no aplica directamente*
Este trabajo implementa en su lugar la **generación de pares de entrenamiento** (positivos y negativos) como técnica de aumento de datos para redes siamesas. Cada imagen original participa en múltiples pares, multiplicando el tamaño efectivo del dataset por el parámetro `n_pares_por_clase`.

### Evaluación de similitud entre pares
La comparación entre pares de la misma persona y de personas distintas se documenta con:
- Visualización de pares positivos y negativos (§3.1)
- Curva ROC con AUC y umbral óptimo de Youden (§7)
- Matriz de confusión con métricas de precisión, recall y F1 (§7)

### Implementación de la red siamesa (Transfer Learning conceptual)
La sub-red comparte pesos entre ambas ramas — esto es análogo al Transfer Learning: las representaciones aprendidas para comparar una imagen se transfieren automáticamente a la comparación de cualquier otro par.

### Entrenamiento y evaluación
El modelo se entrena con Early Stopping y ReduceLROnPlateau. Los resultados se presentan con curvas de aprendizaje, tabla de métricas y visualización en espacio latente (PCA 2D).

### Documentación y organización del código
Cada celda de código incluye comentarios línea a línea explicando el propósito de cada instrucción. El notebook está estructurado en 10 secciones progresivas con celdas Markdown de contexto teórico antes de cada bloque de código.

---

## Estructura del notebook

| Sección | Contenido |
|---------|-----------|
| §1 | Importaciones y semillas de reproducibilidad |
| §2 | Carga y exploración visual del dataset Olivetti |
| §3 | Generación de pares positivos y negativos |
| §3.1 | Visualización de pares generados |
| §4 | Arquitectura siamesa completa con `model.summary()` |
| §5 | Función de pérdida y compilación del modelo |
| §6 | Entrenamiento con Early Stopping y ReduceLROnPlateau |
| §6.1 | Curvas de Loss y Accuracy |
| §7 | Evaluación: AUC-ROC, umbral óptimo, matriz de confusión |
| §8 | Visualización del espacio latente con PCA 2D |
| §9 | Demostración visual de predicciones individuales |
| §10 | Tabla resumen de métricas y conclusiones automáticas |

---

## Librerías utilizadas

Las mismas librerías de los trabajos anteriores de la materia:

```python
numpy, tensorflow / keras, matplotlib, pandas
scikit-learn (fetch_olivetti_faces, train_test_split, métricas)
seaborn
```

No se utilizan librerías adicionales ni externas.

---

## Instrucciones de uso en Google Colab

1. Subir el archivo `.ipynb` a Google Colab.
2. Ejecutar las celdas en orden secuencial (Ctrl+F9 o Entorno de ejecución → Ejecutar todo).
3. El dataset Olivetti Faces se descarga automáticamente desde scikit-learn (~1 MB).
4. No se requiere GPU para este notebook (tiempo estimado: 3–8 minutos en CPU).

---

## Conceptos clave

**Red siamesa:** arquitectura con dos ramas que comparten pesos, diseñada para aprender una función de similitud en lugar de clasificar clases fijas.

**Pares positivos / negativos:** la red se entrena con pares de imágenes etiquetados como "misma persona" o "distinta persona", lo que permite aprender con combinaciones de ejemplos en lugar de imágenes individuales.

**Espacio latente:** representación vectorial de baja dimensión donde la distancia euclidiana refleja similitud visual. Si la red aprende correctamente, imágenes de la misma persona tienen vectores cercanos.

**Contrastive Loss / BCE:** la función de pérdida que dirige el entrenamiento para minimizar la distancia entre pares positivos y maximizarla entre negativos hasta un margen.
