# Week 8 – CNN & Transfer Learning

**Curso:** CADI Deep Learning | Universidad de Cundinamarca  
**Actividad:** Semana 8 – Implementación CNN + aproximación a Transfer Learning

---

## Dataset

**Fashion-MNIST** (integrado en `keras.datasets`):
- 70 000 imágenes en escala de grises (28×28 px), 10 categorías de ropa
- 60 000 para entrenamiento / 10 000 para test
- Clases: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

---

## Arquitecturas implementadas

### Modelo A – CNN base (desde cero)
```
Input(28×28×1)
→ Conv2D(32, 3×3) + ReLU → MaxPool(2)
→ Conv2D(64, 3×3) + ReLU → MaxPool(2)
→ Conv2D(128, 3×3) + ReLU
→ GlobalAveragePooling2D
→ Dense(128) + ReLU → Dropout(0.4)
→ Dense(10) + Softmax
```

### Modelo B – Transfer Learning con MobileNetV2
```
Input(96×96×3)
→ MobileNetV2 (base congelada, pesos ImageNet)
→ GlobalAveragePooling2D
→ Dense(128) + ReLU → Dropout(0.3)
→ Dense(10) + Softmax
```
Solo la cabeza densa es entrenable; la base permanece congelada.

---

## Comparación realizada

| Modelo | Parámetros entrenables | Test Accuracy (aprox.) |
|---|---|---|
| CNN base | ~200 K | ~91 % |
| Transfer Learning (MobileNetV2) | ~130 K | ~88–90 % |

Ambos modelos usan EarlyStopping (`patience=3`) y Adam(lr=1e-3).

---

## Resultado principal

La CNN base entrenada desde cero supera levemente al modelo con Transfer Learning en este escenario, dado que Fashion-MNIST tiene suficientes muestras y las características de ImageNet no son ideales para imágenes monocromáticas de ropa. Sin embargo, el modelo de Transfer Learning converge más rápido y es más robusto cuando los datos son escasos.

---

## Cómo ejecutar

1. Abrir `Semana8Actividad8.ipynb` en [Google Colab](https://colab.research.google.com/)
2. Activar GPU: `Entorno de ejecución → Cambiar tipo de entorno → T4 GPU`
3. Ejecutar **Run all** (`Ctrl + F9`)
4. Tiempo estimado de ejecución completa: ~8–12 minutos con GPU

> **No requiere instalaciones adicionales:** TensorFlow, Keras y scikit-learn están preinstalados en Colab.

