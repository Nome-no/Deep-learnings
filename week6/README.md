# Semana 6 – Bias/Variance, Overfitting y Regularización

**CADI Deep Learning | Actividad 6**

---

## Objetivo

Explorar el trade-off **bias/variance** y el fenómeno de **sobreajuste (overfitting)** entrenando dos redes neuronales sobre el dataset de diabetes (Pima Indians), y demostrar cómo las técnicas de regularización mejoran la capacidad de generalización del modelo.

---

## Dataset

**Pima Indians Diabetes Database** — cargado con `fetch_openml(name="diabetes", version=1)` desde scikit-learn.

| Característica | Detalle |
|---|---|
| Muestras | 768 |
| Features | 8 (glucosa, BMI, edad, insulina, etc.) |
| Tarea | Clasificación binaria (diabetes positivo / negativo) |
| Balance de clases | ~65 % negativo / 35 % positivo |

---

## Técnicas de regularización aplicadas

| Técnica | Descripción | Aplicada en |
|---|---|---|
| **L2 (Weight Decay)** | Penaliza pesos grandes con `λ=1e-4` en la función de pérdida | Modelo regularizado |
| **Dropout (30 %)** | Desactiva aleatoriamente el 30 % de neuronas por epoch | Modelo regularizado |
| **Early Stopping** | Detiene el entrenamiento si `val_loss` no mejora en 10 épocas consecutivas | Modelo regularizado |

---

## Comparación realizada

Se entrenaron dos modelos con la **misma arquitectura** (128 → 64 → 1 neuronas) para que la única variable sea la regularización:

- **Modelo BASE:** Sin ninguna regularización. 100 épocas fijas.
- **Modelo REGULARIZADO:** L2 + Dropout + Early Stopping.

Los modelos se comparan mediante:
1. **Curvas de aprendizaje** (train loss vs val loss por epoch) — evidencia de overfitting/generalización.
2. **Matrices de Confusión** — análisis de errores por clase.
3. **Tabla de métricas:** Accuracy, Precision, Recall, F1-Score, Specificity, Sensitivity y AUC-ROC.

---

## Cómo ejecutar

1. Subir `Actividad6Semana6.ipynb` a [Google Colab]
2. Ir a **Runtime → Run all** (o `Ctrl+F9`).
3. El dataset se descarga automáticamente desde OpenML en la primera ejecución.


## Conclusiones (resumen)

1. El modelo base exhibe **overfitting**: `train_loss` desciende continuamente mientras `val_loss` diverge, evidenciando alta varianza.
2. **Dropout** impide que la red memorice los datos al apagar neuronas aleatoriamente en cada paso de entrenamiento.
3. **L2** penaliza pesos grandes, produciendo modelos más suaves y con menor tendencia al sobreajuste.
4. **Early Stopping** garantiza recuperar los pesos del epoch de mejor generalización, sin sobreentrenar.
5. El modelo regularizado obtiene **F1-Score y AUC-ROC iguales o superiores** al modelo base en datos de prueba, confirmando la mejora en generalización a costa de un leve incremento de bias en entrenamiento.

