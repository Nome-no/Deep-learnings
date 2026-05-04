# Semana 11 – Red Neuronal Recurrente (LSTM) para Predicción de Series de Tiempo

**CADI Deep Learning | Universidad de Cundinamarca**

---

## Objetivo

Implementar una **Red Neuronal Recurrente** en su variante **LSTM (Long Short-Term Memory)** orientada a la predicción de series de tiempo, comprendiendo cómo los modelos recurrentes capturan dependencias temporales para predecir valores futuros a partir de información pasada.

---

## Dataset

**Airline Passengers** (Box & Jenkins, 1976)  
144 registros mensuales del número de pasajeros aéreos internacionales (1949–1960).  
Fuente pública: `https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv`

| Característica | Valor |
|---|---|
| Registros | 144 meses |
| Variable | Pasajeros (miles) |
| Patrón | Tendencia creciente + estacionalidad anual |

---

## Variante RNN implementada

| Aspecto | Detalle |
|---|---|
| **Variante** | LSTM (Long Short-Term Memory) |
| **Arquitectura** | `LSTM(64, return_sequences=True)` → `LSTM(32)` → `Dense(1)` |
| **Ventana temporal** | `look_back = 12` meses |
| **Entrada** | `(samples, 12, 1)` — formato requerido por Keras |
| **Salida** | 1 valor continuo (regresión) |

### ¿Por qué LSTM sobre RNN simple?

La RNN vanilla sufre del problema del **gradiente que se desvanece** en secuencias largas, lo que impide aprender dependencias de largo plazo. LSTM soluciona esto con tres compuertas:
- **Forget gate:** descarta información irrelevante del pasado
- **Input gate:** incorpora nueva información al estado de celda
- **Output gate:** controla qué parte del estado se expone como predicción

---

## Preprocesamiento

| Paso | Decisión | Justificación |
|------|----------|---------------|
| Normalización | MinMaxScaler → [0, 1] | Las RNN son sensibles a la escala de entrada |
| División | 80 % train / 20 % test | Se respeta el orden temporal — sin shuffle |
| Ventana deslizante | `look_back = 12` | Captura la estacionalidad anual completa |
| Reshape | `(n, 12, 1)` | Formato `(samples, timesteps, features)` requerido por LSTM |

---

## Entrenamiento

| Hiperparámetro | Valor | Razón |
|---|---|---|
| Optimizador | Adam | Adaptativo, estándar para RNN |
| Función de pérdida | MSE | Regresión de valores continuos |
| Épocas máximas | 150 | Early Stopping detiene antes |
| Batch size | 16 | Actualización frecuente, adecuado para series cortas |
| Validation split | 15 % | Monitoreo de val_loss para Early Stopping |
| Early Stopping patience | 10 | Restaura mejores pesos automáticamente |

---

## Métricas de evaluación

- **MSE** — Error cuadrático medio (penaliza errores grandes)
- **RMSE** — Raíz del MSE, en la misma unidad que la serie (miles de pasajeros)
- **MAE** — Error absoluto medio, robusto a valores extremos

---

## Estructura del notebook

| Sección | Contenido |
|---------|-----------|
| 1. Importaciones | Librerías y semillas de reproducibilidad |
| 2. Carga y visualización | Dataset, descripción estadística y gráfica de la serie |
| 3. Preprocesamiento | Normalización, división temporal y ventana deslizante |
| 4. Arquitectura LSTM | Definición del modelo con comentarios línea a línea |
| 5. Entrenamiento | Ajuste con Early Stopping |
| 6. Curvas de entrenamiento | Train/val loss por época con marcador del mejor epoch |
| 7. Evaluación | MSE, RMSE y MAE en train y test |
| 8. Visualización | Predicciones vs. serie real completa |
| 9. Análisis | Interpretación de métricas y comportamiento del modelo |
| 10. Conclusiones | 5 conclusiones sobre el modelo y la arquitectura |

---

## Cómo ejecutar

1. Abrir `Semana11_RNN_SeriesTiempo.ipynb` en [Google Colab](https://colab.research.google.com/)
2. Ejecutar todas las celdas en orden (`Ctrl + F9`)
3. No se requiere GPU — el modelo es liviano y corre en CPU en menos de 2 minutos

---

## Librerías utilizadas

| Librería | Uso |
|---|---|
| `tensorflow / keras` | Construcción y entrenamiento del modelo LSTM |
| `numpy` | Operaciones numéricas y ventana deslizante |
| `pandas` | Carga del dataset y tabla de resultados |
| `matplotlib` | Curvas de entrenamiento y predicciones |
| `sklearn` | MinMaxScaler, MSE, MAE |

---

## Conexión con actividades anteriores

| Semana | Tema | Relación |
|--------|------|----------|
| Sem 5 | Optimización e hiperparámetros | Mismo pipeline de entrenamiento y evaluación |
| Sem 6 | Regularización | Early Stopping aplicado aquí para evitar sobreajuste |
| Sem 7 | Convolución / CNN | Arquitectura diferente; RNN procesa secuencias en vez de imágenes |
| Sem 9 | Transfer Learning | Ambas son estrategias de arquitectura avanzada |
