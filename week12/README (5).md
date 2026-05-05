# Semana 12 – Mecanismo de Atención y Transformer Básico para Series de Tiempo

**CADI Deep Learning | Universidad de Cundinamarca**

---

## Objetivo

Implementar el **mecanismo de atención** aplicado a datos secuenciales y desarrollar una aproximación funcional de un modelo tipo **Transformer**, evaluando su desempeño frente al modelo LSTM desarrollado en la Semana 11.

---

## Dataset

**Airline Passengers** (Box & Jenkins, 1976)  
144 registros mensuales del número de pasajeros aéreos internacionales (1949–1960).  
Mismo dataset de la Semana 11 — permite comparación directa entre modelos.

---

## Arquitecturas implementadas

| Modelo | Mecanismo central | Parámetros |
|--------|------------------|------------|
| LSTM + Atención | Estado oculto LSTM + pesos de atención de Bahdanau | ~25 000 |
| Transformer Básico | Multi-Head Self-Attention pura (sin recurrencia) | ~15 000 |

---

## Estructura del notebook

| Sección | Contenido |
|---------|-----------|
| 1. Importaciones | Librerías y semillas de reproducibilidad |
| 2. Carga del dataset | Descarga y visualización de Airline Passengers |
| 3. Preprocesamiento | Normalización, división temporal, ventana deslizante |
| 4. Capa de Atención | Implementación de `AttentionLayer` (Bahdanau simplificado) |
| 5. Modelo LSTM + Atención | Arquitectura Input → LSTM → Atención → Dense |
| 6. Entrenamiento (Atención) | Fit con EarlyStopping, curvas de pérdida |
| 7. Transformer Básico | Bloque encoder: Multi-Head Attention + Add&Norm + FFN |
| 8. Entrenamiento (Transformer) | Fit con EarlyStopping, curvas de pérdida |
| 9. Evaluación comparativa | MSE, RMSE, MAE en escala original |
| 10. Visualización predicciones | Comparación real vs. predicho en test |
| 11. Pesos de Atención | Mapa de calor de los pesos α por paso temporal |
| 12. Análisis corto/largo plazo | Distribución de la atención por posición temporal |
| 13. Tabla resumen | Comparación LSTM (Sem. 11) vs. Atención vs. Transformer |
| 14. Conclusiones | 5 conclusiones técnicas sobre ventajas y limitaciones |

---

## Cómo ejecutar

1. Abrir `Semana12_Atencion_Transformer.ipynb` en [Google Colab](https://colab.research.google.com/)
2. Activar GPU *(opcional)*: *Entorno de ejecución → Cambiar tipo de entorno → T4 GPU*
3. **Ejecutar todo** (`Ctrl + F9`)
4. Tiempo estimado: **5–8 minutos** en CPU

---

## Librerías utilizadas

- `tensorflow / keras` — construcción y entrenamiento de modelos, `MultiHeadAttention`, `LayerNormalization`
- `numpy` — operaciones numéricas y álgebra lineal
- `pandas` — carga y manipulación del dataset, tabla de resultados
- `matplotlib` — visualización de curvas, predicciones y pesos de atención
- `scikit-learn` — `MinMaxScaler`, `mean_squared_error`, `mean_absolute_error`

---

## Conceptos clave

### Mecanismo de Atención (Bahdanau)

Permite al modelo ponderar la importancia relativa de cada paso temporal:

$$e_t = \tanh(W \cdot h_t + b), \quad \alpha_t = \text{softmax}(e_t), \quad \text{context} = \sum_t \alpha_t \cdot h_t$$

### Transformer (encoder-only)

Reemplaza la recurrencia con Multi-Head Self-Attention, procesando toda la secuencia en paralelo:

```
Input → Proyección lineal → [MultiHeadAttention → Add&Norm → FFN → Add&Norm] → GlobalAvgPool → Dense → Output
```

---

## Comparación con Semana 11

| Aspecto | LSTM (Semana 11) | LSTM + Atención | Transformer |
|---------|-----------------|-----------------|-------------|
| Dependencias largas | Limitada | Mejorada | Excelente |
| Interpretabilidad | Baja | Alta (pesos α) | Media |
| Paralelización | No | Parcial | Total |
| Datos requeridos | Pocos | Pocos | Muchos |

---

## Evidencia en GitHub

El notebook `Semana12_Atencion_Transformer.ipynb` y este `README.md` deben subirse al repositorio del curso como evidencia reproducible.
