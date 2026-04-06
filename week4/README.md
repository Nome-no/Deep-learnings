# 🩺 Optimización de Redes Neuronales: Predicción de Diabetes
**Institución:** Universidad de Cundinamarca  
**Especialización:** Inteligencia Artificial  
**Materia:** Deep Learning (Semana 4)

---

## 📋 Descripción del Proyecto
Este proyecto consiste en el diseño, implementación y evaluación de una red neuronal profunda aplicada al dataset **Diabetes** de OpenML. El objetivo central es realizar un análisis comparativo sobre cómo diferentes estrategias de **optimización** y la variación de la **tasa de aprendizaje (Learning Rate)** afectan la convergencia y precisión del modelo.

## 🚀 Tecnologías Utilizadas
* **Google Colab** (Entorno de ejecución)
* **TensorFlow / Keras** (Arquitectura de la red)
* **Scikit-Learn** (Preprocesamiento y carga de datos)
* **Matplotlib** (Visualización de métricas)

---

## 📊 Metodología Experimental

### 1. Configuración de Escenarios
Se compararon tres estados de entrenamiento para observar el impacto real de la optimización:
* **Sin Optimización:** Línea base con pesos aleatorios (Sin entrenamiento).
* **SGD (Stochastic Gradient Descent):** Optimizador clásico con tasa de 0.01.
* **Adam (Adaptive Moment Estimation):** Optimizador adaptativo con tasa de 0.001.

### 2. Prueba de Sensibilidad (Learning Rate)
Se evaluó el comportamiento de SGD bajo tres configuraciones distintas:
| Tasa de Aprendizaje | Impacto Observado |
| :--- | :--- |
| **0.1 (Alto)** | Convergencia rápida pero con riesgo de inestabilidad y oscilaciones. |
| **0.01 (Medio)** | Comportamiento estable y aprendizaje constante. |
| **0.0001 (Bajo)** | Aprendizaje extremadamente lento (estancamiento). |

---

## 📈 Resultados y Evidencias
El notebook genera visualizaciones que permiten contrastar la evolución del *Loss* (pérdida). 

* **Conclusión de Optimizadores:** **Adam** demostró ser superior a SGD, alcanzando un error menor en menos épocas. Esto se debe a su capacidad de ajustar la tasa de aprendizaje de forma individual para cada parámetro.
* **Importancia del Escalado:** Se aplicó `StandardScaler` a los datos, ya que los algoritmos basados en gradiente son altamente sensibles a la escala de las variables de entrada.

---

## 📝 Conclusiones Técnicas
1.  **Motor de Aprendizaje:** La comparación con el modelo "Sin Optimización" demuestra que la arquitectura por sí sola no tiene capacidad predictiva; el optimizador es el motor que permite el ajuste de pesos.
2.  **Eficiencia de Adam:** Adam es la opción recomendada para este dataset debido a que gestiona mejor los momentos y reduce la necesidad de un ajuste manual exhaustivo del Learning Rate.
3.  **Trazabilidad:** Los resultados son 100% reproducibles gracias al uso de semillas aleatorias (`random_seed`) y el manejo estructurado de hiperparámetros.

---

## 🛠️ Cómo ejecutar
1. Abrir el archivo `.ipynb` en **Google Colab**.
2. Ejecutar las celdas en orden secuencial.
3. Las gráficas comparativas se generarán automáticamente al final del entrenamiento.

---
**Autor:** Fabian Buitrago  
