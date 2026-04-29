# 🧠 Semana 3 — Backpropagation y Funciones de Activación en Redes Neuronales

**Curso:** CADI Deep Learning — Universidad de Cundinamarca  
**Docente:** Mónica Fonseca

---

## 🎯 Objetivo

Implementar y validar el proceso de aprendizaje de una red neuronal mediante **backpropagation** y el uso de **funciones de activación** (Sigmoide y ReLU), evidenciando cómo la red ajusta sus parámetros (pesos y sesgos) para reducir el error durante el entrenamiento y cómo la elección de la función de activación influye en el comportamiento y la capacidad de aprendizaje del modelo.

---

## 🔧 ¿Qué se implementó?

| Componente | Descripción |
|---|---|
| **Funciones de activación** | Sigmoide y ReLU implementadas desde cero, incluyendo sus derivadas para backpropagation. |
| **Red neuronal configurable** | Clase `RedNeuronal` con arquitectura definible por lista de capas (`[2, 4, 1]`), selección de activación y entrenamiento completo. |
| **Forward pass** | Propagación de la entrada por todas las capas, guardando cada activación intermedia. |
| **Pérdida (MSE)** | Cálculo del Error Cuadrático Medio en cada época para medir el progreso del entrenamiento. |
| **Backpropagation** | Cálculo de deltas capa por capa (de salida a entrada) y actualización de pesos con descenso de gradiente. |

---

## 🧪 Pruebas realizadas

Se utilizó el problema **XOR** como dataset de clasificación binaria (4 casos posibles), que no es linealmente separable y requiere una red con capa oculta para resolverse. Se entrenaron dos redes con la misma arquitectura `[2, 4, 1]` e idénticos pesos iniciales, diferenciadas únicamente por la función de activación (Sigmoide vs ReLU), permitiendo una comparación directa del proceso de convergencia, la velocidad de descenso del error y la calidad de las predicciones finales.

---

## 📊 Resultados principales

Ambas redes (Sigmoide y ReLU) alcanzaron el **100% de accuracy** en el dataset XOR tras el entrenamiento, aunque con curvas de pérdida distintas: Sigmoide desciende de forma más suave y gradual, mientras que ReLU puede mostrar caídas más abruptas dependiendo de la tasa de aprendizaje.

---

## ▶️ Cómo ejecutar

1. Abrir el archivo `Semana3_Mejorado.ipynb` en [Google Colab](https://colab.research.google.com/).
2. Ir a **Entorno de ejecución → Ejecutar todo** (o `Ctrl + F9`).
3. No se requiere instalar librerías adicionales: el notebook usa únicamente `numpy`, `pandas` y `matplotlib`, todas disponibles por defecto en Colab.
4. Las celdas deben ejecutarse en orden secuencial: las funciones de activación deben estar definidas antes de construir la red.

---

## 📁 Estructura del repositorio

```
week3/
├── Semana3_Mejorado.ipynb   # Notebook principal ejecutable
└── README.md                # Este archivo
```
