# Semana 15 - Data Journey, Acceso y Manipulacion de Datos, Monitoreo y Logging, y Model Serving con Weights & Biases
---

## Descripcion

En esta actividad se desarrolla un **flujo completo de aprendizaje profundo** sobre el dataset **MNIST** en Google Colab, cubriendo el ciclo de vida del dato y del modelo. El objetivo es que el equipo comprenda que un modelo no termina en el entrenamiento, sino que hace parte de un proceso mas amplio que incluye la preparacion de datos, el seguimiento de metricas, la trazabilidad de experimentos y alternativas para ponerlos en operacion.

El proyecto utiliza **Weights & Biases (W&B)** como plataforma de seguimiento de experimentos, integrando monitoreo en tiempo real, registro de hiperparametros y versionado de artefactos del modelo.

---

## Contenido del Notebook

| Seccion | Descripcion |
|---|---|
| 1. Importaciones | Instalacion de wandb, configuracion de librerias y semillas |
| 2. Data Journey | Ingestion, inspeccion, normalizacion, aplanamiento y particion documentados paso a paso |
| 3. Acceso y Manipulacion | Visualizacion por clase, imagenes crudas vs. normalizadas, histograma de intensidades |
| 4. Arquitectura MLP | Red densa Dense(256) + Dropout + Dense(128) + Dropout + Dense(10, softmax) |
| 5. Configuracion W&B | Login, inicializacion del experimento, registro de hiperparametros |
| 6. Entrenamiento + Logging | Entrenamiento con WandbCallback y EarlyStopping, metricas por epoca |
| 7. Curvas de Entrenamiento | Loss y Accuracy por epoca, registradas en W&B como imagen |
| 8. Evaluacion y Confusion | Test accuracy, matriz de confusion manual 10x10, accuracy por clase |
| 9. Model Serving | Guardado .keras, registro como artefacto W&B, carga y prediccion sobre nuevas muestras |
| 10. Conclusiones | Analisis cuantitativo y cualitativo del ciclo de vida completo |
| 11. Resumen Final | Tabla de hiperparametros, arquitectura e integracion W&B |

---

## Arquitectura del Modelo

```
INPUT (784 dimensiones = 28x28 pixeles)
       |
  Dense(256, ReLU)
       |
  Dropout(0.3)
       |
  Dense(128, ReLU)
       |
  Dropout(0.3)
       |
  Dense(10, softmax)
       |
 Clase predicha (0-9) con probabilidad
```

---

## Flujo del Experimento con Weights & Biases

```
wandb.init()  →  config: hiperparametros
      |
 model.fit() + WandbCallback
      |          ├── loss por epoca
      |          ├── val_loss por epoca
      |          ├── accuracy por epoca
      |          └── histogramas de pesos
      |
 wandb.log() →  test_loss, test_accuracy
      |          curvas_entrenamiento (imagen)
      |          matriz_confusion (imagen)
      |
 wandb.Artifact() → modelo .keras versionado
      |
 run.finish() → experimento cerrado en dashboard
```

---

## Hiperparametros

| Parametro | Valor |
|---|---|
| Arquitectura | MLP (red densa) |
| Epocas maximas | 15 |
| Batch size | 128 |
| Learning rate | 0.001 |
| Optimizador | Adam |
| Funcion de perdida | Sparse Categorical Crossentropy |
| Dropout | 0.3 |
| Neuronas capa 1 | 256 |
| Neuronas capa 2 | 128 |
| EarlyStopping patience | 4 epocas |
| Validacion | 15% del conjunto de entrenamiento |

---

## Librerias Utilizadas

- `tensorflow` / `keras` - Construccion, compilacion y entrenamiento del modelo MLP
- `numpy` - Manipulacion de arreglos, construccion manual de la matriz de confusion
- `matplotlib` - Visualizacion de imagenes, histogramas y curvas de entrenamiento
- `wandb` - Seguimiento de experimentos, registro de metricas, artefactos y dashboard

> `wandb` se instala con `!pip install wandb --quiet` en la primera celda del notebook.  
> El resto de librerias estan preinstaladas en Google Colab.

---

## Como Ejecutar

1. Subir el archivo `Semana15_Actividad15.ipynb` a [Google Colab](https://colab.research.google.com)
2. Crear una cuenta gratuita en [wandb.ai](https://wandb.ai) y obtener la API Key desde `Settings > API Keys`
3. Ejecutar las celdas en orden con `Shift + Enter` o `Entorno de ejecucion > Ejecutar todo`
4. Cuando se ejecute la celda 5, ingresar la API Key de W&B cuando se solicite
5. Al finalizar, el dashboard del experimento estara disponible en `https://wandb.ai/<usuario>/semana15-deep-learning-mnist`

> **Sin cuenta W&B:** el codigo detecta automaticamente si no hay autenticacion y ejecuta en `mode='disabled'`, completando todas las secciones localmente sin registro remoto.

---

## Correspondencia con la Rubrica

| Criterio | Implementacion |
|---|---|
| **Investigacion y comprension de conceptos clave** | Secciones 2-3: Data Journey documentado etapa por etapa; acceso y manipulacion con visualizacion comparativa; conceptos de monitoreo y Model Serving explicados en celdas Markdown |
| **Implementacion practica en Weights & Biases** | Secciones 5-6-9: wandb.init() con config completo, WandbCallback integrado al entrenamiento, artefacto del modelo versionado con wandb.Artifact() |
| **Monitoreo y logging del modelo** | Secciones 6-7-8: metricas por epoca (loss, val_loss, accuracy, val_accuracy), histogramas de pesos, curvas de entrenamiento y matriz de confusion registradas en W&B con wandb.log() |
| **Colaboracion en el trabajo en equipo** | Codigo organizado en secciones claras con responsabilidades separadas; comentado linea a linea para facilitar la revision entre miembros del equipo |
| **Documentacion y presentacion del proyecto** | Notebook con celdas Markdown explicativas por seccion, tablas resumen, arquitectura en ASCII, README detallado con flujo del experimento y correspondencia con la rubrica |
