# Semana 14 - Aplicacion de los Conceptos de GANs en Google Colab

---

## Descripcion

En esta actividad se implementa una **Red Generativa Adversarial (GAN)** desde cero utilizando el dataset **MNIST** en Google Colab. El objetivo es comprender la logica de funcionamiento de una GAN a partir de sus dos componentes principales: el generador y el discriminador, asi como el proceso de entrenamiento adversarial entre ambos modelos.

---

## Contenido del Notebook

| Seccion | Descripcion |
|---|---|
| 1. Importaciones | Configuracion de librerias y semillas de aleatoriedad |
| 2. Preprocesamiento | Carga de MNIST, normalizacion a [-1,1] y aplanamiento |
| 3. Generador | Arquitectura Dense 100 -> 256 -> 512 -> 784 con LeakyReLU y tanh |
| 4. Discriminador | Arquitectura Dense 784 -> 512 -> 256 -> 1 con LeakyReLU, Dropout y sigmoid |
| 5. Compilacion | Compilacion independiente de D y pipeline G->D congelado |
| 6. Entrenamiento | Bucle adversarial comentado linea por linea (30 epocas, batch=128) |
| 7. Curvas de perdida | Grafico D-Loss y G-Loss con referencia al equilibrio teorico (ln 2 ~ 0.693) |
| 8. Visualizacion | Comparacion imagenes generadas vs. imagenes reales de MNIST |
| 9. Conclusiones | Analisis cuantitativo y cualitativo del proceso de entrenamiento |
| 10. Resumen | Tabla de hiperparametros y arquitectura utilizada |

---

## Arquitectura de la GAN

```
RUIDO (100 dims)
      |
  [GENERADOR]
  Dense(256) + LeakyReLU(0.2)
  Dense(512) + LeakyReLU(0.2)
  Dense(784) + tanh
      |
 Imagen sintetica (784 dims = 28x28)
      |
  [DISCRIMINADOR]
  Dense(512) + LeakyReLU(0.2) + Dropout(0.3)
  Dense(256) + LeakyReLU(0.2) + Dropout(0.3)
  Dense(1)   + sigmoid
      |
 Probabilidad: real (1) o falsa (0)
```

---

## Hiperparametros

| Parametro | Valor |
|---|---|
| Espacio latente | 100 dimensiones |
| Epocas | 30 |
| Batch size | 128 |
| Learning rate | 0.0002 |
| Beta_1 (Adam) | 0.5 |
| Dropout | 0.3 |
| LeakyReLU alpha | 0.2 |

---

## Librerias Utilizadas

- `tensorflow` / `keras` - Construccion y entrenamiento de los modelos
- `numpy` - Generacion de ruido, operaciones numericas
- `matplotlib` - Visualizacion de imagenes y curvas de perdida

> No se utilizaron librerias adicionales fuera de las anteriores.

---

## Como Ejecutar

1. Subir el archivo `Semana14_GANs.ipynb` a [Google Colab](https://colab.research.google.com)
2. Seleccionar un entorno de ejecucion con GPU: `Entorno de ejecucion > Cambiar tipo de entorno de ejecucion > GPU`
3. Ejecutar las celdas en orden con `Shift + Enter` o `Entorno de ejecucion > Ejecutar todo`

