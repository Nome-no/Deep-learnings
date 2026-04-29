
# Semana 7 – Convolución de Matrices con Padding y Stride

**CADI Deep Learning | Universidad de Cundinamarca**

## Objetivo

Implementar manualmente la operación de convolución 2D con NumPy y aplicarla sobre una imagen real, demostrando con evidencia práctica cómo el **padding** y el **stride** modifican las dimensiones del mapa de características y la información preservada.

---

## Qué se comparó

| Variable | Valores evaluados | Variable fija |
|---|---|---|
| **Padding** | 0 (valid) vs 1 (same) | stride = 1 |
| **Stride** | 1, 2, 4 | padding = 1 |

En todas las comparaciones se usó el **mismo kernel Sobel 3×3** para garantizar que la única variable que cambia es la que se está analizando.

---

## Evidencia principal

- `padding=0` con imagen 512×512 y kernel 3×3 → salida **510×510** (se pierden los bordes).  
- `padding=1` → salida **512×512** (dimensiones preservadas, bordes incluidos).  
- Duplicar el stride de 1→2 reduce la salida a la mitad (**256×256**); stride=4 la reduce a **128×128**.

---

## Cómo ejecutar

1. Abrir [`Semana7_CNN_Convolucion.ipynb`](./Semana7_CNN_Convolucion.ipynb) en [Google Colab](https://colab.research.google.com/).
2. Ir a **Runtime → Run all** (o `Ctrl+F9`).
3. No se requieren instalaciones adicionales — todas las librerías (`numpy`, `matplotlib`, `scikit-image`) están disponibles por defecto en Colab.

---

## Estructura del notebook

| Sección | Contenido |
|---|---|
| 1. Importaciones | NumPy, Matplotlib, scikit-image |
| 2. Implementación manual | Función `convolve2d` con padding y stride |
| 3. Verificación | Prueba sobre matriz 4×4 controlada |
| 4. Imagen real | Carga y normalización de imagen cameraman 512×512 |
| 5. Kernel base | Definición del filtro Sobel (único en todo el notebook) |
| 6. Comparación Padding | valid vs same, mismas condiciones |
| 7. Comparación Stride | stride 1/2/4, dimensiones resultantes |
| 8. Análisis cuantitativo | Media absoluta y desviación estándar por configuración |
| 9. Zoom en bordes | Inspección visual del impacto del padding en las orillas |
| 10. Conclusiones | 3 puntos técnicos basados en la evidencia |
