# Implementación de Perceptrón Simple - CADI Deep Learning

## 🎯 Objetivo de la Actividad
Desarrollar una neurona artificial (perceptrón) desde sus fundamentos matemáticos para comprender el proceso de toma de decisiones binarias. El modelo simula una compuerta lógica **AND**, donde la neurona solo se activa (salida 1) si ambas entradas son verdaderas.

## 🧠 Arquitectura de la Solución
La implementación se basa en el cálculo del puntaje de pre-activación ($z$) mediante el producto punto entre entradas y pesos, sumando un sesgo:

$$z = \sum (x_i \cdot w_i) + b$$

### Componentes:
1. **Entradas (Inputs):** Valores binarios $[0, 1]$.
2. **Pesos (Weights):** Coeficientes que determinan la relevancia de cada entrada.
3. **Sesgo (Bias):** Valor que desplaza la función de activación para controlar el umbral de disparo.
4. **Función de Activación:** Función de paso de Heaviside (si $z \ge 0 \rightarrow 1$, de lo contrario $0$).

## 🧪 Pruebas Realizadas
Se configuraron pesos de $0.6$ y un sesgo de $-1.0$ para validar los cuatro estados de una tabla de verdad:
* **Caso [0,0]:** $z = -1.0$ (Salida: 0)
* **Caso [0,1]:** $z = -0.4$ (Salida: 0)
* **Caso [1,0]:** $z = -0.4$ (Salida: 0)
* **Caso [1,1]:** $z = 0.2$ (Salida: 1) ✅ *Único caso de activación.*

## 📈 Conclusiones Técnicas
* **Sensibilidad de los Pesos:** Al aumentar los pesos, la neurona se vuelve más sensible a las entradas, permitiendo que valores bajos de $x$ activen la salida.
* **Rol del Sesgo:** El sesgo actúa como un "umbral de exigencia". Un sesgo de $-1.0$ obliga a que la suma de las entradas multiplicadas por sus pesos supere la unidad para que la neurona responda positivamente. Sin el sesgo, la neurona se activaría con cualquier entrada mayor a cero.
Dicho de otro modo se volvería "muy fácil de convencer". Incluso con entradas [0,0], el puntaje $z$ sería $0.5$, por lo que la salida sería 1. Un sesgo positivo alto hace que la neurona esté siempre "encendida", mientras que un sesgo negativo (como el que usamos de $-1.0$) la hace selectiva y exigente.

## 🚀 Cómo Ejecutar
1. Clonar el repositorio.
2. Abrir `week2/Perceptron_Basico.ipynb` en Google Colab o Jupyter Notebook.
3. Ejecutar las celdas en orden secuencial. No requiere dependencias externas más allá de `pandas` para la visualización de resultados.
