# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:03:37 2023

Tarea 3 - Funciones de prueba para optimización multimodal, multiobjetivo y multidimensional

@author: Dell
"""
import numpy as np  # Importa la biblioteca NumPy para manipulación numérica
import matplotlib.pyplot as plt  # Importa la biblioteca Matplotlib para visualización
from mpl_toolkits.mplot3d import Axes3D  # Importa la herramienta 3D de Matplotlib

# Crear datos de ejemplo: genera arreglos de valores X e Y, y crea una malla (grid) X-Y
x = np.linspace(0, np.pi, 100)  # Crea 100 valores equidistantes en el rango -5 a 5
y = np.linspace(0, np.pi, 100)
x, y = np.meshgrid(x, y)  # Crea una malla de valores X e Y para usar en la gráfica
# Parámetro de ajuste para la función de Michalewicz
M=20
# Calcula los valores Z para una superficie en función de los valores X e Y
z = -(np.sin(x) * np.sin((1 * x**2) / np.pi)**M) - (np.sin(y) * np.sin((2 * y**2) / np.pi)**M)

# Crear una figura en 3D
fig = plt.figure()  # Crea una figura para la gráfica
ax = fig.add_subplot(111, projection='3d')  # Agrega un subplot 3D a la figura

# Crear la gráfica de superficie
surface = ax.plot_surface(x, y, z, cmap='viridis')  # Crea la gráfica de superficie con colores viridis

# Agregar etiquetas y título
ax.set_xlabel('X')  # Agrega etiqueta al eje X
ax.set_ylabel('Y')  # Agrega etiqueta al eje Y
ax.set_zlabel('Z')  # Agrega etiqueta al eje Z
ax.set_title('Gráfica de la función Michalewicz en 3D')  # Agrega un título a la gráfica

# Mostrar la gráfica
plt.show()  # Muestra la gráfica en una ventana emergente