# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:03:37 2023

Tarea 3 - Funciones de prueba para optimización multimodal, multiobjetivo y multidimensional

@author: Dell
"""
import numpy as np  # Importa la biblioteca NumPy para manipulación numérica
import matplotlib.pyplot as plt  # Importa la biblioteca Matplotlib para visualización

x = np.linspace(-512, 512, 500)  # Crea 100 valores equidistantes en el rango -5 a 5
y = np.linspace(-512, 512, 500)
x, y = np.meshgrid(x, y)  # Crea una malla de valores X e Y para usar en la gráfica

z = -(y + 47) * np.sin(np.sqrt(np.abs(y + x/2 + 47))) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))

fig = plt.figure()  # Crea una figura para la gráfica
ax = fig.add_subplot(111, projection='3d')  # Agrega un subplot 3D a la figura

surface = ax.plot_surface(x, y, z, cmap='viridis')  # Crea la gráfica de superficie con colores viridis

ax.set_xlabel('X')  # Agrega etiqueta al eje X
ax.set_ylabel('Y')  # Agrega etiqueta al eje Y
ax.set_zlabel('Z')  # Agrega etiqueta al eje Z
ax.set_title('Gráfica de la función EggHolder en 3D')  # Agrega un título a la gráfica

plt.show()  # Muestra la gráfica en una ventana emergente
