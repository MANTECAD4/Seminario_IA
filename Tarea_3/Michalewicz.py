# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:03:37 2023

Tarea 3 - Funciones de prueba para optimización multimodal, multiobjetivo y multidimensional

@author: Dell
"""
import numpy as np  # Importa la biblioteca NumPy para manipulación numérica
import matplotlib.pyplot as plt  # Importa la biblioteca Matplotlib para visualización

x = np.linspace(0, np.pi, 100)  # Crea 100 valores equidistantes en el rango 0 a pi
y = np.linspace(0, np.pi, 100)
x, y = np.meshgrid(x, y)  # Crea una malla de valores X e Y para usar en la gráfica
M=10# Parámetro de ajuste para la función de Michalewicz
z = -((np.sin(x) * np.sin((1 * x**2) / np.pi)**(2*M)) + (np.sin(y) * np.sin((2 * y**2) / np.pi)**(2*M)))

fig = plt.figure()  # Crea una figura para la gráfica
ax = fig.add_subplot(111, projection='3d')  # Agrega un subplot 3D a la figura

surface = ax.plot_surface(x, y, z, cmap='viridis')  # Crea la gráfica de superficie con colores viridis

ax.set_xlabel('X')  # Agrega etiqueta al eje X
ax.set_ylabel('Y')  # Agrega etiqueta al eje Y
ax.set_zlabel('Z')  # Agrega etiqueta al eje Z
ax.set_title('Gráfica de la función Michalewicz en 3D')  # Agrega un título a la gráfica

plt.show()  # Muestra la gráfica en una ventana emergente