# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 16:03:37 2023

Tarea 3 - Funciones de prueba para optimización multimodal, multiobjetivo y multidimensional

@author: Dell
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Importa la herramienta 3D de Matplotlib

# Crear datos de ejemplo: genera arreglos de valores X e Y, y crea una malla (grid) X-Y
x = np.linspace(-10, 10, 50)  # Crea 50 valores equidistantes en el rango -10 a 10
y = np.linspace(-10, 10, 50)
x, y = np.meshgrid(x, y)  # Crea una malla de valores X e Y para usar en la gráfica

z = x**2 + y**2  # Coordenada z en función de theta

# Crea una figura en 3D
fig = plt.figure()  # Crea una figura para la gráfica
ax = fig.add_subplot(111, projection='3d')  # Agrega un subplot 3D a la figura

# Crear la gráfica de la esfera
surface = ax.plot_surface(x, y, z, cmap='viridis')  # Crea la gráfica de superficie con colores viridis

# Agregar etiquetas y título
ax.set_xlabel('X')  # Agrega etiqueta al eje X
ax.set_ylabel('Y')  # Agrega etiqueta al eje Y
ax.set_zlabel('Z')  # Agrega etiqueta al eje Z
ax.set_title('Gráfica de la esfera unimodal en 3D')  # Agrega un título a la gráfica

# Mostrar la gráfica
plt.show()  # Muestra la gráfica en una ventana emergente
