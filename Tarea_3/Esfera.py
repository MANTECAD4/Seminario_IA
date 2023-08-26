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
x = np.linspace(-100, 100, 200)  # Crea 100 valores equidistantes en el rango -5 a 5
y = np.linspace(-100, 100, 200)
x, y = np.meshgrid(x, y)  # Crea una malla de valores X e Y para usar en la gráfica

z = x**2 + y**2  # Coordenada z en función de theta

# Crea una figura en 3D
fig = plt.figure()  # Crea una figura para la gráfica
ax = fig.add_subplot(111, projection='3d')  # Agrega un subplot 3D a la figura

# Crear la gráfica de la esfera
ax.plot_surface(x, y, z, color='b', alpha=0.6)  # Crea la gráfica de la esfera con color azul y transparencia 0.6

# Mostrar la gráfica
plt.show()  # Muestra la gráfica en una ventana emergente
