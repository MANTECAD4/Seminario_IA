# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 15:14:02 2023

@author: Dell
"""

import numpy as np  # Importa la biblioteca NumPy para manipulación numérica
import matplotlib.pyplot as plt  # Importa la biblioteca Matplotlib para visualización

# Crear datos de ejemplo: genera arreglos de valores X e Y, y crea una malla (grid) X-Y
x = np.linspace(-10, 10, 70)  # Crea 100 valores equidistantes en el rango -5 a 5
y = np.linspace(-10, 10, 70)
x, y = np.meshgrid(x, y)  # Crea una malla de valores X e Y para usar en la gráfica

# Calcula los valores Z para una superficie en función de los valores X e Y
z = np.floor(x)**2+np.floor(y)**2

# Crear una figura en 3D
fig = plt.figure()  # Crea una figura para la gráfica
ax = fig.add_subplot(111, projection='3d')  # Agrega un subplot 3D a la figura

# Crear la gráfica de superficie
surface = ax.plot_surface(x, y, z, cmap='viridis')  # Crea la gráfica de superficie con colores viridis

# Agregar etiquetas y título
ax.set_xlabel('X')  # Agrega etiqueta al eje X
ax.set_ylabel('Y')  # Agrega etiqueta al eje Y
ax.set_zlabel('Z')  # Agrega etiqueta al eje Z
ax.set_title('Gráfica de la función Step en 3D')  # Agrega un título a la gráfica

# Mostrar la gráfica
plt.show()  # Muestra la gráfica en una ventana emergente