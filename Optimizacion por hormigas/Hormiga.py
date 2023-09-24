# -*- coding: utf-8 -*-
"""

Tarea 3 - Funciones de prueba para optimización multimodal, multiobjetivo y multidimensional

"""
import numpy as np  
import matplotlib.pyplot as plt  

def Step(x,y):
    return np.floor(x)**2+np.floor(y)**2

x = np.linspace(-10, 10, 70)  # Crea 100 valores equidistantes en el rango -5 a 5
y = np.linspace(-10, 10, 70)
x, y = np.meshgrid(x, y) 

z = Step(x,y)

fig = plt.figure() 
ax = fig.add_subplot(111, projection='3d')  

surface = ax.plot_surface(x, y, z, cmap='viridis')  

ax.set_xlabel('X') 
ax.set_ylabel('Y') 
ax.set_zlabel('Z') 
ax.set_title('Gráfica de la función Step')  

plt.show() 