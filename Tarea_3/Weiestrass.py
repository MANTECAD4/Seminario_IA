import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def wierstrass(x, y, a, b, k_max):
    result = 0
    for k in range(k_max):
        result += (a**k * np.cos(2 * np.pi * b**k * (x + 0.5)) - a**k*np.cos(np.pi*b**k)) + (a**k * np.cos(2 * np.pi * b**k * (y + 0.5)) - a**k*np.cos(np.pi*b**k))
    return result

# Crear una figura en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Crear valores para x e y
x = np.linspace(-0.5, 0.5, 150)
y = np.linspace(-0.5, 0.5, 150)
x, y = np.meshgrid(x, y)

# Evaluar la función de Wierstrass en 3D
z = wierstrass(x, y, a=0.5, b=3, k_max=20)

# Crear la gráfica de la función de Wierstrass en 3D
ax.plot_surface(x, y, z, cmap='viridis')

# Configurar etiquetas de los ejes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Función de Weierstrass en 3D')

# Mostrar la gráfica
plt.show()
