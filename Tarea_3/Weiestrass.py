import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def wierstrass(x, y, a, b, num_terms):
    result = 0
    for i in range(num_terms):
        result += a**i * np.cos(2 * np.pi * b**i * (x + 0.5)) + a**i * np.cos(2 * np.pi * b**i * (y + 0.5))
    return result

# Crear una figura en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Crear valores para x e y
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
x, y = np.meshgrid(x, y)

# Evaluar la función de Wierstrass en 3D
z = wierstrass(x, y, a=0.5, b=3, num_terms=50)

# Crear la gráfica de la función de Wierstrass en 3D
ax.plot_surface(x, y, z, cmap='viridis')

# Configurar etiquetas de los ejes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Función de Wierstrass en 3D')

# Mostrar la gráfica
plt.show()
