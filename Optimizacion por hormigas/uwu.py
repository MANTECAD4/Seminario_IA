import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Datos para la gráfica 3D
x3d = np.linspace(-5, 5, 100)
y3d = np.linspace(-5, 5, 100)
X3D, Y3D = np.meshgrid(x3d, y3d)
Z3D = np.sin(np.sqrt(X3D**2 + Y3D**2))

# Datos para la gráfica 2D
x2d = np.linspace(-5, 5, 100)
y2d = np.sin(x2d)

# Crear una figura y ejes 3D para la gráfica 3D
fig = plt.figure()
ax3d = fig.add_subplot(111, projection='3d')

# Graficar la superficie en 3D
ax3d.plot_surface(X3D, Y3D, Z3D, cmap='viridis')

# Etiquetas y título para la gráfica 3D
ax3d.set_xlabel('X etiqueta 3D')
ax3d.set_ylabel('Y etiqueta 3D')
ax3d.set_zlabel('Z etiqueta 3D')
plt.title('Gráfica 3D')

# Crear una segunda figura para la gráfica 2D
plt.figure()

# Graficar la curva en 2D
plt.plot(x2d, y2d, 'r')

# Etiquetas y título para la gráfica 2D
plt.xlabel('X etiqueta 2D')
plt.ylabel('Y etiqueta 2D')
plt.title('Gráfica 2D')

plt.show()
