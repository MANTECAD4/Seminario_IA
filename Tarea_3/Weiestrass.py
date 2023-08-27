import numpy as np
import matplotlib.pyplot as plt

def wierstrass(x, y, a, b, k_max):
    result = 0
    for k in range(k_max):
        result += (a**k * np.cos(2 * np.pi * b**k * (x + 0.5)) - a**k*np.cos(np.pi*b**k)) + (a**k * np.cos(2 * np.pi * b**k * (y + 0.5)) - a**k*np.cos(np.pi*b**k))
    return result

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-0.5, 0.5, 150)
y = np.linspace(-0.5, 0.5, 150)
x, y = np.meshgrid(x, y)
z = wierstrass(x, y, a=0.5, b=3, k_max=20)

ax.plot_surface(x, y, z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Función de Weierstrass en 3D')
plt.show()
