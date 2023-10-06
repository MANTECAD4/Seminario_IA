
import numpy as np
import matplotlib.pyplot as plt

# Función Weierstrass
def Weierstrass(x, y, a, b, k_max):
    result = 0
    for k in range(k_max):
        result += (a**k * np.cos(2 * np.pi * b**k * (x + 0.5)) - a**k*np.cos(np.pi*b**k)) + (a**k * np.cos(2 * np.pi * b**k * (y + 0.5)) - a**k*np.cos(np.pi*b**k))
    return result

# Parámetros de PSO
num_particulas = 30         #Numero de particulas
num_iteraciones = 100       #Numero de iteraciones
lb = 0.5                    #Limite superior
ub = -0.5                   #Limite inferior
num_puntos=200              #Numero de puntos del espacio discretizado
w_max = 0.9                 #Coeficientes de incercia
w_min = 0.2
c1, c2 = 2,2                #Coeficientes de atracción personal y social
vMax = (ub - lb) * 0.2      #Velocidad máxima
vMin  = -vMax               #Velocidad mínima

# Espacio de búsqueda discretizado
rango_x = np.linspace(ub, lb, num_puntos)
rango_y = np.linspace(ub, lb, num_puntos)
x_grid, y_grid = np.meshgrid(rango_x, rango_y)

z = Weierstrass(x_grid, y_grid, a=0.5, b=3, k_max=20)
#Iniciliaización de las partículas
particulas = []
for i in range(num_particulas):
    particula = {
        'x': np.random.choice(rango_x),
        'y': np.random.choice(rango_y),
        'pbest':{'x': None, 'y': None},
        'costo': float('inf'),
        'gbest': {'x': None, 'y': None}
    }
    particulas.append(particula)

# Bucle de optimización por enjambre de particulas
for iteracion in range(num_iteraciones):
    pass


# Grafica de la función Weierstrass y la evolución de la mejor solución encontrada
"""
plt.plot(range(num_iteraciones), best_costs)
plt.xlabel('Iteración')
plt.ylabel('Mejor Costo')
plt.title('Evolución del Mejor Costo en Cada Iteración')
plt.grid(True)


print("Mejor solución encontrada:")
print(f"X: {mejor_solucion[0]}")
print(f"Y: {mejor_solucion[1]}")
print(f"Z: {mejor_costo}")

print("Valor optimo: ", z.min())
ax.scatter([mejor_solucion[0]], [mejor_solucion[1]], [mejor_costo], color='red', marker='o', s=100, label='Mejor Solución')
"""

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(x_grid, y_grid, z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Gráfica de la función Weierstrass minimizada por PSO')

plt.legend()
plt.show()