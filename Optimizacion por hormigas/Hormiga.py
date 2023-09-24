import numpy as np
import matplotlib.pyplot as plt

# Función Weierstrass
def Weierstrass(x, y, a, b, k_max):
    result = 0
    for k in range(k_max):
        result += (a**k * np.cos(2 * np.pi * b**k * (x + 0.5)) - a**k*np.cos(np.pi*b**k)) + (a**k * np.cos(2 * np.pi * b**k * (y + 0.5)) - a**k*np.cos(np.pi*b**k))
    return result

# Parámetros de la optimización por colonia de hormigas
num_ants = 20
num_iterations = 100
alpha = 1.0  # Importancia de las feromonas
beta = 2.0   # Importancia de la función de coste
evaporation_rate = 0.5

# Espacio de búsqueda discretizado
x_range = np.linspace(-0.5, 0.5, 150)
y_range = np.linspace(-0.5, 0.5, 150)
x_grid, y_grid = np.meshgrid(x_range, y_range)

# Inicialización de feromonas en cada celda
pheromones = np.ones_like(x_grid)

# Bucle de optimización por colonia de hormigas
best_solution = None
best_cost = float('inf')

for iteration in range(num_iterations):
    solutions = []
    costs = []
    
    for ant in range(num_ants):
        x, y = np.random.choice(x_range), np.random.choice(y_range)
        cost = Weierstrass(x, y, a=0.5, b=3, k_max=20)
        solutions.append((x, y))
        costs.append(cost)
        
        if cost < best_cost:
            best_solution = (x, y)
            best_cost = cost
        
    # Actualización de feromonas
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            pheromones[i, j] *= (1 - evaporation_rate)
    
    for solution, cost in zip(solutions, costs):
        x_idx = np.argmin(np.abs(x_range - solution[0]))
        y_idx = np.argmin(np.abs(y_range - solution[1]))
        pheromones[x_idx, y_idx] += 1.0 / (cost + 1e-6)  # Agregar una pequeña constante para evitar la división por cero

# Graficar la superficie de la función Weierstrass y la mejor solución encontrada
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.linspace(-0.5, 0.5, 150)
y = np.linspace(-0.5, 0.5, 150)
x, y = np.meshgrid(x, y)
z = Weierstrass(x, y, a=0.5, b=3, k_max=20)

print("Mejor solución encontrada:")
print(f"X: {best_solution[0]}")
print(f"Y: {best_solution[1]}")
print(f"Z: {best_cost}")

print("Valor optimo: ", z.min())
surface = ax.plot_surface(x, y, z, cmap='viridis')
ax.scatter([best_solution[0]], [best_solution[1]], [best_cost], color='red', marker='o', s=100, label='Mejor Solución')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Gráfica de la función Weierstrass con optimización por colonia de hormigas')

plt.legend()
plt.show()


