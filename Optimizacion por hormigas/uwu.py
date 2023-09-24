import numpy as np
import matplotlib.pyplot as plt

# Función Step
def Step(x, y):
    return np.floor(x) ** 2 + np.floor(y) ** 2

# Parámetros de la optimización por colonia de hormigas
num_ants = 10
num_iterations = 100
alpha = 1.0
beta = 2.0
evaporation_rate = 0.5

# Espacio de búsqueda discretizado
x_range = np.linspace(-10, 10, 100)
y_range = np.linspace(-10, 10, 100)
x_grid, y_grid = np.meshgrid(x_range, y_range)

# Inicialización de feromonas en cada celda
pheromones = np.ones_like(x_grid)

# Listas para almacenar los mejores costos en cada iteración
best_costs = []

# Bucle de optimización por colonia de hormigas
best_solution = None
best_cost = float('inf')

for iteration in range(num_iterations):
    solutions = []
    costs = []
    
    for ant in range(num_ants):
        x, y = np.random.choice(x_range), np.random.choice(y_range)
        cost = Step(x, y)
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
        pheromones[x_idx, y_idx] += 1.0 / (cost + 1e-6)
    
    # Almacenar el mejor costo en esta iteración
    best_costs.append(best_cost)

# Graficar la evolución de las soluciones
plt.plot(range(num_iterations), best_costs)
plt.xlabel('Iteración')
plt.ylabel('Mejor Costo')
plt.title('Evolución del Mejor Costo en Cada Iteración')
plt.grid(True)
plt.show()

print("Mejor solución encontrada:")
print(f"X: {best_solution[0]}")
print(f"Y: {best_solution[1]}")
print(f"Z: {best_cost}")
