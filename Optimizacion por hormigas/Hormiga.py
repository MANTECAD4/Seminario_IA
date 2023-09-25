import numpy as np
import matplotlib.pyplot as plt

# Función Weierstrass
def Weierstrass(x, y, a, b, k_max):
    result = 0
    for k in range(k_max):
        result += (a**k * np.cos(2 * np.pi * b**k * (x + 0.5)) - a**k*np.cos(np.pi*b**k)) + (a**k * np.cos(2 * np.pi * b**k * (y + 0.5)) - a**k*np.cos(np.pi*b**k))
    return result


# Parámetros de la optimización por colonia de hormigas
num_hormigas = 10
num_iterations = 80
alpha = 1.0  # Importancia de las feromonas
beta = 1.5   # Importancia de la función de coste
evaporation_rate = 0.6

# Espacio de búsqueda discretizado
x_range = np.linspace(-0.5, 0.5, 150)
y_range = np.linspace(-0.5, 0.5, 150)
x_grid, y_grid = np.meshgrid(x_range, y_range)

# Inicialización de feromonas en cada celda
pheromones = np.ones_like(x_grid)

# Bucle de optimización por colonia de hormigas
best_solution = None
best_cost = float('inf')
best_costs = []


for iteration in range(num_iterations):
    solutions = []
    costs = []
    
    for hormiga in range(num_hormigas):
        if(iteration%10==0):
            total_pheromones = pheromones.sum()
            pheromone_probabilities = pheromones / (total_pheromones + 1e-6)
            x_probs = pheromone_probabilities.sum(axis=1)  # Suma a lo largo del eje de las filas
            y_probs = pheromone_probabilities.sum(axis=0)  # Suma a lo largo del eje de las columnas
            x_probs /= x_probs.sum()
            y_probs /= y_probs.sum()
            x = np.random.choice(x_range, p=x_probs)
            y = np.random.choice(y_range, p=y_probs)
        else:
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
        pheromones[x_idx, y_idx] += 1.0 / (cost + 1e-6)  # Agregar una pequeña consthormigae para evitar la división por cero
    best_costs.append(best_cost)
    #print("iteracion: ",iteration,"->",best_cost,)

# Graficar la superficie de la función Weierstrass y la mejor solución encontrada
plt.plot(range(num_iterations), best_costs)
plt.xlabel('Iteración')
plt.ylabel('Mejor Costo')
plt.title('Evolución del Mejor Costo en Cada Iteración')
plt.grid(True)
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


