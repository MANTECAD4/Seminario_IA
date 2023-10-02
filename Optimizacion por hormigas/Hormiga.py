import numpy as np
import matplotlib.pyplot as plt

# Función Weierstrass
def Weierstrass(x, y, a, b, k_max):
    result = 0
    for k in range(k_max):
        result += (a**k * np.cos(2 * np.pi * b**k * (x + 0.5)) - a**k*np.cos(np.pi*b**k)) + (a**k * np.cos(2 * np.pi * b**k * (y + 0.5)) - a**k*np.cos(np.pi*b**k))
    return result

# Parámetros de la optimización por colonia de hormigas
num_hormigas = 50
num_iteraciones = 100
alpha = 1.0  
beta = 1.5   
evaporacion = 0.5

# Espacio de búsqueda discretizado
rango_x = np.linspace(-0.5, 0.5, 200)
rango_y = np.linspace(-0.5, 0.5, 200)
x_grid, y_grid = np.meshgrid(rango_x, rango_y)

# Inicialización de feromonas en cada celda
feromonas = np.ones_like(x_grid)

mejor_solucion = None
mejor_costo= float('inf')
best_costs = []
prob_random = 0.3

# Bucle de optimización por colonia de hormigas
for iteracion in range(num_iteraciones):
    soluciones = []
    costos = []
    
    for hormiga in range(num_hormigas):
        rand_num = np.random.rand()
        
        if rand_num < prob_random:
            # Asignación aleatoria
            x, y = np.random.choice(rango_x), np.random.choice(rango_y)
        else:
            # Asignación por feromonas
            total_feromonas = feromonas.sum()
            probabilidad_feromonas = feromonas / (total_feromonas + 1e-6)
            x_probs = probabilidad_feromonas.sum(axis=1) 
            y_probs = probabilidad_feromonas.sum(axis=0)  
            x_probs /= x_probs.sum()
            y_probs /= y_probs.sum()
            x = np.random.choice(rango_x, p=x_probs)
            y = np.random.choice(rango_y, p=y_probs)

        cost = Weierstrass(x, y, a=0.5, b=3, k_max=20)
        soluciones.append((x, y))
        costos.append(cost)
        
        if cost < mejor_costo:
            mejor_solucion = (x, y)
            mejor_costo= cost
        
    # Actualización de feromonas
    for i in range(len(rango_x)):
        for j in range(len(rango_y)):
            feromonas[i, j] *= (1 - evaporacion)
    
    for solution, cost in zip(soluciones, costos):
        x_idx = np.argmin(np.abs(rango_x - solution[0]))
        y_idx = np.argmin(np.abs(rango_y - solution[1]))
        feromonas[x_idx, y_idx] += 1.0 / (cost + 1e-6)  
    best_costs.append(mejor_costo)

# Grafica de la función Weierstrass y la evolución de la mejor solución encontrada
plt.plot(range(num_iteraciones), best_costs)
plt.xlabel('Iteración')
plt.ylabel('Mejor Costo')
plt.title('Evolución del Mejor Costo en Cada Iteración')
plt.grid(True)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x, y = np.meshgrid(rango_x, rango_y)
z = Weierstrass(x, y, a=0.5, b=3, k_max=20)

print("Mejor solución encontrada:")
print(f"X: {mejor_solucion[0]}")
print(f"Y: {mejor_solucion[1]}")
print(f"Z: {mejor_costo}")

print("Valor optimo: ", z.min())
surface = ax.plot_surface(x, y, z, cmap='viridis')
ax.scatter([mejor_solucion[0]], [mejor_solucion[1]], [mejor_costo], color='red', marker='o', s=100, label='Mejor Solución')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Gráfica de la función Weierstrass minimizada por ACO')

plt.legend()
plt.show()