
import numpy as np
import matplotlib.pyplot as plt

vMin = -0.5
vMax = 0.5
class Anticuerpo:
    global vMin, vMax
    def __init__(self, nVar):
        self.posicion = np.random.uniform(vMin, vMax, nVar)
        self.costo = funcion_objetivo(*self.posicion)

# Función Weierstrass como objetivo
def funcion_objetivo(x, y):
    a=0.5
    b=3
    k_max=20
    result = 0
    for k in range(k_max):
        result += (a**k * np.cos(2 * np.pi * b**k * (x + 0.5)) - a**k*np.cos(np.pi*b**k)) + (a**k * np.cos(2 * np.pi * b**k * (y + 0.5)) - a**k*np.cos(np.pi*b**k))
    return result


# Parámetros del algoritmo clonal
nVar = 2                    # Numero de variables de decisión
nAnticuerpos = 50               # Tamaño de la población de células
num_iteraciones = 100       # Numero de iteraciones
beta = 0.5                  # Factor de clonación
pMutacion = 0.1             # Probabilidad de mutación
num_puntos = 1000
rango_x = np.linspace(vMin, vMax, num_puntos)
rango_y = np.linspace(vMin, vMax, num_puntos)
x_grid, y_grid = np.meshgrid(rango_x, rango_y)
z = funcion_objetivo(x_grid, y_grid)

# Inicialización de la población de células
poblacion = [Anticuerpo(nVar) for _ in range(nAnticuerpos)]

mejor_global = Anticuerpo(nVar)
mejor_global.costo = float('inf')
mejores_costos = []
mejor_anticuerpo = min(poblacion, key=lambda anticuerpo: anticuerpo.costo)
# Bucle principal del algoritmo clonal
for iteracion in range(num_iteraciones):
    # Clonación y mutación

    anticuerpos_clonadas = []
    for anticuerpo in poblacion:
        num_clones = int(beta * nAnticuerpos)
        anticuerpos_clonadas.extend([Anticuerpo(nVar) for _ in range(num_clones)])
    
    # Selección
    poblacion.extend(anticuerpos_clonadas)
    poblacion = sorted(poblacion, key=lambda anticuerpo: anticuerpo.costo)[:nAnticuerpos]
    
    # Mutación
    for anticuerpo in poblacion:
        if np.random.rand() < pMutacion:
            anticuerpo.posicion = anticuerpo.posicion * (1 + np.random.uniform(vMin, vMax, nVar))
            anticuerpo.costo = funcion_objetivo(*anticuerpo.posicion)
        if (anticuerpo.costo < mejor_global.costo):
            mejor_global.posicion = anticuerpo.posicion.copy()
            mejor_global.costo = funcion_objetivo(*mejor_global.posicion)
    mejores_costos.append(mejor_global.costo)

# Mostrar resultados
print("Mejor solución encontrada:")
print(f"Posición: {mejor_global.posicion}")
print(f"Costo: {mejor_global.costo}")

# Graficar la evolución del mejor costo
plt.plot(range(num_iteraciones), mejores_costos)
plt.xlabel('Iteración')
plt.ylabel('Mejor Costo')
plt.title('Evolución del Mejor Costo en Cada Iteración')
plt.grid(True)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(x_grid, y_grid, z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Gráfica de la función Weierstrass minimizada por SIA')
ax.scatter(mejor_global.posicion[0], mejor_global.posicion[1], mejor_global.costo , color='red', marker='o', s=100, label='Mejor Solución')

plt.legend()
plt.show()