
import numpy as np
import matplotlib.pyplot as plt
class Individuo :
    def __init__(self):
        self.posicion = []
        self.costo = float('inf')
# Función Weierstrass como objetivo
def funcion_objetivo(x, y, a, b, k_max):
    result = 0
    for k in range(k_max):
        result += (a**k * np.cos(2 * np.pi * b**k * (x + 0.5)) - a**k*np.cos(np.pi*b**k)) + (a**k * np.cos(2 * np.pi * b**k * (y + 0.5)) - a**k*np.cos(np.pi*b**k))
    return result

# Parámetros de Evolución diferencial
nVar = 2                    # Numero de variables de decision
nPop = 30                   # Tamaño de la poblacion
num_iteraciones = 100       # Numero de iteraciones
vMax = 0.5                  # Limite superior
vMin = -0.5                 # Limite inferior
num_puntos = 1000           # Numero de puntos del espacio discretizado
sf = 0.8                    # factor escalar
pCR = 0.2                   # Probabilidad de cruza

rango_x = np.linspace(vMin, vMax, num_puntos)
rango_y = np.linspace(vMin, vMax, num_puntos)
x_grid, y_grid = np.meshgrid(rango_x, rango_y)
z = funcion_objetivo(x_grid, y_grid, a=0.5, b=3, k_max=20)

# Iniciliaizaciones
poblacion = []
for i in range(nPop):
    ind = Individuo()
    ind.posicion = np.random.uniform(vMin, vMax, 2)
    ind.costo = funcion_objetivo(ind.posicion[0], ind.posicion[1], a=0.5, b=3, k_max=20)
    poblacion.append(ind)

mejor_solucion = Individuo()
mejor_solucion.costo = float('inf')
mejores_costos = []

# Bucle principal DE
for iteracion in range(num_iteraciones):

    for i in range (nPop):

        # Seleccion aleatoria de 3 individuos de la poblacion
        a = np.random.choice(poblacion)
        b = np.random.choice(poblacion)
        c = np.random.choice(poblacion)

        # Generacion del vector de prueba
        v = a.posicion + sf * (b.posicion - c.posicion)
        v = np.clip(v, vMin, vMax)

        u = np.zeros_like(poblacion[i].posicion)
        j0 = np.random.randint(0,2)
        for j in range (len(u)):
            if ( j0 == j or np.random.rand() <= pCR):
                u[j] = v[j]
            else:
                u[j] = poblacion[i].posicion[j]
        # Evaluacion del nuevo individuo
        nueva_solucion = Individuo()
        nueva_solucion.posicion = u.copy()
        nueva_solucion.costo = funcion_objetivo(nueva_solucion.posicion[0], nueva_solucion.posicion[1], a=0.5, b=3, k_max=20)

        # Comparacion entre el individuo actual y el nuevo
        if nueva_solucion.costo < poblacion[i].costo:
            poblacion[i] = nueva_solucion
            # Comparacion entre el individuo actual y el mejor encontrado
            if nueva_solucion.costo < mejor_solucion.costo:
                mejor_solucion = nueva_solucion
    mejores_costos.append(mejor_solucion.costo)

# Redondeo de las coordenadas de la mejor solución
redondeo_x = min(rango_x, key=lambda x: abs(x - mejor_solucion.posicion[0]))
redondeo_y = min(rango_y, key=lambda y: abs(y - mejor_solucion.posicion[1]))

# Establecer la posición de la partícula en las coordenadas más cercanas
mejor_solucion.posicion[0] = redondeo_x
mejor_solucion.posicion[1] = redondeo_y
mejor_solucion.costo = funcion_objetivo(mejor_solucion.posicion[0], mejor_solucion.posicion[1], a=0.5, b=3, k_max=20)

# Grafica de la función funcion_objetivo y la evolución de la mejor solución encontrada
plt.plot(range(num_iteraciones), mejores_costos)
plt.xlabel('Iteración')
plt.ylabel('Mejor Costo')
plt.title('Evolución del Mejor Costo en Cada Iteración')
plt.grid(True)

# Se muestran los resultados del algoritmo
print("Mejor solución encontrada:")
print(f"X: {mejor_solucion.posicion[0]}")
print(f"Y: {mejor_solucion.posicion[1]}")
print(f"Z: {mejor_solucion.costo}")
print("Valor óptimo: ", z.min())

# Grafica de la función Weierstrass en 3D con gbest marcada 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(x_grid, y_grid, z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Gráfica de la función funcion_objetivo minimizada por DE')
ax.scatter(mejor_solucion.posicion[0], mejor_solucion.posicion[1], mejor_solucion.costo , color='red', marker='o', s=100, label='Mejor Solución')

plt.legend()
plt.show()