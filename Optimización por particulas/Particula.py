
import numpy as np
import matplotlib.pyplot as plt

# Función Weierstrass como objetivo
def funcion_objetivo(x, y, a, b, k_max):
    result = 0
    for k in range(k_max):
        result += (a**k * np.cos(2 * np.pi * b**k * (x + 0.5)) - a**k*np.cos(np.pi*b**k)) + (a**k * np.cos(2 * np.pi * b**k * (y + 0.5)) - a**k*np.cos(np.pi*b**k))
    return result

# Parámetros de PSO
num_particulas = 30         # Numero de particulas
num_iteraciones = 100       # Numero de iteraciones
lb = 0.5                    # Limite superior
ub = -0.5                   # Limite inferior
num_puntos = 1000            # Numero de puntos del espacio discretizado
w_max = 0.9                 # Coeficientes de incercia
w_min = 0.2
c1, c2 = 2,2                # Coeficientes de atracción personal y social
vMax = (lb - ub) * 0.2      # Velocidad máxima
vMin  = -vMax               # Velocidad mínima

rango_x = np.linspace(ub, lb, num_puntos)
rango_y = np.linspace(ub, lb, num_puntos)
x_grid, y_grid = np.meshgrid(rango_x, rango_y)
z = funcion_objetivo(x_grid, y_grid, a=0.5, b=3, k_max=20)

# Iniciliaización de las partículas
particulas = []
for i in range(num_particulas):
    particula = {
        'actual':{'x': np.random.choice(rango_x), 'y': np.random.choice(rango_y), 'z': float('inf')},
        'pbest':{'x': np.random.choice(rango_x), 'y': np.random.choice(rango_y), 'z': float('inf')},
        'velocidad': {'v_x': 0.0, 'v_y': 0.0}
    }
    particulas.append(particula)

gbest = {'x': np.random.choice(rango_x), 'y': np.random.choice(rango_y), 'z': float('inf')}
mejores_costos = []

# Bucle de optimización por enjambre de particulas
for iteracion in range(num_iteraciones):
    
    for particula in particulas:

        # Cálculo de la nueva velocidad
        w = w_max - (w_max - w_min) * iteracion / num_iteraciones
        r1 = np.random.rand()
        r2 = np.random.rand()
        
        particula['velocidad']['v_x'] = w * particula['velocidad']['v_x'] + c1 * r1 * (particula['pbest']['x'] - particula['actual']['x']) + c2 * r2 * (gbest['x'] - particula['actual']['x'])
        particula['velocidad']['v_y'] = w * particula['velocidad']['v_y'] + c1 * r1 * (particula['pbest']['y'] - particula['actual']['y']) + c2 * r2 * (gbest['y'] - particula['actual']['y'])

        # Aplicar límites a la velocidad en las dimensiones x e y
        particula['velocidad']['v_x'] = max(min(particula['velocidad']['v_x'], vMax), vMin)
        particula['velocidad']['v_y'] = max(min(particula['velocidad']['v_y'], vMax), vMin)
        
        # Actualización de la posición basada en la nueva velocidad
        particula['actual']['x'] += particula['velocidad']['v_x']
        particula['actual']['y'] += particula['velocidad']['v_y']
        
        # Aplicar límites de posición en las dimensiones x e y
        particula['actual']['x'] = max(min(particula['actual']['x'], lb), ub)
        particula['actual']['y'] = max(min(particula['actual']['y'], lb), ub)

        # Evaluación de la particula
        particula['actual']['z'] = funcion_objetivo(particula['actual']['x'], particula['actual']['y'], a=0.5, b=3, k_max=20)

        # Actualización de mejor experiencia personal
        if( particula['actual']['z'] < particula['pbest']['z'] ):
            particula['pbest'] = particula['actual'].copy()

        #Actualización de mejor experiencia global
        if( particula['pbest']['z'] < gbest['z'] ):
            gbest = particula['actual'].copy()
    
    mejores_costos.append(gbest['z'])

# Redondeo de las coordenadas de la mejor solución
redondeo_x = min(rango_x, key=lambda x: abs(x - gbest['x']))
redondeo_y = min(rango_y, key=lambda y: abs(y - gbest['y']))

# Establecer la posición de la partícula en las coordenadas más cercanas
gbest['x'] = redondeo_x
gbest['y'] = redondeo_y
gbest['z'] = funcion_objetivo(gbest['x'], gbest['y'], a=0.5, b=3, k_max=20)

# Grafica de la función funcion_objetivo y la evolución de la mejor solución encontrada
plt.plot(range(num_iteraciones), mejores_costos)
plt.xlabel('Iteración')
plt.ylabel('Mejor Costo')
plt.title('Evolución del Mejor Costo en Cada Iteración')
plt.grid(True)

# Se muestran los resultados del algoritmo
print("Mejor solución encontrada:")
print(f"X: {gbest['x']}")
print(f"Y: {gbest['y']}")
print(f"Z: {gbest['z']}")
print("Valor óptimo: ", z.min())

# Grafica de la función Weierstrass en 3D con gbest marcada 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(x_grid, y_grid, z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Gráfica de la función funcion_objetivo minimizada por PSO')
ax.scatter(gbest['x'], gbest['y'], gbest['z'] , color='red', marker='o', s=100, label='Mejor Solución')

plt.legend()
plt.show()