
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
num_puntos=150              #Numero de puntos del espacio discretizado
w_max = 0.9                 #Coeficientes de incercia
w_min = 0.2
c1, c2 = 0.5,0.5            #Coeficientes de atracción personal y social
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
        'actual':{'x': np.random.choice(rango_x), 'y': np.random.choice(rango_y), 'z': float('inf')},
        'pbest':{'x': None, 'y': None, 'z': float('inf')},
        'velocidad': {'v_x': 0.0, 'v_y': 0.0}
    }
    particulas.append(particula)

gbest = {'x': None, 'y': None, 'z': float('inf')}
mejores_costos = []
# Bucle de optimización por enjambre de particulas
for iteracion in range(num_iteraciones):
    
    for i in particulas:

        i['actual']['z'] = Weierstrass(i['actual']['x'], i['actual']['y'], a=0.5, b=3, k_max=20)

        if( i['actual']['z'] < i['pbest']['z'] ):
            i['pbest'] = i['actual'].copy()
            #i['pbest']['x'], i['pbest']['y'], i['pbest']['z'] = i['actual']['x'], i['actual']['y'], i['actual']['z']

        if( i['pbest']['z'] < gbest['z'] ):
            gbest = i['actual'].copy()
            #gbest['x'], gbest['y'], gbest['z'] = i['actual']['x'], i['actual']['y'], i['actual']['z']
        
        w = w_max - (w_max - w_min) * iteracion / num_iteraciones
        r1 = np.random.rand()
        r2 = np.random.rand()

        # Cálculo de la nueva velocidad
        i['velocidad']['v_x'] = w * i['velocidad']['v_x'] + c1 * r1 * (i['pbest']['x'] - i['actual']['x']) + c2 * r2 * (gbest['x'] - i['actual']['x'])
        i['velocidad']['v_y'] = w * i['velocidad']['v_y'] + c1 * r1 * (i['pbest']['y'] - i['actual']['y']) + c2 * r2 * (gbest['y'] - i['actual']['y'])

        # Actualización de la posición basada en la nueva velocidad
        i['actual']['x'] += i['velocidad']['v_x']
        i['actual']['y'] += i['velocidad']['v_y']

    mejores_costos.append(gbest['z'])

# Grafica de la función Weierstrass y la evolución de la mejor solución encontrada
plt.plot(range(num_iteraciones), mejores_costos)
plt.xlabel('Iteración')
plt.ylabel('Mejor Costo')
plt.title('Evolución del Mejor Costo en Cada Iteración')
plt.grid(True)


print("Mejor solución encontrada:")
print(f"X: {gbest['x']}")
print(f"Y: {gbest['y']}")
print(f"Z: {gbest['z']}")
print("Valor optimo: ", z.min())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(x_grid, y_grid, z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Gráfica de la función Weierstrass minimizada por PSO')
ax.scatter(gbest['x'], gbest['y'], gbest['z'] , color='red', marker='o', s=100, label='Mejor Solución')

plt.legend()
plt.show()