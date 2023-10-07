import numpy as np
import matplotlib.pyplot as plt

# Función objetivo que queremos optimizar (puedes cambiarla)
def objective_function(x):
    return x[0]**2 + x[1]**2 + x[2]**2

# Parámetros del PSO
num_particles = 30
num_dimensions = 3
max_iterations = 100
c1 = 2.0  # Coeficiente cognitivo
c2 = 2.0  # Coeficiente social
w = 0.7   # Inercia

# Inicialización de las partículas y velocidades
particles = np.random.rand(num_particles, num_dimensions)
velocities = np.random.rand(num_particles, num_dimensions)
personal_best_positions = particles.copy()
personal_best_values = np.array([objective_function(p) for p in particles])
global_best_index = np.argmin(personal_best_values)
global_best_position = personal_best_positions[global_best_index]
global_best_value = personal_best_values[global_best_index]

# Arrays para almacenar la evolución de la mejor solución
best_values = []

# Bucle principal del PSO
for iteration in range(max_iterations):
    for i in range(num_particles):
        # Actualizar velocidad
        r1, r2 = np.random.rand(2)
        cognitive_component = c1 * r1 * (personal_best_positions[i] - particles[i])
        social_component = c2 * r2 * (global_best_position - particles[i])
        velocities[i] = w * velocities[i] + cognitive_component + social_component

        # Actualizar posición
        particles[i] += velocities[i]

        # Evaluar la nueva posición
        new_value = objective_function(particles[i])

        # Actualizar la mejor posición personal si es necesario
        if new_value < personal_best_values[i]:
            personal_best_values[i] = new_value
            personal_best_positions[i] = particles[i]

            # Actualizar la mejor posición global si es necesario
            if new_value < global_best_value:
                global_best_value = new_value
                global_best_position = particles[i]

    best_values.append(global_best_value)

# Visualizar la convergencia
plt.plot(best_values)
plt.xlabel('Iteración')
plt.ylabel('Mejor Valor')
plt.title('Convergencia del PSO')
plt.show()

print("Mejor valor encontrado:", global_best_value)
print("Mejor posición encontrada:", global_best_position)
