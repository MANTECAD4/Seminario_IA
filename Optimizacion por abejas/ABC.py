import numpy as np
import matplotlib.pyplot as plt

def FuncionObjetivo(posicion):
    a=0.5
    b=3
    k_max=20
    result = 0
    for k in range(k_max):
        result += (a**k * np.cos(2 * np.pi * b**k * (posicion[0,0] + 0.5)) - a**k*np.cos(np.pi*b**k)) + (a**k * np.cos(2 * np.pi * b**k * (posicion[0,1] + 0.5)) - a**k*np.cos(np.pi*b**k))
    return result

def ABC(FuncionObjetivo, NumVar, LimiteInferior, LimiteSuperior, Generaciones, TamPoblacion):
    # Tamaño de la matriz de variables de decisión
    MatrizVar = [1, NumVar]
    # Número de abejas espectadoras
    AbejaAFK = TamPoblacion
    # Parámetro de límite de abandono
    L = round(0.6 * NumVar * TamPoblacion)
    # Límite superior del coeficiente de aceleración
    a = 1

    # Inicialización
    # Estructura de abeja
    class Abeja:
        def __init__(self):
            self.Posicion = []
            self.Costo = []

    # Inicializar población
    Poblacion = [Abeja() for _ in range(TamPoblacion)]
    # Inicializar la mejor solución encontrada
    MejorSolucion = Abeja()
    MejorSolucion.Costo = float('inf')
    # Crear población inicial
    for i in range(TamPoblacion):
        Poblacion[i].Posicion = np.random.uniform(LimiteInferior, LimiteSuperior, MatrizVar)
        Poblacion[i].Costo = FuncionObjetivo(Poblacion[i].Posicion)
        if Poblacion[i].Costo <= MejorSolucion.Costo:
            MejorSolucion = Poblacion[i]

    # Contador de abandono
    Contador = np.zeros(TamPoblacion)
    # Matriz para mantener los mejores valores de costo
    MejoresCostos = np.zeros(Generaciones)

    # Bucle principal
    for Iteracion in range(Generaciones):
        # Abejas reclutadas
        for i in range(TamPoblacion):
            # Se elige k al azar, no igual a i
            K = list(range(0, i)) + list(range(i + 1, TamPoblacion))
            k = np.random.choice(K)
            # Coeficiente de aceleración
            phi = a * np.random.uniform(-1, 1, MatrizVar)
            # Nueva posición de la abeja
            NuevaAbeja = Abeja()
            NuevaAbeja.Posicion = Poblacion[i].Posicion + phi * (Poblacion[i].Posicion - Poblacion[k].Posicion)
            # Aplicar límites
            NuevaAbeja.Posicion = np.clip(NuevaAbeja.Posicion, LimiteInferior, LimiteSuperior)

            # Evaluación
            NuevaAbeja.Costo = FuncionObjetivo(NuevaAbeja.Posicion)
            # Comparación
            if NuevaAbeja.Costo <= Poblacion[i].Costo:
                Poblacion[i] = NuevaAbeja
            else:
                Contador[i] += 1

        # Calcular valores de aptitud y probabilidades de selección
        F = np.zeros(TamPoblacion)
        CostoPromedio = np.mean([p.Costo for p in Poblacion])
        # Convertir Costo a Aptitud
        for i in range(TamPoblacion):
            F[i] = np.exp(-Poblacion[i].Costo / CostoPromedio)

        P = F / np.sum(F)

        # Abejas espectadoras
        for m in range(AbejaAFK):
            # Seleccionar sitio de origen
            i = np.random.choice(range(TamPoblacion), p=P)
            # Se elige k al azar, no igual a i
            K = list(range(0, i)) + list(range(i + 1, TamPoblacion))
            k = np.random.choice(K)
            # Definir el coeficiente de aceleración
            phi = a * np.random.uniform(-1, 1, MatrizVar)
            # Nueva posición de abeja
            NuevaAbeja = Abeja()
            NuevaAbeja.Posicion = Poblacion[i].Posicion + phi * (Poblacion[i].Posicion - Poblacion[k].Posicion)
            # Aplicar límites
            NuevaAbeja.Posicion = np.clip(NuevaAbeja.Posicion, LimiteInferior, LimiteSuperior)

            # Evaluación
            NuevaAbeja.Costo = FuncionObjetivo(NuevaAbeja.Posicion)
            # Comparación
            if NuevaAbeja.Costo <= Poblacion[i].Costo:
                Poblacion[i] = NuevaAbeja
            else:
                Contador[i] += 1

        # Abejas exploradoras
        for i in range(TamPoblacion):
            if Contador[i] >= L:
                Poblacion[i].Posicion = np.random.uniform(LimiteInferior, LimiteSuperior, MatrizVar)
                Poblacion[i].Costo = FuncionObjetivo(Poblacion[i].Posicion)
                Contador[i] = 0

        # Actualizar la mejor solución encontrada
        for i in range(TamPoblacion):
            if Poblacion[i].Costo <= MejorSolucion.Costo:
                MejorSolucion.Posicion = Poblacion[i].Posicion.copy()
                MejorSolucion.Costo = FuncionObjetivo(MejorSolucion.Posicion)

        # Almacenar el mejor costo encontrado
        MejoresCostos[Iteracion] = MejorSolucion.Costo
    
    Resultado = {'gBest': MejorSolucion, 'MejoresCostos': MejoresCostos}
    return Resultado

# Parámetros
NumVar = 2
LimiteInferior = -0.5
LimiteSuperior = 0.5
Generaciones = 50
TamPoblacion = 50

# Ejecución algoritmo
resultado = ABC(FuncionObjetivo, NumVar, LimiteInferior, LimiteSuperior, Generaciones, TamPoblacion)
print("Mejor solucion encontrada: Posicion -> ", resultado['gBest'].Posicion,", Costo ->",resultado['gBest'].Costo)

# Grafica de la función funcion_objetivo y la evolución de la mejor solución encontrada
plt.plot(range(Generaciones), resultado['MejoresCostos'])
plt.xlabel('Iteración')
plt.ylabel('Mejor Costo')
plt.title('Evolución del Mejor Costo en Cada Iteración')
plt.grid(True)

plt.show()

