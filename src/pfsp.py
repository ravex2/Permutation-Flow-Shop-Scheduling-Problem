import numpy as np
import os


# individuos, y poblacion en la asociacion del algiritmo genetico
def fitness_pfsp(permutacion, tiempos):
    cantidad_maquinas = tiempos.shape[1]
    finalizacion_maquinas = np.zeros(cantidad_maquinas, dtype=np.int64)

    for trabajo in permutacion:
        finalizacion_trabajo = np.int64(0)
        for maquina in range(cantidad_maquinas):
            inicio = max(finalizacion_trabajo, finalizacion_maquinas[maquina])
            finalizacion_trabajo = inicio + tiempos[trabajo, maquina]
            finalizacion_maquinas[maquina] = finalizacion_trabajo

    return int(finalizacion_maquinas[-1])

# probar
def leer_instancia(ruta):
    with open(ruta, encoding="utf-8") as archivo:
        lineas = [linea.split("#", 1)[0].strip() for linea in archivo]
    lineas = [linea for linea in lineas if linea]

    '''
    if not lineas:
        raise ValueError("La instancia esta vacia.")
    n, m = map(int, lineas[0].split())
    if n < 1 or m < 1:
        raise ValueError("Se requiere al menos un trabajo y una maquina.")
    filas = [list(map(int, linea.split())) for linea in lineas[1:]]
    if len(filas) != m or any(len(fila) != n for fila in filas):
        raise ValueError(f"Se esperaban {m} filas de {n} tiempos.")
    tiempos = np.array(filas, dtype=np.int64).T.copy()
    if np.any(tiempos < 0):
        raise ValueError("Los tiempos no pueden ser negativos.")
    '''
    filas = [list(map(int, linea.split())) for linea in lineas[1:]]
    tiempos = np.array(filas, dtype=np.int64).T.copy()
    return tiempos
