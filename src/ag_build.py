import numpy as np
import random

from src.pfsp import fitness_pfsp

''' 
Implementacion del algoritmo genetico para el problema PFSP
'''

def crear_poblacion(cantidad_trabajos, tam_pob, generador):
    poblacion = []
    plantilla = np.arange(cantidad_trabajos, dtype=np.int64)
    print(plantilla)
    for _ in range(tam_pob):
        individuo = plantilla.copy()
        individuo = np.array(generador.sample(individuo.tolist(), cantidad_trabajos))
        poblacion.append(individuo)
    return poblacion


def seleccion_torneo(poblacion, tiempos, generador, tam_torneo=3):
    participantes = generador.sample(
        poblacion,
        min(tam_torneo, len(poblacion)),
    )
    return min(participantes, key=lambda individuo: fitness_pfsp(individuo, tiempos))


def cruce_ordenado(primer_padre, segundo_padre, generador):
    inicio, fin = sorted(generador.sample(range(len(primer_padre)), 2))
    hijo = np.full(len(primer_padre), -1, dtype=np.int64)
    hijo[inicio:fin] = primer_padre[inicio:fin]
    genes_restantes = [
        trabajo for trabajo in segundo_padre if trabajo not in hijo[inicio:fin]
    ]

    posicion = 0
    for indice in range(len(hijo)):
        if hijo[indice] == -1:
            hijo[indice] = genes_restantes[posicion]
            posicion += 1
    return hijo


def mutar_intercambio(individuo, por_mul, generador):
    if generador.random() < por_mul:
        primera, segunda = generador.sample(range(len(individuo)), 2)
        individuo[primera], individuo[segunda] = (
            individuo[segunda],
            individuo[primera],
        )


# remplazo para la nueva generacion
# busquedas local

def ejecutar_algoritmo_genetico(
    tiempos,
    tam_pob,
    por_cru,
    por_mul,
    num_ite,
    semilla,
    frecuencia_local=0,
    max_evaluaciones_local=100,
):
    generador = random.Random(semilla)
    poblacion = crear_poblacion(len(tiempos), tam_pob, generador)
    mejor = min(poblacion, key=lambda individuo: fitness_pfsp(individuo, tiempos))

    for generacion in range(num_ite):
        nueva_poblacion = [mejor.copy()]
        while len(nueva_poblacion) < tam_pob:
            primer_padre = seleccion_torneo(poblacion, tiempos, generador)
            segundo_padre = seleccion_torneo(poblacion, tiempos, generador)

            if generador.random() < por_cru:
                hijo = cruce_ordenado(primer_padre, segundo_padre, generador)
            else:
                hijo = primer_padre.copy()

            mutar_intercambio(hijo, por_mul, generador)
            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion
        candidato = min(
            poblacion,
            key=lambda individuo: fitness_pfsp(individuo, tiempos),
        )
        if frecuencia_local and (generacion + 1) % frecuencia_local == 0:
            mejorado = busqueda_local_insercion(
                candidato, tiempos, generador, max_evaluaciones_local
            )
            indice = None
            for i, individuo in enumerate(poblacion):
                if individuo is candidato:
                    indice = i
                    break 


            poblacion[indice] = mejorado
            candidato = mejorado
        if fitness_pfsp(candidato, tiempos) < fitness_pfsp(mejor, tiempos):
            mejor = candidato.copy()

    return mejor, fitness_pfsp(mejor, tiempos)




def busqueda_local_insercion(individuo, tiempos, generador, max_evaluaciones=100):
    """Primera mejora por insercion, limitada por evaluaciones de vecinos."""
    mejor = individuo.copy()
    costo = fitness_pfsp(mejor, tiempos)
    evaluaciones = 0
    while evaluaciones < max_evaluaciones:
        mejora = False
        origenes = list(range(len(mejor)))
        generador.shuffle(origenes)
        for origen in origenes:
            destinos = list(range(len(mejor)))
            generador.shuffle(destinos)
            for destino in destinos:
                if origen == destino:
                    continue
                vecino = mejor.tolist()
                vecino.insert(destino, vecino.pop(origen))
                vecino = np.array(vecino, dtype=np.int64)
                valor = fitness_pfsp(vecino, tiempos)
                evaluaciones += 1
                if valor < costo:
                    mejor, costo, mejora = vecino, valor, True
                    break
                if evaluaciones >= max_evaluaciones:
                    return mejor
            if mejora:
                break
        if not mejora:
            break
    return mejor


def ejecutar_algoritmo_memetico(
    tiempos, tam_pob, por_cru, por_mul, num_ite, semilla,
    generacion=1, max_gen_evaluada=100,
):
    if generacion < 1:
        raise ValueError("La frecuencia local debe ser al menos 1.")
    return ejecutar_algoritmo_genetico(
        tiempos, tam_pob, por_cru, por_mul, num_ite, semilla,
        generacion, max_gen_evaluada,
    )