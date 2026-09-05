import numpy as np
import random

from src.pfsp import fitness_pfsp


''' 
Implementacion del algoritmo genetico para el problema PFSP
'''

def crear_poblacion(cantidad_trabajos, tam_pob, generador):
    poblacion = []
    plantilla = np.arange(cantidad_trabajos, dtype=np.int64)
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
    """Cruce OX: genera un hijo que conserva una permutacion valida."""
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



def ejecutar_algoritmo_genetico(
    tiempos,
    tam_pob,
    por_cru,
    por_mul,
    num_ite,
    semilla,
):
    generador = random.Random(semilla)
    poblacion = crear_poblacion(len(tiempos), tam_pob, generador)
    mejor = min(poblacion, key=lambda individuo: fitness_pfsp(individuo, tiempos))

    for _ in range(num_ite):
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
        if fitness_pfsp(candidato, tiempos) < fitness_pfsp(mejor, tiempos):
            mejor = candidato.copy()

    return mejor, fitness_pfsp(mejor, tiempos)
