import sys
import os 
import argparse
import random
import numpy as np
from src.pfsp import fitness_pfsp


def resolve_path_windows(path, directory):
    ruta = os.path.expanduser(path)
    if os.path.isabs(ruta):
        return os.path.normpath(path)

    script_directory = os.path.dirname(os.path.abspath(__file__))
    if sys.platform == "win32":
        relative_path = directory + "\\" + ruta
    else:
        relative_path = directory + "/" + ruta

    return os.path.normpath(os.path.join(script_directory, relative_path))


# listar los nombre de archivos de un directorio
def lsDirectory(directory):
    files = []
    for filename in os.listdir(directory):
        ruta_archivo = os.path.join(directory, filename)
        if os.path.isfile(ruta_archivo):
            files.append(filename)
    return files


# analizar si esto esta bien
def leer_instancia(ruta):
    with open(ruta, "r", encoding="utf-8") as archivo:
        tokens = []
        for linea in archivo:
            linea = linea.split("#", 1)[0]
            tokens.extend(linea.split())

    if len(tokens) < 2:
        raise ValueError("El archivo no contiene la cantidad de trabajos y maquinas.")

    try:
        cantidad_trabajos = int(tokens[0])
        cantidad_maquinas = int(tokens[1])
        tiempos = [int(token) for token in tokens[2:]]
    except ValueError as error:
        raise ValueError("El archivo contiene valores que no son enteros.") from error

    cantidad_tiempos = cantidad_trabajos * cantidad_maquinas
    if len(tiempos) != cantidad_tiempos:
        raise ValueError(
            f"Se esperaban {cantidad_tiempos} tiempos y se encontraron {len(tiempos)}."
        )

    matriz_tiempos = np.array(
        tiempos,
        dtype=np.int64,
    ).reshape(cantidad_trabajos, cantidad_maquinas)
    return cantidad_trabajos, cantidad_maquinas, matriz_tiempos













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


def createParser():
    parser = argparse.ArgumentParser(
        description="Ejecuta el algoritmo genetico con los parametros indicados.",
        epilog=(
            "Ejemplo: python3 ag.py 42 100 0.8 0.1 500 entrada.txt salida.txt"
        ),
    )
    parser.add_argument("semilla", type=int, help="Semilla del generador aleatorio.")
    parser.add_argument(
        "tam_pob",
        type=int,
        help="Tamano de la poblacion inicial.",
    )
    parser.add_argument(
        "por_cru",
        type=float,
        help="Probabilidad de cruce, entre 0 y 1.",
    )
    parser.add_argument(
        "por_mul",
        type=float,
        help="Probabilidad de mutacion, entre 0 y 1.",
    )
    parser.add_argument(
        "num_ite",
        type=int,
        help="Cantidad de iteraciones del algoritmo.",
    )
    parser.add_argument(
        "entrada",
        help="Nombre del archivo de entrada dentro de data/.",
    )
    parser.add_argument(
        "salida",
        help="Nombre del archivo de salida dentro de result/.",
    )
    return parser


def main():
    files = lsDirectory("data")
    print("Archivos de entrada disponibles en data/: ", files)
    
    if not files:
        createParser().error("No se encontraron archivos de entrada en el directorio data/.")
    
    args = createParser().parse_args()

    # restricciones:
    if args.tam_pob < 2:
        createParser().error("tam_pob debe ser mayor o igual que 2.")
    if not 0 <= args.por_cru <= 1 or not 0 <= args.por_mul <= 1:
        createParser().error("por_cru y por_mul deben estar entre 0 y 1.")
    if args.num_ite < 0:
        createParser().error("num_ite no puede ser negativo.")

    semilla = args.semilla
    tam_pob = args.tam_pob
    por_cru = args.por_cru
    por_mul = args.por_mul
    num_ite = args.num_ite
    entrada = resolve_path_windows(args.entrada, "data")
    salida = resolve_path_windows(args.salida, "result")

    if not os.path.isfile(entrada):
        createParser().error(
            f"No se encontro el archivo de entrada: {entrada}"
        )

    os.makedirs(os.path.dirname(salida), exist_ok=True)


    with open(entrada, "r", encoding="utf-8") as f:
        data_info = f.readline().strip().split()
        cantidad_trabajos = int(data_info[0])


    '''
    try:
        cantidad_trabajos, cantidad_maquinas, tiempos = leer_instancia(entrada)
    except ValueError as error:
        createParser().error(str(error))

    mejor, makespan = ejecutar_algoritmo_genetico(
        tiempos,
        args.tam_pob,
        args.por_cru,
        args.por_mul,
        args.num_ite,
        args.semilla,
    )
    saveData(salida, cantidad_trabajos, cantidad_maquinas, makespan, mejor)
    '''


def saveData(salida, cantidad_trabajos, cantidad_maquinas, makespan, mejor):
    with open(salida, "w", encoding="utf-8") as archivo:
        archivo.write(f"Trabajos: {cantidad_trabajos}\n")
        archivo.write(f"Maquinas: {cantidad_maquinas}\n")
        archivo.write(f"Makespan: {makespan}\n")
        archivo.write(
            "Permutacion: " + " ".join(str(trabajo + 1) for trabajo in mejor) + "\n"
        )

    print(f"Makespan: {makespan}")
    print(f"Resultado guardado en: {salida}")



if __name__ == "__main__":
    main()