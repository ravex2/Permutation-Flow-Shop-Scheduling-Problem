
## Implementacion de algoritmos geneticos

El programa resuelve instancias de **Permutation Flow Shop Scheduling Problem (PFSP)**
con un algoritmo genetico. La instancia debe contener comentarios iniciados con `#`,
una primera linea con `cantidad_trabajos cantidad_maquinas` y luego la matriz de
tiempos de procesamiento.

```sh
python3 ag.py SEMILLA TAM_POB PROB_CRUCE PROB_MUTACION ITERACIONES ENTRADA SALIDA
```

Ejemplo usando `data/ta001.txt`:

```sh
python3 ag.py 42 30 0.8 0.1 100 ta001.txt ta001_resultado.txt
```

La entrada se busca en `data/` y el resultado se escribe en `result/`, salvo que
se indique una ruta absoluta. El resultado contiene el makespan y la permutacion
de trabajos encontrada.



Download Datase 2:
```sh
# download repo 1
curl -fL --retry 3 \ -o flowshop_instances.zip \ 'https://ndownloader.figshare.com/files/48152884'

# download repo 2
mkdir -p data && curl -L 'https://github.com/arnaud-m/pisco/archive/refs/heads/master.tar.gz' |
tar -xz -C data --strip-components=8 \ 'pisco-master/pisco-shop/src/main/benchmarks/instances/flow-shop/taillard'

# mac os
curl -fL 'https://api.figshare.com/v2/file/download/48152884' -o flowshop.zip && mkdir -p flowshop && ditto -x -k flowshop.zip flowshop && tar -czf flowshop.tar.gz flowshop

```

```py
# run example:
python3 ag.py 42 30 0.8 0.1 100 ta001.txt ta001_numpy.txt && printf '\n--- resultado ---\n' && cat result/ta001_numpy.txt
```