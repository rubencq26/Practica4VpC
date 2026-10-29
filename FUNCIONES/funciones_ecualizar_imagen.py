import numpy as np


def ecualizar_imagen_v1(I):
    """
    Ecualización uniforme de una imagen de intensidad (uint8).

    V1: recorrido a nivel de píxel. Se calcula el histograma recorriendo
    la imagen píxel a píxel y, para cada píxel, se calcula su función de
    transformación q = 255 * CDF(p) para generar la imagen de salida.
    """
    filas, columnas = I.shape
    N = filas * columnas
    niveles = 256

    # Histograma: recorrido píxel a píxel
    h = np.zeros(niveles, dtype=np.int64)
    for i in range(filas):
        for j in range(columnas):
            h[I[i, j]] += 1

    # Imagen de salida: la transformación se calcula para cada píxel
    Ieq = np.zeros_like(I, dtype=np.uint8)
    for i in range(filas):
        for j in range(columnas):
            p = I[i, j]
            cdf = h[:p + 1].sum() / N
            Ieq[i, j] = np.round(255 * cdf)

    return Ieq


def ecualizar_imagen_v2(I):
    """
    V2: tabla de transformación + indexación vectorizada. Se calcula la
    transformación para cada nivel de gris posible (LUT) y se aplica de
    forma global sobre la imagen.
    """
    L = 256
    N = I.size

    h = np.histogram(I, bins=L, range=(0, L))[0]

    # Tabla de transformación: g' = round((L-1)/N * sum_{i=0}^{g} H(i))
    lut = np.zeros(L, dtype=np.uint8)
    acumulado = 0
    for g in range(L):
        acumulado += h[g]
        lut[g] = np.round((L - 1) / N * acumulado)

    # Indexación vectorizada
    return lut[I]


def ecualizar_imagen_v3(I):
    """
    V3: recorrido por nivel de gris. Para cada uno de los 256 niveles se
    calcula su transformación y se aplica a todos los píxeles de ese nivel.
    """
    L = 256
    N = I.size

    h = np.histogram(I, bins=L, range=(0, L))[0]

    Ieq = np.zeros_like(I, dtype=np.uint8)
    acumulado = 0
    for g in range(L):
        acumulado += h[g]
        nuevo = np.round((L - 1) / N * acumulado)
        Ieq[I == g] = nuevo

    return Ieq
