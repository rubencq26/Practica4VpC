import numpy as np
import cv2
from funciones_ecualizar_imagen import ecualizar_imagen_v1, ecualizar_imagen_v2, ecualizar_imagen_v3

def ecualizar_imagen_local_v1(I, V, opcionRelleno="replicate"):
    Ieq = np.zeros_like(I, dtype=np.uint8)

    alto = V.shape[0] // 2
    ancho = V.shape[1] // 2

    bordes = {
        "replicate":cv2.BORDER_REPLICATE,
        "zeros":cv2.BORDER_CONSTANT,
        "reflect":cv2.BORDER_REFLECT
    }

    padding = cv2.copyMakeBorder(I, alto, alto, ancho, ancho, borderType=bordes[opcionRelleno.lower()])

    filas = I.shape[0]
    columnas = I.shape[1]

    for i in range(filas):
        for j in range(columnas):
            seleccion = padding[i:i+V.shape[0], j:j+V.shape[1]]
            Ieq[i, j] = ecualizar_imagen_v1(seleccion)[alto, ancho]
    return Ieq
            



def ecualizar_imagen_local_v2(I, V, opcionRelleno="replicate"):
    Ieq = np.zeros_like(I, dtype=np.uint8)

    alto = V.shape[0] // 2
    ancho = V.shape[1] // 2

    bordes = {
        "replicate":cv2.BORDER_REPLICATE,
        "zeros":cv2.BORDER_CONSTANT,
        "reflect":cv2.BORDER_REFLECT
    }

    padding = cv2.copyMakeBorder(I, alto, alto, ancho, ancho, borderType=bordes[opcionRelleno.lower()])

    filas = I.shape[0]
    columnas = I.shape[1]

    for i in range(filas):
        for j in range(columnas):
            seleccion = padding[i:i+V.shape[0], j:j+V.shape[1]]
            Ieq[i, j] = ecualizar_imagen_v2(seleccion)[alto, ancho]
    return Ieq
            


def ecualizar_imagen_local_v3(I, V, opcionRelleno="replicate"):
    Ieq = np.zeros_like(I, dtype=np.uint8)

    alto = V.shape[0] // 2
    ancho = V.shape[1] // 2

    bordes = {
        "replicate":cv2.BORDER_REPLICATE,
        "zeros":cv2.BORDER_CONSTANT,
        "reflect":cv2.BORDER_REFLECT
    }

    padding = cv2.copyMakeBorder(I, alto, alto, ancho, ancho, borderType=bordes[opcionRelleno.lower()])

    filas = I.shape[0]
    columnas = I.shape[1]

    for i in range(filas):
        for j in range(columnas):
            seleccion = padding[i:i+V.shape[0], j:j+V.shape[1]]
            Ieq[i, j] = ecualizar_imagen_v3(seleccion)[alto, ancho]
    return Ieq


def ecualizar_imagen_local_por_ventanas(I, V, opcionRelleno="replicate", W=(1,1)):
    Ieq_zonal = np.zeros_like(I, dtype=np.uint8)

    filas = I.shape[0]
    columnas = I.shape[1]
    

    for i in range(W[0]):
        for j in range(W[1]):
            h0 = filas // W[0] * i
            h1 = filas if i == W[0] else filas // W[0] * (i + 1)

            w0 = columnas // W[1] * j
            w1 = columnas if j == W[1] else columnas // W[1] * (j + 1)

            chunk = I[h0:h1, w0:w1]
            chunk_eq = ecualizar_imagen_local_v2(chunk, V, opcionRelleno)
            Ieq_zonal[h0:h1, w0:w1] = chunk_eq
    return Ieq_zonal