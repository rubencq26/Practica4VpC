import numpy as np
import cv2

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
            h = np.histogram(seleccion, bins=256, range=(0,256))[0]
            p = int(I[i, j])
            Ieq[i, j] = np.round(255 * h[:p + 1].sum() / seleccion.size)
    return Ieq



def ecualizar_imagen_local(I, V, opcionRelleno="replicate"):
    Ieq = np.zeros_like(I, dtype=np.uint8)

    alto = V.shape[0] // 2
    ancho = V.shape[1] // 2

    bordes = {
        "replicate": cv2.BORDER_REPLICATE,
        "zeros": cv2.BORDER_CONSTANT,
        "reflect": cv2.BORDER_REFLECT
    }

    padding = cv2.copyMakeBorder(I, alto, alto, ancho, ancho, borderType=bordes[opcionRelleno.lower()])

    filas, columnas = I.shape

    for i in range(filas):
        for j in range(columnas):
            seleccion = padding[i:i+V.shape[0], j:V.shape[1]]
            h = np.bincount(seleccion.ravel(), minlength=256)

            lut = np.round(255 * np.cumsum(h) / seleccion.size).astype(np.uint8)
            Ieq[i, j] = lut[I[i, j]]
    return Ieq


def ecualizar_imagen_local_v3(I, V, opcionRelleno="replicate"):
    Ieq = np.zeros_like(I, dtype=np.uint8)

    bordes = {
        "replicate": cv2.BORDER_REPLICATE,
        "zeros": cv2.BORDER_CONSTANT,
        "reflect": cv2.BORDER_REFLECT
    }

    borde = bordes[opcionRelleno.lower()]
    kernel = V.astype(np.float32)

    for g in np.unique(I):
        mask = (I <= g).astype(np.float32)
        conteo = np.rint(cv2.filter2D(mask, -1, kernel, borderType=borde))

        sel = (I == g)
        Ieq[sel] = np.round(255 * conteo[sel] / V.size)
    return Ieq