import numpy as np
import cv2

def especificar_imagen(I_canal, I_patron_canal, ROI_img=None, ROI_img_patron=None):
    """
    Especificación de histograma por canal usando OpenCV,
    considerando opcionalmente regiones de interés (ROI).

    Parámetros:
    - I_canal: Canal de la imagen a procesar (2D numpy array).
    - I_patron_canal: Canal de la imagen patrón (2D numpy array).
    - ROI_img: Máscara booleana de la imagen a procesar (opcional).
    - ROI_img_patron: Máscara booleana de la imagen patrón (opcional).

    Retorna:
    - I_canal_eq: Canal especificado tras aplicar la LUT.
    - LUT: Look-Up Table calculada para la especificación.
    """

    bins = 256
    
    # Convertir imágenes a uint8 si no lo son
    if I_canal.dtype != np.uint8:
        I_canal = np.clip(I_canal, 0, 255).astype(np.uint8)
    if I_patron_canal.dtype != np.uint8:
        patron_canal = np.clip(I_patron_canal, 0, 255).astype(np.uint8)

    # Normalizar ROIs si existen a 0-1 uint8 
    if ROI_img is not None:
        ROI_img = (ROI_img > 0).astype(np.uint8)
    else:
        ROI_img = None

    if ROI_img_patron is not None:
        ROI_img_patron = (ROI_img_patron > 0).astype(np.uint8)
    else:
        ROI_img_patron = None

    # Calcular histogramas usando máscaras si se proporcionan
    hist_img = cv2.calcHist([I_canal], [0],
                             ROI_img,
                             [bins], [0, bins]).flatten()
    
    hist_pat = cv2.calcHist([I_patron_canal], [0],
                             ROI_img_patron,
                             [bins], [0, bins]).flatten()

    # Evitar división por cero
    if np.sum(hist_img) == 0 or np.sum(hist_pat) == 0:
        raise ValueError("Histograma vacío: comprueba ROI o imágenes.")

    # Calcular CDF normalizado
    Hnorm_img = np.cumsum(hist_img) / np.sum(hist_img)
    Hnorm_pat = np.cumsum(hist_pat) / np.sum(hist_pat)

    # Construir LUT normalizada [0, 1]
    LUT = np.interp(Hnorm_img, Hnorm_pat, np.linspace(0, 1, bins))

    # Aplicar LUT y convertir a uint8
    I_canal_eq = np.round((LUT[I_canal] * 255)).astype(np.uint8)

    return I_canal_eq, LUT