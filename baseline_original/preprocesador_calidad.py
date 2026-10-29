"""
preprocesador_calidad.py
Extraccion de features de CALIDAD (Dataset 2: manzanas Premium/Comercial/
Descarte) - distinto de preprocesador.py, que es para MORFOLOGIA/variedad
(Dataset 1). Son dos conjuntos de variables distintos, definidos asi desde
el documento parcial.

Reutiliza segmentar() y centrar_y_escalar() de preprocesador.py (el fix de
Otsu ya validado sobre las fotos reales de calidad, 8/9 OK), pero calcula
un conjunto de variables DISTINTO: es_grande, es_pesada, textura_firme,
sin_defectos_superficie, color_homogeneo.

POR QUE PERCENTIL DE LOTE Y NO UMBRAL FIJO:
Estas fotos vienen de una fuente secundaria (Kaggle), no de una camara
propia a distancia fija sobre la cinta. El area en pixeles no es un
tamano fisico calibrado, y no hay umbrales absolutos confiables para
estas fotos especificas. Por eso las 5 variables se binarizan por
MEDIANA DENTRO DEL PROPIO LOTE que se procesa: son relativas a las
demas manzanas de esa corrida, no una medida fisica absoluta. Es la
misma logica que ya se habia acordado para es_grande/es_pesada,
extendida a las 5 por consistencia. Documentar esto en el manual como
limitacion (alcances y limitaciones).

Uso tipico:
    from preprocesador_calidad import procesar_lote_calidad
    filas = procesar_lote_calidad([("ruta1.jpg", "premium"), ...])
"""
import cv2
import numpy as np
from preprocesador import segmentar, centrar_y_escalar, contorno_principal, TAM_CANVAS

# Umbrales de referencia calibrados a partir de las medianas del dataset
# curado (180 manzanas reales: 60 Premium, 60 Comercial, 60 Descarte).
# Permiten evaluar frutas individuales en tiempo real sin requerir un lote completo.
UMBRALES_REFERENCIA_CALIDAD = {
    "area": 10320.5,
    "volumen_proxy": 8567.8,
    "textura_laplaciano": 439.7,
    "defecto_score": 0.171,
    "desvio_hue": 12.5,
}


def binarizar_calidad(c, umbrales=UMBRALES_REFERENCIA_CALIDAD):
    """Convierte los valores continuos de calidad a binarios (0/1)."""
    return {
        "es_grande": int(c["area"] >= umbrales["area"]),
        "es_pesada": int(c["volumen_proxy"] >= umbrales["volumen_proxy"]),
        "textura_firme": int(c["textura_laplaciano"] <= umbrales["textura_laplaciano"]),
        "sin_defectos_superficie": int(c["defecto_score"] <= umbrales["defecto_score"]),
        "color_homogeneo": int(c["desvio_hue"] <= umbrales["desvio_hue"]),
    }


def _defecto_score(gris, mascara):
    """Fraccion de pixeles dentro de la mascara notablemente mas oscuros
    que el promedio de la fruta (proxy de manchas/golpes/podredumbre)."""
    pix = gris[mascara > 0]
    if pix.size == 0:
        return 0.0
    media, std = float(pix.mean()), float(pix.std())
    oscuros = (gris < (media - std)) & (mascara > 0)
    return float(oscuros.sum()) / float((mascara > 0).sum())


def _medir_tamano(img_bgr_original, mascara_original):
    """area/volumen SIN normalizar escala: si se midiera despues de
    centrar_y_escalar() todas las frutas darian ~el mismo tamano (esa
    funcion normaliza tamano a proposito, para morfologia). Para calidad
    necesitamos lo contrario: el tamano relativo real en la foto original."""
    c = contorno_principal(mascara_original)
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    area_hull = cv2.contourArea(hull)
    solidez = (area / area_hull) if area_hull > 0 else 0.0
    return area, area * solidez


def _medir_superficie(img_bgr_centrado, mascara_centrada):
    """textura/defectos/color SI sobre el marco centrado y escalado: aca
    conviene un tamano de canvas consistente para que la varianza del
    Laplaciano y el conteo de pixeles oscuros sean comparables entre
    fotos con distinto zoom/resolucion original."""
    c = contorno_principal(mascara_centrada)
    area = cv2.contourArea(c)

    gris = cv2.cvtColor(img_bgr_centrado, cv2.COLOR_BGR2GRAY)
    laplaciano = cv2.Laplacian(gris, cv2.CV_64F)
    textura = float(np.var(laplaciano[mascara_centrada > 0])) if area > 0 else 0.0

    defecto = _defecto_score(gris, mascara_centrada)

    hsv = cv2.cvtColor(img_bgr_centrado, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    desvio_hue = float(np.std(hue[mascara_centrada > 0])) if area > 0 else 0.0

    return textura, defecto, desvio_hue


def extraer_crudos_calidad(img_original, mascara_original, img_centrado, mascara_centrada):
    """Valores CONTINUOS (sin binarizar) de las 5 variables de calidad."""
    area, volumen_proxy = _medir_tamano(img_original, mascara_original)
    textura, defecto, desvio_hue = _medir_superficie(img_centrado, mascara_centrada)
    return {
        "area": area,                       # tamano real relativo (SIN normalizar)
        "volumen_proxy": volumen_proxy,      # proxy de es_pesada (masa/volumen)
        "textura_laplaciano": textura,      # menor = superficie mas lisa/firme
        "defecto_score": defecto,           # menor = sin defectos visibles
        "desvio_hue": desvio_hue,           # menor = color mas homogeneo
    }


def procesar_lote_calidad(rutas_con_etiqueta, tam=TAM_CANVAS):
    """
    rutas_con_etiqueta: lista de (ruta_imagen, calidad_real).
    Devuelve una tupla (filas_binarias, crudos, medianas).
    Maneja errores por imagen individual sin cortar el lote completo
    (requisito de manejo de errores / corridas infructuosas).
    """
    crudos, etiquetas, errores = [], [], []
    for ruta, etiqueta in rutas_con_etiqueta:
        try:
            img = cv2.imread(ruta)
            if img is None:
                raise FileNotFoundError(ruta)
            mascara = segmentar(img)
            img_c, masc_c = centrar_y_escalar(img, mascara, tam)
            crudos.append(extraer_crudos_calidad(img, mascara, img_c, masc_c))
            etiquetas.append(etiqueta)
        except Exception as e:
            errores.append((ruta, str(e)))

    if errores:
        print(f"AVISO: {len(errores)} imagenes fallaron y se omitieron de este lote:")
        for ruta, err in errores[:10]:
            print(f"  - {ruta}: {err}")

    if not crudos:
        raise RuntimeError("Ninguna imagen del lote pudo procesarse.")

    def mediana(clave):
        vals = sorted(c[clave] for c in crudos)
        return vals[len(vals) // 2]

    med = {
        "area": mediana("area"),
        "volumen_proxy": mediana("volumen_proxy"),
        "textura_laplaciano": mediana("textura_laplaciano"),
        "defecto_score": mediana("defecto_score"),
        "desvio_hue": mediana("desvio_hue"),
    }

    filas = []
    for i, c in enumerate(crudos):
        filas.append({
            **binarizar_calidad(c, med),
            "calidad_real": etiquetas[i],
        })
    return filas, crudos, med


def procesar_imagen_calidad(ruta, fruta_id="F_TEST", tam=TAM_CANVAS,
                            umbrales=UMBRALES_REFERENCIA_CALIDAD, devolver_crudos=False):
    """Procesa una unica imagen de manzana y extrae las 5 variables de calidad
    binarizadas usando los umbrales de referencia calibrados.
    Ideal para la prueba en vivo ante los profesores.
    """
    img = cv2.imread(ruta)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
    mascara = segmentar(img)
    img_c, masc_c = centrar_y_escalar(img, mascara, tam)
    crudos = extraer_crudos_calidad(img, mascara, img_c, masc_c)
    fila = {"fruta_id": fruta_id, **binarizar_calidad(crudos, umbrales)}
    if devolver_crudos:
        return fila, crudos, img_c, masc_c, mascara
    return fila

