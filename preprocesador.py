"""
preprocesador.py
Modulo de preprocesamiento de imagenes para el TFI de ART1
(Sistema de Clasificacion Automatizada de Frutas - Carpenter-Grossberg).

Implementa el pipeline acordado con la catedra:
  1. Segmentacion (elimina fondo, aisla la fruta)
  2. Centrado y escalado (normaliza posicion y tamano en el frame)  [sugerido por D. Lopez De Luise]
  3. Extraccion de features en escala de grises (forma, textura, tallo, compacidad)
  4. Extraccion de un feature de color (gradiente/uniformidad de color) [sugerido por D. Lopez De Luise]
  5. Binarizacion por umbrales -> fila lista para CarGross.py

Version del preprocesamiento
  - v1: el del primer avance (segmentar, extraer_features, binarizar). Se conserva
    SIN CAMBIOS para reproducir los modelos entregados (formato 1).
  - v2: mejoras documentadas en docs/informe_cambios.md (segmentacion por
    candidatos con puntaje de plausibilidad, tallo como protuberancia, tono
    circular). Se agregan al final de este archivo; cada una puede activarse por
    separado desde la configuracion de preprocesamiento guardada en el modelo.

Requiere: opencv-python, numpy
Uso tipico:
    from preprocesador import procesar_imagen
    fila = procesar_imagen("ruta/a/fruta.jpg", fruta_id="F001")
"""
import cv2
import numpy as np

TAM_CANVAS = 200  # lado del cuadrado final tras centrar y escalar

# Umbrales de binarizacion. Se calibraron observando los valores reales
# obtenidos sobre una muestra de Fruits-360 (ver informe_corridas).
# Documentados aca para que puedan ajustarse sin tocar la logica.
UMBRALES = {
    "aspecto_esferico": 0.80,     # >= : es_esferica = 1
    "textura_laplaciano": 150.0,  # >= : cascara_rugosa = 1
    "profundidad_defecto": 0.06,  # >= (relativo al alto) : tiene_tallo = 1
    "solidez": 0.90,              # >= : densidad_alta = 1 (proxy de compacidad)
    "gris_promedio": 130.0,       # <= : pigmentacion_oscura = 1
    "desvio_hue": 12.0,           # <= : color uniforme = 1 (homogeneo)
}


def segmentar(img_bgr):
    """Separa la fruta del fondo usando un enfoque hibrido robusto:
      1. Canal de Saturacion (HSV): las frutas tienen pigmentacion cromatica
         viva frente a fondos neutros (mesas, manteles, fondos blancos o grises).
      2. Si la mascara de saturacion es valida (entre 3% y 75% del area total),
         se adopta por su alta precision ante sombras y reflejos.
      3. Si no, se utiliza umbralizacion adaptativa de Otsu en escala de grises
         con deteccion automatica de polaridad (fondos claros u oscuros).
    """
    h, w = img_bgr.shape[:2]
    area_total = h * w
    kernel = np.ones((7, 7), np.uint8)

    # ── Metodo 1: Segmentacion por Saturacion Cromatica (HSV) ──
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    blur_s = cv2.GaussianBlur(sat, (7, 7), 0)
    _, bin_s = cv2.threshold(blur_s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    clean_s = cv2.morphologyEx(bin_s, cv2.MORPH_OPEN, kernel, iterations=2)
    clean_s = cv2.morphologyEx(clean_s, cv2.MORPH_CLOSE, kernel, iterations=3)

    conts_s, _ = cv2.findContours(clean_s, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if conts_s:
        mayor_s = max(conts_s, key=cv2.contourArea)
        area_s = cv2.contourArea(mayor_s)
        # Si el contorno de saturacion tiene proporcion coherente de fruta
        if 0.03 * area_total <= area_s <= 0.75 * area_total:
            masc = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(masc, [mayor_s], -1, 255, thickness=cv2.FILLED)
            return masc

    # ── Metodo 2: Segmentacion por Escala de Grises (Otsu Adaptativo) ──
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur_g = cv2.GaussianBlur(gris, (7, 7), 0)

    # Muestrear el perimetro exterior para evaluar el brillo del fondo
    margen_h = max(1, h // 25)
    margen_w = max(1, w // 25)
    borde_pixels = np.concatenate([
        blur_g[0:margen_h, :].flatten(),
        blur_g[-margen_h:, :].flatten(),
        blur_g[:, 0:margen_w].flatten(),
        blur_g[:, -margen_w:].flatten(),
    ])
    brillo_borde = float(np.mean(borde_pixels))
    otsu_val, _ = cv2.threshold(blur_g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fondo_es_claro = brillo_borde >= otsu_val
    modo_thresh = cv2.THRESH_BINARY_INV if fondo_es_claro else cv2.THRESH_BINARY

    _, bin_otsu = cv2.threshold(blur_g, otsu_val, 255, modo_thresh)
    cleaned = cv2.morphologyEx(bin_otsu, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)

    conts_g, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascara = np.zeros((h, w), dtype=np.uint8)
    if conts_g:
        mayor_g = max(conts_g, key=cv2.contourArea)
        cv2.drawContours(mascara, [mayor_g], -1, 255, thickness=cv2.FILLED)

    # Fallback seguro: si la mascara es anomalamente gigante (>85%) o nula (<2%)
    area_masc = cv2.countNonZero(mascara)
    if area_masc / area_total > 0.85 or area_masc / area_total < 0.02:
        modo_alt = cv2.THRESH_BINARY if fondo_es_claro else cv2.THRESH_BINARY_INV
        _, bin_alt = cv2.threshold(blur_g, otsu_val, 255, modo_alt)
        cleaned_alt = cv2.morphologyEx(bin_alt, cv2.MORPH_OPEN, kernel, iterations=2)
        cleaned_alt = cv2.morphologyEx(cleaned_alt, cv2.MORPH_CLOSE, kernel, iterations=3)
        conts_alt, _ = cv2.findContours(cleaned_alt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if conts_alt:
            mayor_alt = max(conts_alt, key=cv2.contourArea)
            area_alt = cv2.contourArea(mayor_alt)
            if 0.03 * area_total <= area_alt <= 0.80 * area_total:
                mascara = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mascara, [mayor_alt], -1, 255, thickness=cv2.FILLED)

    return mascara



def contorno_principal(mascara):
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None
    return max(contornos, key=cv2.contourArea)


def centrar_y_escalar(img_bgr, mascara, tam=TAM_CANVAS):
    """Recorta al bounding box de la fruta, la centra sobre un canvas
    cuadrado y la escala a tam x tam manteniendo la relacion de aspecto.
    Sin este paso, el tamano/posicion de la fruta en la foto original
    (que depende de la camara, no de la fruta) contaminaria las
    variables morfologicas."""
    c = contorno_principal(mascara)
    if c is None:
        raise ValueError("No se detecto fruta en la imagen (revisar segmentacion)")
    x, y, w, h = cv2.boundingRect(c)
    recorte = img_bgr[y:y + h, x:x + w]
    recorte_masc = mascara[y:y + h, x:x + w]

    escala = (tam * 0.9) / max(w, h)  # 0.9 deja un margen
    nuevo_w, nuevo_h = max(1, int(w * escala)), max(1, int(h * escala))
    recorte = cv2.resize(recorte, (nuevo_w, nuevo_h))
    recorte_masc = cv2.resize(recorte_masc, (nuevo_w, nuevo_h))

    canvas = np.full((tam, tam, 3), 255, dtype=np.uint8)
    canvas_masc = np.zeros((tam, tam), dtype=np.uint8)
    off_x, off_y = (tam - nuevo_w) // 2, (tam - nuevo_h) // 2
    canvas[off_y:off_y + nuevo_h, off_x:off_x + nuevo_w] = recorte
    canvas_masc[off_y:off_y + nuevo_h, off_x:off_x + nuevo_w] = recorte_masc
    return canvas, canvas_masc


def extraer_features(img_bgr, mascara):
    """Calcula los valores CONTINUOS de cada variable (antes de binarizar).
    Se devuelven crudos para poder loguearlos/calibrar umbrales."""
    c = contorno_principal(mascara)
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    area_hull = cv2.contourArea(hull)
    x, y, w, h = cv2.boundingRect(c)

    # es_esferica: relacion de aspecto del bounding box (1.0 = perfectamente esferica)
    aspecto = min(w, h) / max(w, h)

    # cascara_rugosa: energia de bordes (Laplaciano) dentro de la mascara
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    laplaciano = cv2.Laplacian(gris, cv2.CV_64F)
    textura = float(np.var(laplaciano[mascara > 0])) if area > 0 else 0.0

    # tiene_tallo: mayor defecto de convexidad relativo al alto de la fruta
    prof_defecto = 0.0
    if len(c) > 3:
        hull_idx = cv2.convexHull(c, returnPoints=False)
        # hull_idx puede ser None o tener < 4 puntos en contornos simples
        if hull_idx is not None and len(hull_idx) > 3:
            try:
                defectos = cv2.convexityDefects(c, hull_idx)
                if defectos is not None and defectos.ndim == 3:
                    # Forma esperada: (N, 1, 4) — acceso seguro
                    prof_max = float(np.max(defectos[:, 0, 3])) / 256.0
                    prof_defecto = prof_max / h
            except (cv2.error, IndexError, ValueError):
                # Contorno degenrado: no hay defectos calculables
                prof_defecto = 0.0

    # densidad_alta (proxy de compacidad): solidez = area / area del casco convexo
    solidez = (area / area_hull) if area_hull > 0 else 0.0

    # pigmentacion_oscura: nivel de gris promedio dentro de la mascara
    gris_prom = float(np.mean(gris[mascara > 0])) if area > 0 else 255.0

    # color uniforme (NUEVO, sugerido por Daniela): desvio del canal Hue
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    desvio_hue = float(np.std(hue[mascara > 0])) if area > 0 else 0.0

    return {
        "aspecto_esferico": aspecto,
        "textura_laplaciano": textura,
        "profundidad_defecto": prof_defecto,
        "solidez": solidez,
        "gris_promedio": gris_prom,
        "desvio_hue": desvio_hue,
    }


def binarizar(valores, umbrales=UMBRALES):
    return {
        "es_esferica": int(valores["aspecto_esferico"] >= umbrales["aspecto_esferico"]),
        "cascara_rugosa": int(valores["textura_laplaciano"] >= umbrales["textura_laplaciano"]),
        "tiene_tallo": int(valores["profundidad_defecto"] >= umbrales["profundidad_defecto"]),
        "densidad_alta": int(valores["solidez"] >= umbrales["solidez"]),
        "pigmentacion_oscura": int(valores["gris_promedio"] <= umbrales["gris_promedio"]),
        "color_uniforme": int(valores["desvio_hue"] <= umbrales["desvio_hue"]),
    }


def procesar_imagen(ruta, fruta_id, tam=TAM_CANVAS, umbrales=UMBRALES, devolver_crudos=False):
    img = cv2.imread(ruta)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
    mascara = segmentar(img)
    img_c, masc_c = centrar_y_escalar(img, mascara, tam)
    crudos = extraer_features(img_c, masc_c)
    fila = {"fruta_id": fruta_id, **binarizar(crudos, umbrales)}
    if devolver_crudos:
        return fila, crudos, img_c, masc_c
    return fila


# ===========================================================================
# PREPROCESAMIENTO v2 (mejoras; el codigo v1 de arriba no se modifica)
# ===========================================================================
TAM_TRABAJO = 320   # lado maximo con el que corre la segmentacion v2 (velocidad)
SAT_MIN_TONO = 25   # pixeles con saturacion menor tienen tono indefinido (casi grises)


def _impar(k):
    k = int(round(k))
    return k if k % 2 == 1 else k + 1


def _rellenar_contorno(binaria):
    """Deja solo el contorno externo mayor, relleno."""
    cont, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(binaria)
    if cont:
        cv2.drawContours(out, [max(cont, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return out


def _componente_central(binaria, min_frac=0.01):
    """Componente conexa con mejor (area x centralidad); descarta las < min_frac."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    h, w = binaria.shape
    mejor, mejor_s = None, -1.0
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a < min_frac * h * w:
            continue
        cx, cy = cen[i]
        d = np.hypot((cx - w / 2) / w, (cy - h / 2) / h)
        s = a * (1.0 - min(1.0, 1.2 * d))
        if s > mejor_s:
            mejor, mejor_s = i, s
    if mejor is None:
        return np.zeros_like(binaria)
    return ((lab == mejor).astype(np.uint8)) * 255


def _candidato_lab(img):
    """Distancia en Lab al color de fondo estimado en el marco de la imagen,
    umbral de Otsu. No supone color de fondo ni que la fruta sea saturada."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    mh, mw = max(2, h // 20), max(2, w // 20)
    borde = np.concatenate([lab[:mh].reshape(-1, 3), lab[-mh:].reshape(-1, 3),
                            lab[:, :mw].reshape(-1, 3), lab[:, -mw:].reshape(-1, 3)])
    fondo = np.median(borde, axis=0)
    dist = np.linalg.norm(lab - fondo, axis=2)
    d8 = np.clip(dist / (dist.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
    _, b = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = max(h, w)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((_impar(0.012 * m),) * 2, np.uint8))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((_impar(0.03 * m),) * 2, np.uint8))
    return _rellenar_contorno(_componente_central(b))


def _candidato_grabcut(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(w * 0.06), int(h * 0.06), int(w * 0.88), int(h * 0.88))
    cv2.setRNGSeed(0)  # kmeans interno de GrabCut: resultado reproducible
    try:
        cv2.grabCut(img, mask, rect, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((h, w), np.uint8)
    b = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((_impar(0.012 * max(h, w)),) * 2, np.uint8))
    return _rellenar_contorno(_componente_central(b))


def puntaje_plausibilidad(mascara):
    """Puntaje [0,1] de que la mascara sea UN objeto compacto, centrado y que no
    ocupe el borde de la imagen. Es una heuristica sin etiquetas (no usa la clase
    de la fruta) para elegir entre candidatos de segmentacion; 0 = descartable."""
    h, w = mascara.shape
    area = cv2.countNonZero(mascara) / (h * w)
    if area < 0.02 or area > 0.85:
        return 0.0
    cont, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cont, key=cv2.contourArea)
    a_hull = cv2.contourArea(cv2.convexHull(c))
    solidez = cv2.contourArea(c) / a_hull if a_hull > 0 else 0.0
    borde = (np.count_nonzero(mascara[0]) + np.count_nonzero(mascara[-1]) +
             np.count_nonzero(mascara[:, 0]) + np.count_nonzero(mascara[:, -1])) / (2.0 * (h + w))
    ys, xs = np.nonzero(mascara)
    cent = 1.0 - min(1.0, np.hypot(xs.mean() / w - 0.5, ys.mean() / h - 0.5) / 0.7071)
    return float(solidez * max(0.0, 1.0 - 2.5 * borde) * (0.6 + 0.4 * cent))


def segmentar_v2(img_bgr, devolver_info=False):
    """Segmentacion v2: genera candidatos independientes (distancia Lab al fondo,
    GrabCut con ventana central y la segmentacion v1) y elige el de mayor
    puntaje de plausibilidad. Si ninguno es plausible conserva el de v1 (la
    decision queda registrada en info['advertencia'])."""
    h, w = img_bgr.shape[:2]
    esc = min(1.0, TAM_TRABAJO / max(h, w))
    chica = cv2.resize(img_bgr, (max(1, int(w * esc)), max(1, int(h * esc))),
                       interpolation=cv2.INTER_AREA) if esc < 1.0 else img_bgr
    candidatos = {"lab": _candidato_lab(chica), "grabcut": _candidato_grabcut(chica),
                  "v1": segmentar(chica)}
    puntajes = {k: puntaje_plausibilidad(m) for k, m in candidatos.items()}
    elegido = max(puntajes, key=lambda k: puntajes[k])
    advertencia = None
    if puntajes[elegido] == 0.0:
        elegido = "v1"
        advertencia = "ningun candidato de segmentacion fue plausible"
    elif puntajes[elegido] < 0.35:
        advertencia = f"segmentacion dudosa (puntaje {puntajes[elegido]:.2f})"
    masc = candidatos[elegido]
    if esc < 1.0:
        masc = cv2.resize(masc, (w, h), interpolation=cv2.INTER_NEAREST)
    if devolver_info:
        return masc, {"candidato": elegido, "puntajes": puntajes, "advertencia": advertencia}
    return masc


def medir_tallo(mascara):
    """Protuberancia fina mas larga del contorno, relativa al ancho de la fruta.
    Mide un APENDICE (tallo/pedunculo/punta), no una concavidad: el criterio
    v1 (profundidad maxima de defectos de convexidad) mide hendiduras y quedaba
    en 0. Procedimiento: cuerpo = apertura de la mascara con un disco de 0.7 del
    radio inscripto maximo R; resto = mascara - cuerpo; se toma la componente
    del resto (area >= 8 px) mas alejada del cuerpo; valor = distancia / (2R).
    Se mide sobre el canvas centrado (resolucion comparable entre fotos)."""
    m = (mascara > 127).astype(np.uint8)
    if cv2.countNonZero(m) == 0:
        return 0.0
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    R = float(dt.max())
    if R < 3:
        return 0.0
    r = max(2, int(round(0.7 * R)))
    cuerpo = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    resto = cv2.subtract(m, cuerpo)
    dist = cv2.distanceTransform((1 - cuerpo).astype(np.uint8), cv2.DIST_L2, 5)
    n, lab, st, _ = cv2.connectedComponentsWithStats(resto, connectivity=8)
    mejor = 0.0
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 8:
            mejor = max(mejor, float(dist[lab == i].max()))
    return mejor / (2.0 * R)


def estadisticas_tono(img_bgr, mascara):
    """Estadisticos cromaticos dentro de la mascara con tono CIRCULAR.
    En OpenCV el tono H va de 0 a 179 y es ciclico: los rojos quedan en ambos
    extremos (~0 y ~175), y la desviacion estandar lineal que usaba v1
    (desvio_hue) los ve como muy heterogeneos aunque sean uniformes. Aqui:
      - desvio_tono: desvio circular ponderado por saturacion, en unidades de H
        de OpenCV (comparable con desvio_hue de v1);
      - tono_medio: tono circular medio en grados (0-360);
      - saturacion_media: S/255 medio;
      - frac_oscuros: fraccion de pixeles con V < 25% (zonas muy oscuras)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    dentro = mascara > 0
    if not dentro.any():
        return {"desvio_tono": 0.0, "tono_medio": 0.0, "saturacion_media": 0.0, "frac_oscuros": 0.0}
    H = hsv[:, :, 0][dentro].astype(np.float64)
    S = hsv[:, :, 1][dentro].astype(np.float64)
    V = hsv[:, :, 2][dentro].astype(np.float64)
    sel = S >= SAT_MIN_TONO
    if sel.sum() < 10:  # casi sin color: el tono no esta definido
        desvio, tono = 0.0, 0.0
    else:
        ang = H[sel] * 2.0 * np.pi / 180.0
        w = S[sel] / 255.0
        C, Sn = np.sum(w * np.cos(ang)), np.sum(w * np.sin(ang))
        Rm = min(1.0, np.hypot(C, Sn) / np.sum(w))
        desvio = float(np.sqrt(max(0.0, -2.0 * np.log(max(Rm, 1e-9)))) * 180.0 / np.pi / 2.0)
        tono = float((np.degrees(np.arctan2(Sn, C))) % 360.0)
    return {"desvio_tono": desvio, "tono_medio": tono,
            "saturacion_media": float(S.mean() / 255.0),
            "frac_oscuros": float(np.mean(V < 0.25 * 255))}


def centrar_y_escalar_v2(img_bgr, mascara, tam=TAM_CANVAS):
    """Igual que centrar_y_escalar, pero la mascara reescalada se vuelve a
    binarizar (v1 dejaba valores intermedios por la interpolacion lineal)."""
    img_c, masc_c = centrar_y_escalar(img_bgr, mascara, tam)
    return img_c, ((masc_c > 127).astype(np.uint8)) * 255


def profundidad_defecto_v2(mascara):
    """Mayor defecto de convexidad (hendidura) relativo al alto, independiente de
    la version de OpenCV.

    CAUSA RAIZ de tiene_tallo = 0 en todo el dataset: extraer_features (v1)
    exige defectos.ndim == 3. Con OpenCV 4.x convexityDefects devuelve (N,1,4),
    pero con OpenCV 5.x (el instalado: 5.0.0) devuelve (N,4); la condicion falla,
    el bloque se salta sin error y la profundidad queda en 0.0 para TODAS las
    imagenes. Aqui se aplana a (N,4) en ambos casos."""
    cont, _ = cv2.findContours((mascara > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cont:
        return 0.0
    c = max(cont, key=cv2.contourArea)
    if len(c) <= 3:
        return 0.0
    hull_idx = cv2.convexHull(c, returnPoints=False)
    if hull_idx is None or len(hull_idx) <= 3:
        return 0.0
    try:
        defectos = cv2.convexityDefects(c, hull_idx)
    except cv2.error:
        return 0.0
    if defectos is None:
        return 0.0
    prof = float(np.max(np.asarray(defectos).reshape(-1, 4)[:, 3])) / 256.0
    alto = cv2.boundingRect(c)[3]
    return prof / alto if alto > 0 else 0.0


# ===========================================================================
# Descriptores cromaticos adicionales (sugerencia docente: "informacion de
# color como metadato, ej. gradiente de colores") y variantes de medicion
# ===========================================================================
def nucleo_mascara(mascara, fraccion=0.08):
    """Mascara erosionada: deja fuera el borde de la fruta, donde la foto mezcla fruta y fondo.
    Radio = fraccion * sqrt(area). Si la erosion la vaciara, devuelve la mascara original."""
    area = cv2.countNonZero(mascara)
    if area == 0:
        return mascara
    r = max(2, int(round(fraccion * np.sqrt(area))))
    nuc = cv2.erode((mascara > 0).astype(np.uint8) * 255,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    return nuc if cv2.countNonZero(nuc) >= 0.1 * area else mascara


def factores_gris(img_bgr):
    """Factores por canal del supuesto 'mundo gris' (media de cada canal -> media global) calculados
    sobre TODA la foto original (con fondo): si se calcularan solo sobre la fruta, un fruto rojo se
    'corregiria' a gris y perderia el color que se quiere medir."""
    medias = img_bgr.reshape(-1, 3).astype(np.float64).mean(axis=0)
    return medias.mean() / np.maximum(medias, 1.0)


def aplicar_factores(img_bgr, factores):
    return np.clip(img_bgr.astype(np.float64) * factores, 0, 255).astype(np.uint8)


def _dist_ang(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def estadisticas_color(img_bgr, mascara):
    """Descriptores cromaticos dentro de la mascara. Ademas de los de estadisticas_tono():
      - dist_tono_naranja/verde/rojo/amarillo: distancia angular (grados) del tono medio a 30/110/0/60;
      - croma_medio: croma de Lab sqrt(a^2+b^2), poco sensible a la luminosidad;
      - gradiente_color: magnitud media del gradiente espacial de (a, b) (cuanto cambia el color de un punto
        a otro de la fruta; es el 'gradiente de colores' que menciono la docente)."""
    est = estadisticas_tono(img_bgr, mascara)
    dentro = mascara > 0
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a, b = lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0
    if not dentro.any():
        croma = grad = 0.0
    else:
        croma = float(np.mean(np.hypot(a, b)[dentro]))
        ab = cv2.GaussianBlur(np.dstack([a, b]), (0, 0), 1.0)
        gx, gy = cv2.Sobel(ab, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(ab, cv2.CV_32F, 0, 1, ksize=3)
        grad = float(np.mean(np.sqrt((gx ** 2 + gy ** 2).sum(axis=2))[dentro]))
    t = est["tono_medio"]
    est.update(dist_tono_naranja=_dist_ang(t, 30.0), dist_tono_verde=_dist_ang(t, 110.0),
               dist_tono_rojo=_dist_ang(t, 0.0), dist_tono_amarillo=_dist_ang(t, 60.0),
               croma_medio=croma, gradiente_color=grad)
    est.update(_fracciones_de_tono(img_bgr, mascara))
    return est


# Familias de tono (grados) para las fracciones de color dominante
FAMILIAS_TONO = {"frac_rojo": ((345.0, 360.0), (0.0, 15.0)), "frac_naranja": ((15.0, 45.0),),
                 "frac_amarillo": ((45.0, 70.0),), "frac_verde": ((70.0, 170.0),)}


def _fracciones_de_tono(img_bgr, mascara):
    """Fraccion de los pixeles con color definido (S >= SAT_MIN_TONO) dentro de la mascara cuyo tono cae en cada
    familia (rojo, naranja, amarillo, verde). Describe el color DOMINANTE, no solo el tono medio: una manzana
    rojo-amarilla tiene un tono medio naranja pero sus pixeles no son mayoritariamente naranjas."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    dentro = mascara > 0
    sel = dentro & (hsv[:, :, 1] >= SAT_MIN_TONO)
    out = {k: 0.0 for k in FAMILIAS_TONO}
    if sel.sum() >= 10:
        grados = hsv[:, :, 0][sel].astype(np.float64) * 2.0
        for k, rangos in FAMILIAS_TONO.items():
            out[k] = float(np.mean(np.any([(grados >= a) & (grados < b) for a, b in rangos], axis=0)))
    return out


def textura_normalizada(img_bgr, mascara):
    """Varianza del laplaciano dividida por el contraste de la fruta (varianza de grises) dentro de la mascara:
    menos dependiente de la iluminacion y del tipo de fruta que la varianza cruda."""
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    dentro = mascara > 0
    if not dentro.any():
        return 0.0
    lap = cv2.Laplacian(gris, cv2.CV_64F)
    return float(np.var(lap[dentro]) / (np.var(gris[dentro].astype(np.float64)) + 1.0))


# ---------------------------------------------------------------------------
# Normalizacion de la iluminacion (paso de preprocesamiento previo a segmentar y medir)
# ---------------------------------------------------------------------------
NORMALIZACIONES = ("ninguna", "estirar", "gris_mundo", "gamma_media")


def normalizar_iluminacion(img_bgr, modo="ninguna"):
    """Reduce la dependencia de la exposicion y del tinte de la foto ANTES de segmentar y medir. No cambia la
    definicion de ninguna variable: solo las condiciones en que se miden.
      'estirar'    : estiramiento de contraste (percentiles 1 y 99 de la luminosidad de toda la foto);
      'gris_mundo' : balance de blancos del supuesto 'mundo gris' (iguala la media de los tres canales);
      'gamma_media': gamma que lleva la luminosidad media de la foto a 0.5 (limitada a [0.4, 2.5])."""
    if modo == "ninguna":
        return img_bgr
    if modo not in NORMALIZACIONES:
        raise ValueError(f"normalizacion desconocida: {modo!r} (opciones: {NORMALIZACIONES})")
    if modo == "gris_mundo":
        return aplicar_factores(img_bgr, factores_gris(img_bgr))
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    if modo == "estirar":
        lo, hi = np.percentile(gris, [1, 99])
        if hi - lo < 8:
            return img_bgr
        return np.clip((img_bgr.astype(np.float64) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    media = float(np.clip(gris.mean() / 255.0, 0.02, 0.98))                      # gamma_media
    g = float(np.clip(np.log(0.5) / np.log(media), 0.4, 2.5))
    return np.clip(255.0 * (img_bgr.astype(np.float64) / 255.0) ** g, 0, 255).astype(np.uint8)
