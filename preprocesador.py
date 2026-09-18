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
