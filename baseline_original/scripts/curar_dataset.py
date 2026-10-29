"""
scripts/curar_dataset.py
Script de curacion y limpieza exhaustiva del dataset de imagenes y generacion
de datasets CSV definitivos para el TFI ART1.

Criterios:
1. Selecciona 60 imagenes optimas por clase:
   - Morfologia: carozo (60), citrico (60), tropical (60) -> 180 filas
   - Calidad   : premium (60), comercial (60), descarte (60) -> 180 filas
2. Filtra imagenes ruidosas (multiples frutas, fallas de segmentacion, cortes).
3. Elimina del disco todas las imagenes descartadas y carpetas no usadas.
4. Genera dataset_morfologia.csv y dataset_calidad_real.csv.
5. Reentrena los modelos art1_morfologia.json y art1_calidad.json.
"""
import os
import sys
import shutil
import csv
import cv2
import numpy as np

DIR_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DIR_RAIZ)

from preprocesador import (
    segmentar, centrar_y_escalar, extraer_features, binarizar, UMBRALES, TAM_CANVAS
)
from preprocesador_calidad import (
    extraer_crudos_calidad, binarizar_calidad, procesar_lote_calidad, UMBRALES_REFERENCIA_CALIDAD
)
from CarGross import entrenar_modelo

BASE_IMGS = os.path.join(DIR_RAIZ, "datasets", "processed_images_fruits")
RUTA_CSV_MORF = os.path.join(DIR_RAIZ, "datasets", "dataset_morfologia.csv")
RUTA_CSV_CAL = os.path.join(DIR_RAIZ, "datasets", "dataset_calidad_real.csv")
DIR_MODELOS = os.path.join(DIR_RAIZ, "modelos")

OBJETIVO_POR_CLASE = 60


def evaluar_imagen_morfologia(ruta_img, tipo_esperado):
    """Evalua si una imagen cumple con estandares morfologicos estrictos."""
    img = cv2.imread(ruta_img)
    if img is None:
        return None, "No se pudo leer"
    
    h, w = img.shape[:2]
    area_total = h * w
    masc = segmentar(img)
    area_masc = cv2.countNonZero(masc)
    
    # 1. Proporcion de area coherente
    ratio_area = area_masc / area_total
    if ratio_area < 0.08 or ratio_area > 0.70:
        return None, f"Ratio de area invalido: {ratio_area:.2f}"
    
    # 2. Contorno principal dominante (evitar racimos o fragmentos de pasto)
    conts, _ = cv2.findContours(masc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not conts:
        return None, "Sin contornos"
    mayor = max(conts, key=cv2.contourArea)
    area_mayor = cv2.contourArea(mayor)
    if area_masc > 0 and (area_mayor / area_masc) < 0.85:
        return None, "Multiples objetos o fragmentacion"
    
    try:
        img_c, masc_c = centrar_y_escalar(img, masc, tam=TAM_CANVAS)
        crudos = extraer_features(img_c, masc_c)
    except Exception as e:
        return None, str(e)
    
    # 3. Validacion segun especie
    aspecto = crudos["aspecto_esferico"]
    solidez = crudos["solidez"]
    textura = crudos["textura_laplaciano"]
    desvio_hue = crudos["desvio_hue"]

    if tipo_esperado == "carozo":
        # Manzana: esferica/subesferica, compacta, piel relativamente lisa
        if aspecto < 0.76:
            return None, f"Carozo no esferico: {aspecto:.2f}"
        if solidez < 0.88:
            return None, f"Carozo baja solidez: {solidez:.2f}"
        if textura > 500:
            return None, f"Carozo demasiado rugoso/ruidoso: {textura:.1f}"

    elif tipo_esperado == "citrico":
        # Naranja: esferica, textura de piel porosa o pareja
        if aspecto < 0.80:
            return None, f"Citrico no esferico (posible racimo): {aspecto:.2f}"
        if desvio_hue > 25.0:
            return None, f"Citrico color heterogeneo/sucio: {desvio_hue:.1f}"

    elif tipo_esperado == "tropical":
        # Banana: silueta alargada caracteristica
        if aspecto > 0.65:
            return None, f"Banana demasiado redonda (racimo o corte): {aspecto:.2f}"

    return crudos, "OK"


def evaluar_imagen_calidad(ruta_img, calidad_esperada):
    """Evalua si una imagen de manzana representa limpiamente su clase de calidad."""
    img = cv2.imread(ruta_img)
    if img is None:
        return None, "No se pudo leer"
    
    masc = segmentar(img)
    if cv2.countNonZero(masc) == 0:
        return None, "Mascara vacia"
    
    try:
        img_c, masc_c = centrar_y_escalar(img, masc, tam=TAM_CANVAS)
        crudos = extraer_crudos_calidad(img, masc, img_c, masc_c)
    except Exception as e:
        return None, str(e)
    
    defecto = crudos["defecto_score"]
    area = crudos["area"]

    if calidad_esperada == "premium":
        # Sin sombras duras ni manchas graves
        if defecto > 0.22:
            return None, f"Premium con defecto alto/sombras: {defecto:.2f}"
        if area < 10000:
            return None, f"Premium de calibre demasiado chico: {area:.0f}"

    elif calidad_esperada == "comercial":
        # Calibre o superficie con leves irregularidades pero viable comercialmente
        if defecto > 0.38:
            return None, f"Comercial con defecto excesivo: {defecto:.2f}"

    elif calidad_esperada == "descarte":
        # Daño evidente real
        if defecto < 0.15:
            return None, f"Descarte sin defecto visible: {defecto:.2f}"

    return crudos, "OK"


def curar_y_limpiar():
    print("=" * 68)
    print("      INICIO DE CURACION Y LIMPIEZA EXHAUSTIVA DE DATASETS")
    print("=" * 68)

    # 1. Definicion de fuentes de imagenes
    carpeta_apple_good = os.path.join(BASE_IMGS, "Good Quality_Fruits", "Apple_Good")
    carpeta_apple_mixed = os.path.join(BASE_IMGS, "Mixed Qualit_Fruits", "Apple")
    carpeta_apple_bad = os.path.join(BASE_IMGS, "Bad Quality_Fruits", "Apple_Bad")
    carpeta_orange_good = os.path.join(BASE_IMGS, "Good Quality_Fruits", "Orange_Good")
    carpeta_banana_good = os.path.join(BASE_IMGS, "Good Quality_Fruits", "Banana_Good")

    imagenes_morf = {
        "carozo": [],
        "citrico": [],
        "tropical": []
    }

    imagenes_calidad = {
        "premium": [],
        "comercial": [],
        "descarte": []
    }

    # Recolectar para Morfologia
    print("\n[1/5] Filtrando y seleccionando imagenes optimas de Morfologia...")

    # Carozo (desde Apple_Good)
    for f in sorted(os.listdir(carpeta_apple_good)):
        if len(imagenes_morf["carozo"]) >= OBJETIVO_POR_CLASE:
            break
        p = os.path.join(carpeta_apple_good, f)
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        crudos, est = evaluar_imagen_morfologia(p, "carozo")
        if crudos is not None:
            imagenes_morf["carozo"].append((p, crudos))

    # Citrico (desde Orange_Good)
    for f in sorted(os.listdir(carpeta_orange_good)):
        if len(imagenes_morf["citrico"]) >= OBJETIVO_POR_CLASE:
            break
        p = os.path.join(carpeta_orange_good, f)
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        crudos, est = evaluar_imagen_morfologia(p, "citrico")
        if crudos is not None:
            imagenes_morf["citrico"].append((p, crudos))

    # Tropical (desde Banana_Good)
    for f in sorted(os.listdir(carpeta_banana_good)):
        if len(imagenes_morf["tropical"]) >= OBJETIVO_POR_CLASE:
            break
        p = os.path.join(carpeta_banana_good, f)
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        crudos, est = evaluar_imagen_morfologia(p, "tropical")
        if crudos is not None:
            imagenes_morf["tropical"].append((p, crudos))

    print(f"  - Carozo   : {len(imagenes_morf['carozo'])} imagenes seleccionadas")
    print(f"  - Citrico  : {len(imagenes_morf['citrico'])} imagenes seleccionadas")
    print(f"  - Tropical : {len(imagenes_morf['tropical'])} imagenes seleccionadas")

    # Recolectar para Calidad de Manzanas
    print("\n[2/5] Filtrando y seleccionando imagenes optimas de Calidad de Manzanas...")

    # Premium
    for f in sorted(os.listdir(carpeta_apple_good)):
        if len(imagenes_calidad["premium"]) >= OBJETIVO_POR_CLASE:
            break
        p = os.path.join(carpeta_apple_good, f)
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        crudos, est = evaluar_imagen_calidad(p, "premium")
        if crudos is not None:
            imagenes_calidad["premium"].append((p, crudos))

    # Comercial
    for f in sorted(os.listdir(carpeta_apple_mixed)):
        if len(imagenes_calidad["comercial"]) >= OBJETIVO_POR_CLASE:
            break
        p = os.path.join(carpeta_apple_mixed, f)
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        crudos, est = evaluar_imagen_calidad(p, "comercial")
        if crudos is not None:
            imagenes_calidad["comercial"].append((p, crudos))

    # Descarte
    for f in sorted(os.listdir(carpeta_apple_bad)):
        if len(imagenes_calidad["descarte"]) >= OBJETIVO_POR_CLASE:
            break
        p = os.path.join(carpeta_apple_bad, f)
        if not f.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        crudos, est = evaluar_imagen_calidad(p, "descarte")
        if crudos is not None:
            imagenes_calidad["descarte"].append((p, crudos))

    print(f"  - Premium  : {len(imagenes_calidad['premium'])} imagenes seleccionadas")
    print(f"  - Comercial: {len(imagenes_calidad['comercial'])} imagenes seleccionadas")
    print(f"  - Descarte : {len(imagenes_calidad['descarte'])} imagenes seleccionadas")

    # Guardar lista de archivos que debemos conservar
    archivos_a_conservar = set()
    for cat in imagenes_morf.values():
        for ruta, _ in cat:
            archivos_a_conservar.add(os.path.normpath(ruta))
    for cat in imagenes_calidad.values():
        for ruta, _ in cat:
            archivos_a_conservar.add(os.path.normpath(ruta))

    print(f"\n[3/5] Limpiando almacenamiento del dataset ({len(archivos_a_conservar)} imagenes unicas a conservar)...")
    eliminados = 0
    carpetas_vacias = 0

    for root, dirs, files in os.walk(BASE_IMGS, topdown=False):
        for f in files:
            p = os.path.normpath(os.path.join(root, f))
            if p not in archivos_a_conservar:
                try:
                    os.remove(p)
                    eliminados += 1
                except Exception:
                    pass
        # Eliminar carpetas que hayan quedado vacias
        if not os.listdir(root):
            try:
                os.rmdir(root)
                carpetas_vacias += 1
            except Exception:
                pass

    print(f"  - Se eliminaron {eliminados} archivos innecesarios/ruidosos.")
    print(f"  - Se removieron {carpetas_vacias} directorios vacios (Guava, Pomegranate, Lime, etc.).")

    # 4. Generar CSV de Morfologia
    print("\n[4/5] Generando datasets CSV...")
    filas_morf = []
    idx_morf = 1
    prefijos_morf = {"carozo": "FM", "citrico": "FC", "tropical": "FT"}

    for tipo, lista in imagenes_morf.items():
        pref = prefijos_morf[tipo]
        for ruta, crudos in lista:
            fid = f"{pref}{idx_morf:03d}"
            idx_morf += 1
            b = binarizar(crudos)
            filas_morf.append({
                "fruta_id": fid,
                **b,
                "tipo_real": tipo
            })

    import random
    random.seed(42)
    random.shuffle(filas_morf)

    cols_morf = ["fruta_id", "es_esferica", "cascara_rugosa", "tiene_tallo",
                 "densidad_alta", "pigmentacion_oscura", "color_uniforme", "tipo_real"]
    with open(RUTA_CSV_MORF, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols_morf)
        w.writeheader()
        w.writerows(filas_morf)

    print(f"  [OK] {len(filas_morf)} filas escritas en: {RUTA_CSV_MORF}")

    # Generar CSV de Calidad
    filas_cal = []
    idx_cal = 1
    prefijos_cal = {"premium": "PRM", "comercial": "COM", "descarte": "DES"}

    # Recalcular medianas sobre el lote curado de calidad
    rutas_cal_etq = []
    for calidad, lista in imagenes_calidad.items():
        for ruta, _ in lista:
            rutas_cal_etq.append((ruta, calidad))

    filas_cal_bin, crudos_cal, medianas_cal = procesar_lote_calidad(rutas_cal_etq)
    cols_cal = ["fruta_id", "es_grande", "es_pesada", "textura_firme",
                "sin_defectos_superficie", "color_homogeneo", "calidad_real"]

    filas_cal_completas = []
    for idx, (f_bin, (ruta, etq)) in enumerate(zip(filas_cal_bin, rutas_cal_etq), 1):
        fid = f"{prefijos_cal[etq]}{idx:03d}"
        filas_cal_completas.append({"fruta_id": fid, **f_bin, "calidad_real": etq})

    random.seed(42)
    random.shuffle(filas_cal_completas)

    with open(RUTA_CSV_CAL, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols_cal)
        w.writeheader()
        w.writerows(filas_cal_completas)

    print(f"  [OK] {len(filas_cal_completas)} filas escritas en: {RUTA_CSV_CAL}")
    print(f"  [INFO] Medianas de calidad calculadas: {medianas_cal}")

    # 5. Reentrenar modelos ART1
    print("\n[5/5] Reentrenando y persistiendo modelos ART1 con datasets limpios...")
    red_morf, meta_morf = entrenar_modelo(RUTA_CSV_MORF, rho=0.75)
    red_morf.guardar(os.path.join(DIR_MODELOS, "art1_morfologia.json"), meta_morf)
    print(f"  - Modelo Morfologia: {len(meta_morf['mapeo_clusters'])} clusters formados.")

    red_cal, meta_cal = entrenar_modelo(RUTA_CSV_CAL, rho=0.70)
    red_cal.guardar(os.path.join(DIR_MODELOS, "art1_calidad.json"), meta_cal)
    print(f"  - Modelo Calidad   : {len(meta_cal['mapeo_clusters'])} clusters formados.")

    print("\n" + "=" * 68)
    print("      LIMPIEZA, CURACION Y REENTRENAMIENTO COMPLETADOS CON EXITO")
    print("=" * 68)


if __name__ == "__main__":
    curar_y_limpiar()
