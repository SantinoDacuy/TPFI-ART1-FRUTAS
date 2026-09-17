"""
probar_imagen.py
Script de prueba en vivo para clasificar imagenes individuales mediante ART1.
Grupo 2 — Redes Neuronales — UADER FCyT

Disenado para la defensa oral ante los profesores:
  1. Recibe una fotografia cualquiera de una fruta.
  2. Ejecuta el pipeline de OpenCV con polaridad de fondo adaptativa.
  3. Etapa 1 (Morfologia): Identifica si es Citrico, Tropical o Carozo/Manzana.
  4. Si es identificada como Manzana/Carozo:
     Ejecuta automaticamente la Etapa 2 (Calidad): Determina si es Premium,
     Comercial o Descarte con los umbrales de referencia calibrados.
  5. Reporta si la fruta resono con un prototipo conocido o si el sistema
     detecto NOVEDAD (caracteristica clave de ART1).
  6. Genera una imagen compuesta 'diagnostico_clasificacion.png' con los
     pasos del preprocesamiento para proyectar en la exposicion.

Uso:
    python probar_imagen.py --imagen "ruta/a/foto.jpg"
    python probar_imagen.py --imagen "datasets/processed_images_fruits/Good Quality_Fruits/Apple_Good/20190809_115448.jpg"
"""

import argparse
import os
import sys
import cv2
import numpy as np

# Rutas base
DIR_RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR_RAIZ)

from CarGross import RedART1, entrenar_modelo
from preprocesador import (
    segmentar, centrar_y_escalar, extraer_features, binarizar, TAM_CANVAS
)
from preprocesador_calidad import (
    extraer_crudos_calidad, binarizar_calidad, UMBRALES_REFERENCIA_CALIDAD
)

DIR_MODELOS = os.path.join(DIR_RAIZ, "modelos")
RUTA_CSV_MORF = os.path.join(DIR_RAIZ, "datasets", "dataset_morfologia.csv")
RUTA_CSV_CAL = os.path.join(DIR_RAIZ, "datasets", "dataset_calidad_real.csv")
MODELO_MORF = os.path.join(DIR_MODELOS, "art1_morfologia.json")
MODELO_CAL = os.path.join(DIR_MODELOS, "art1_calidad.json")


def asegurar_modelos(rho_morf=0.75, rho_cal=0.70):
    """Carga o entrena y persiste los modelos de Morfologia y Calidad."""
    os.makedirs(DIR_MODELOS, exist_ok=True)

    if os.path.isfile(MODELO_MORF):
        red_morf, meta_morf = RedART1.cargar(MODELO_MORF)
    else:
        print("  [INFO] Entrenando modelo de Morfologia (Etapa 1)...")
        red_morf, meta_morf = entrenar_modelo(RUTA_CSV_MORF, rho=rho_morf)
        red_morf.guardar(MODELO_MORF, meta_morf)

    if os.path.isfile(MODELO_CAL):
        red_cal, meta_cal = RedART1.cargar(MODELO_CAL)
    else:
        print("  [INFO] Entrenando modelo de Calidad de Manzanas (Etapa 2)...")
        red_cal, meta_cal = entrenar_modelo(RUTA_CSV_CAL, rho=rho_cal)
        red_cal.guardar(MODELO_CAL, meta_cal)

    return (red_morf, meta_morf), (red_cal, meta_cal)


def banner():
    print("\n" + "=" * 68)
    print("       SISTEMA DE CLASIFICACION AUTOMATIZADA DE FRUTAS — ART1")
    print("             Prueba en Vivo e Inferencia de Dos Etapas")
    print("=" * 68)


def crear_imagen_diagnostico(img_orig, mascara, img_centrada, texto_etapa1, texto_etapa2, ruta_salida):
    """Compone una imagen PNG panoramica con los paneles del preprocesamiento y resultados."""
    alto_canvas = 420
    ancho_canvas = 760
    lienzo = np.full((alto_canvas, ancho_canvas, 3), 30, dtype=np.uint8)

    # Encabezado
    cv2.putText(lienzo, "DIAGNOSTICO DE PREPROCESAMIENTO Y CLASIFICACION ART1",
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(lienzo, (20, 44), (ancho_canvas - 20, 44), (100, 100, 100), 1)

    # Preparar 3 vistas cuadradas (200x200):
    # 1. Original reescalada con borde
    h_orig, w_orig = img_orig.shape[:2]
    escala = 190.0 / max(h_orig, w_orig)
    nw, nh = max(1, int(w_orig * escala)), max(1, int(h_orig * escala))
    orig_res = cv2.resize(img_orig, (nw, nh))
    panel_orig = np.full((200, 200, 3), 45, dtype=np.uint8)
    oy, ox = (200 - nh) // 2, (200 - nw) // 2
    panel_orig[oy:oy + nh, ox:ox + nw] = orig_res

    # 2. Mascara en 3 canales con contorno verde
    panel_masc = cv2.cvtColor(cv2.resize(mascara, (200, 200)), cv2.COLOR_GRAY2BGR)
    contornos, _ = cv2.findContours(cv2.resize(mascara, (200, 200)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        cv2.drawContours(panel_masc, contornos, -1, (0, 255, 0), 2)

    # 3. Centrada y escalada
    panel_centr = cv2.resize(img_centrada, (200, 200))

    # Pegar paneles
    x_coords = [30, 275, 520]
    etiquetas = ["1. Foto Original", "2. Mascara Segmentada", "3. Centrado y Escalado"]
    for x, panel, etq in zip(x_coords, [panel_orig, panel_masc, panel_centr], etiquetas):
        lienzo[65:265, x:x + 200] = panel
        cv2.rectangle(lienzo, (x, 65), (x + 200, 265), (140, 140, 140), 1)
        cv2.putText(lienzo, etq, (x + 10, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    # Separador
    cv2.line(lienzo, (20, 305), (ancho_canvas - 20, 305), (80, 80, 80), 1)

    # Panel de resultados de texto
    cv2.putText(lienzo, f"ETAPA 1 (Morfologia): {texto_etapa1}",
                (30, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1, cv2.LINE_AA)
    if texto_etapa2:
        cv2.putText(lienzo, f"ETAPA 2 (Calidad):     {texto_etapa2}",
                    (30, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 120), 1, cv2.LINE_AA)
    else:
        cv2.putText(lienzo, "ETAPA 2: Omitida (Solo aplicable si la fruta es identificada como Manzana)",
                    (30, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)

    cv2.imwrite(ruta_salida, lienzo)


def main():
    parser = argparse.ArgumentParser(description="Clasifica una fruta individual mediante red ART1")
    parser.add_argument("--imagen", required=True, help="Ruta al archivo de imagen a probar")
    parser.add_argument("--salida-grafico", default="diagnostico_clasificacion.png",
                        help="Ruta donde guardar la composicion visual (default: diagnostico_clasificacion.png)")
    parser.add_argument("--aprender", action="store_true",
                        help="Si se activa, una fruta no reconocida crea un cluster nuevo (plasticidad activa)")
    args = parser.parse_args()

    banner()

    if not os.path.isfile(args.imagen):
        print(f"\n[ERROR] No se encontro el archivo de imagen: '{args.imagen}'")
        sys.exit(1)

    print(f"\nProcesando imagen: {args.imagen}")

    # 1. Cargar modelos
    (red_morf, meta_morf), (red_cal, meta_cal) = asegurar_modelos()

    # 2. Leer y preprocesar imagen
    img = cv2.imread(args.imagen)
    if img is None:
        print(f"[ERROR] OpenCV no pudo decodificar la imagen: {args.imagen}")
        sys.exit(1)

    mascara = segmentar(img)
    if cv2.countNonZero(mascara) == 0:
        print("[ERROR] No se pudo segmentar ningun contorno de fruta en la imagen.")
        sys.exit(1)

    img_c, masc_c = centrar_y_escalar(img, mascara, tam=TAM_CANVAS)

    # 3. ETAPA 1: Morfologia
    crudos_morf = extraer_features(img_c, masc_c)
    bin_morf = binarizar(crudos_morf)
    vec_morf = [bin_morf[col] for col in meta_morf["cols_features"]]

    print("\n" + "-" * 68)
    print("  ETAPA 1: CLASIFICACION MORFOLOGICA (Tipo de Fruta)")
    print("-" * 68)
    print("  Features continuos:")
    for k, v in crudos_morf.items():
        print(f"    - {k:<22}: {v:.4f}")
    print(f"  Vector binario F1     : {vec_morf} ({meta_morf['cols_features']})")

    eval_morf = red_morf.evaluar(vec_morf)
    texto_res_morf = ""
    es_manzana = False

    if eval_morf["resuena"]:
        cid = str(eval_morf["ganador"])
        info_c = meta_morf["mapeo_clusters"].get(cid, {"etiqueta": "Desconocido", "pureza": 1.0})
        etiqueta_tipo = info_c["etiqueta"].upper()
        similitud_pct = eval_morf["ratio"] * 100
        print(f"\n  >> RESONANCIA EXITOSA en Cluster {cid}:")
        print(f"     Tipo identificado  : {etiqueta_tipo}")
        print(f"     Similitud con T_j  : {similitud_pct:.1f}% (vigilancia rho={red_morf.rho})")
        print(f"     Pureza historica   : {info_c['pureza']*100:.0f}% ({info_c['n']} muestras en entrenamiento)")
        texto_res_morf = f"Cluster {cid} -> {etiqueta_tipo} (Match: {similitud_pct:.0f}%)"
        if "carozo" in info_c["etiqueta"].lower():
            es_manzana = True
    else:
        print("\n  >> NOVEDAD DETECTADA (No hubo resonancia con clusters existentes):")
        print(f"     Mejor coincidencia : {eval_morf['ratio']*100:.1f}% (requeria rho={red_morf.rho})")
        if args.aprender:
            nuevo_id = red_morf.clasificar(vec_morf, aprender=True)
            print(f"     [Plasticidad activa] Se creo y aprendio la nueva Categoria {nuevo_id}.")
            texto_res_morf = f"NOVEDAD -> Nueva Categoria {nuevo_id} Creada"
        else:
            print("     [Modo inferencia] El patron no coincide con los prototipos aprendidos.")
            texto_res_morf = f"NOVEDAD (Sin resonancia, match={eval_morf['ratio']*100:.0f}%)"

    # 4. ETAPA 2: Calidad (Solo si es Manzana/Carozo o si el usuario quiere evaluarla)
    texto_res_cal = None
    if es_manzana:
        print("\n" + "-" * 68)
        print("  ETAPA 2: EVALUACION DE CALIDAD COMERCIAL (Exclusivo Manzanas)")
        print("-" * 68)
        crudos_cal = extraer_crudos_calidad(img, mascara, img_c, masc_c)
        bin_cal = binarizar_calidad(crudos_cal, UMBRALES_REFERENCIA_CALIDAD)
        vec_cal = [bin_cal[col] for col in meta_cal["cols_features"]]

        print("  Features continuos de calidad:")
        for k, v in crudos_cal.items():
            print(f"    - {k:<24}: {v:.4f} (umbral ref: {UMBRALES_REFERENCIA_CALIDAD[k]:.1f})")
        print(f"  Vector binario F1 (calidad) : {vec_cal} ({meta_cal['cols_features']})")

        eval_cal = red_cal.evaluar(vec_cal)
        if eval_cal["resuena"]:
            cid_cal = str(eval_cal["ganador"])
            info_cal = meta_cal["mapeo_clusters"].get(cid_cal, {"etiqueta": "Comercial", "pureza": 1.0})
            etiqueta_cal = info_cal["etiqueta"].upper()
            similitud_cal_pct = eval_cal["ratio"] * 100
            print(f"\n  >> RESONANCIA EXITOSA en Cluster {cid_cal}:")
            print(f"     Calidad asignada   : {etiqueta_cal}")
            print(f"     Similitud con T_j  : {similitud_cal_pct:.1f}% (vigilancia rho={red_cal.rho})")
            print(f"     Pureza historica   : {info_cal['pureza']*100:.0f}%")
            texto_res_cal = f"Cluster {cid_cal} -> {etiqueta_cal} (Match: {similitud_cal_pct:.0f}%)"
        else:
            print("\n  >> NOVEDAD EN CALIDAD:")
            print(f"     Mejor coincidencia : {eval_cal['ratio']*100:.1f}%")
            texto_res_cal = f"NOVEDAD (Match: {eval_cal['ratio']*100:.0f}%)"
    else:
        print("\n" + "-" * 68)
        print("  ETAPA 2: OMITIDA")
        print("  (La fruta no fue clasificada como Manzana/Carozo; no aplica control de manzana)")
        print("-" * 68)

    # 5. Guardar visualizacion
    crear_imagen_diagnostico(img, mascara, img_c, texto_res_morf, texto_res_cal, args.salida_grafico)
    print(f"\n  [OK] Composicion visual de diagnostico guardada en: {args.salida_grafico}\n" + "=" * 68 + "\n")


if __name__ == "__main__":
    main()
