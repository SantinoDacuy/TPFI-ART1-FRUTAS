"""
diagnostico_segmentacion.py
Prueba la funcion segmentar() de preprocesador.py sobre muestras reales
del dataset de calidad (Apple_Good, Apple_Bad, Apple/Mixed).
Reporta: pixeles foreground, area del contorno principal, ratio fruta/total,
y si la mascara parece valida. NO modifica ningun archivo.
"""
import sys
import os

# Asegura que se importa el modulo local, no uno instalado
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

BASE = r"C:\Users\santi\Downloads\tfi-art1-frutas\datasets\processed_images_fruits"

MUESTRAS = {
    "Apple_Good": [
        os.path.join(BASE, "Good Quality_Fruits", "Apple_Good", "20190809_115439.jpg"),
        os.path.join(BASE, "Good Quality_Fruits", "Apple_Good", "20190809_115448.jpg"),
        os.path.join(BASE, "Good Quality_Fruits", "Apple_Good", "20190809_115451.jpg"),
    ],
    "Apple_Bad": [
        os.path.join(BASE, "Bad Quality_Fruits", "Apple_Bad", "IMG20200728175856.jpg"),
        os.path.join(BASE, "Bad Quality_Fruits", "Apple_Bad", "IMG20200728175907.jpg"),
        os.path.join(BASE, "Bad Quality_Fruits", "Apple_Bad", "IMG20200728175908.jpg"),
    ],
    "Apple_Mixed": [
        os.path.join(BASE, "Mixed Qualit_Fruits", "Apple", "IMG20200728125930.jpg"),
        os.path.join(BASE, "Mixed Qualit_Fruits", "Apple", "IMG20200728125932.jpg"),
        os.path.join(BASE, "Mixed Qualit_Fruits", "Apple", "IMG20200728125933.jpg"),
    ],
}


def segmentar(img_bgr):
    """Copia exacta de preprocesador.segmentar() para aislar el test."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    fondo = (s < 30) & (v > 200)
    mascara = (~fondo).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    return mascara


def analizar_mascara(img_bgr, mascara, nombre_archivo):
    total_px = mascara.size
    fg_px    = int(np.sum(mascara > 0))
    ratio    = fg_px / total_px

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        area_contorno = 0
        n_contornos   = 0
        valida         = False
    else:
        n_contornos   = len(contornos)
        mayor         = max(contornos, key=cv2.contourArea)
        area_contorno = int(cv2.contourArea(mayor))
        # Heuristicas de validez:
        # - ratio foreground entre 5% y 95% (no es todo fondo ni toda fruta)
        # - contorno principal > 500 px^2
        valida = (0.05 < ratio < 0.95) and (area_contorno > 500)

    # Analisis del fondo: cuantos pixeles son "blanco puro" original
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s_canal = hsv[:, :, 1]
    v_canal = hsv[:, :, 2]
    px_blanco_puro = int(np.sum((s_canal < 30) & (v_canal > 200)))
    ratio_blanco = px_blanco_puro / total_px

    estado = "✅ OK" if valida else "❌ FALLO"

    print(f"  [{estado}] {os.path.basename(nombre_archivo)}")
    print(f"         Imagen: {img_bgr.shape[1]}x{img_bgr.shape[0]} px")
    print(f"         Fondo blanco puro: {ratio_blanco:.1%} del frame")
    print(f"         Foreground (mascara): {ratio:.1%} del frame  ({fg_px} px)")
    print(f"         Contornos detectados: {n_contornos}  |  Area contorno principal: {area_contorno} px²")
    if not valida:
        if ratio_blanco < 0.10:
            print(f"         ⚠️  CAUSA: fondo NO es blanco — segmentador por saturacion falla aqui")
        elif ratio > 0.95:
            print(f"         ⚠️  CAUSA: casi todo el frame es foreground — umbral demasiado permisivo")
        elif area_contorno <= 500:
            print(f"         ⚠️  CAUSA: contorno muy pequeno — segmento ruido, no fruta")
    return valida


def main():
    print("=" * 65)
    print("DIAGNOSTICO DE SEGMENTACION — dataset calidad (manzanas)")
    print("=" * 65)

    total_ok = 0
    total_imgs = 0

    for categoria, rutas in MUESTRAS.items():
        print(f"\n📁 {categoria}")
        print("-" * 50)
        ok_cat = 0
        for ruta in rutas:
            if not os.path.exists(ruta):
                print(f"  ⚠️  No encontrado: {ruta}")
                total_imgs += 1
                continue
            img = cv2.imread(ruta)
            if img is None:
                print(f"  ❌ No se pudo leer: {ruta}")
                total_imgs += 1
                continue
            mascara = segmentar(img)
            ok = analizar_mascara(img, mascara, ruta)
            ok_cat += int(ok)
            total_imgs += 1
        total_ok += ok_cat
        print(f"  → {ok_cat}/{len(rutas)} imágenes OK en {categoria}")

    print("\n" + "=" * 65)
    print(f"RESUMEN: {total_ok}/{total_imgs} imágenes con segmentación válida")
    if total_ok < total_imgs:
        print("⚠️  La segmentación actual NO es robusta para este dataset.")
        print("   Solución sugerida: reemplazar segmentador por uno adaptativo")
        print("   (e.g. Otsu + morfología, o umbral HSV ampliado para fondos variables).")
    else:
        print("✅ Segmentación funciona correctamente en las 9 muestras.")
    print("=" * 65)


if __name__ == "__main__":
    main()
