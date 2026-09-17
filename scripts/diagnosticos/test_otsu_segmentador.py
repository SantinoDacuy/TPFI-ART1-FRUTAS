"""
test_otsu_segmentador.py
Prueba el nuevo segmentador basado en Otsu + morfologia adaptativa
sobre las mismas 9 muestras de Apple que fallo el segmentador original.
Compara ambos metodos lado a lado. NO modifica preprocesador.py.
"""
import sys, os
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

def segmentar_original(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    fondo = (s < 30) & (v > 200)
    mascara = (~fondo).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    return mascara

def segmentar_otsu(img_bgr):
    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (7, 7), 0)
    _, bin_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((7, 7), np.uint8)
    cleaned = cv2.morphologyEx(bin_otsu, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=3)
    contornos, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascara_final = np.zeros_like(cleaned)
    if contornos:
        mayor = max(contornos, key=cv2.contourArea)
        cv2.drawContours(mascara_final, [mayor], -1, 255, thickness=cv2.FILLED)
    return mascara_final

def evaluar(img_bgr, mascara):
    total = mascara.size
    fg    = int(np.sum(mascara > 0))
    ratio = fg / total
    conts, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_mayor = int(cv2.contourArea(max(conts, key=cv2.contourArea))) if conts else 0
    valida = (0.05 < ratio < 0.95) and area_mayor > 500
    return ("OK  " if valida else "FAIL"), ratio, area_mayor

def main():
    print("=" * 75)
    print("COMPARACION: Segmentador ORIGINAL vs OTSU - 9 muestras de manzana")
    print("=" * 75)
    print(f"{'Imagen':<30} {'Cat':<12} {'ORIGINAL':^18} {'OTSU':^18}")
    print("-" * 75)

    ok_orig = ok_otsu = total = 0

    for cat, rutas in MUESTRAS.items():
        for ruta in rutas:
            total += 1
            img = cv2.imread(ruta)
            if img is None:
                print(f"  ERROR leyendo: {ruta}")
                continue
            m_orig = segmentar_original(img)
            m_otsu = segmentar_otsu(img)
            st_o, r_o, a_o = evaluar(img, m_orig)
            st_u, r_u, a_u = evaluar(img, m_otsu)
            ok_orig += int(st_o == "OK  ")
            ok_otsu += int(st_u == "OK  ")
            nombre = os.path.basename(ruta)[:28]
            print(f"  {nombre:<30} {cat:<12} [{st_o}]{r_o:5.1%}  [{st_u}]{r_u:5.1%}  area={a_u}px2")

    print("-" * 75)
    print(f"\nRESUMEN:")
    print(f"  Segmentador ORIGINAL : {ok_orig}/{total} OK")
    print(f"  Segmentador OTSU     : {ok_otsu}/{total} OK")
    if ok_otsu > ok_orig:
        print(f"\n  => OTSU GANA ({ok_otsu} vs {ok_orig}). Reemplazar en preprocesador.py")
    elif ok_otsu == ok_orig:
        print(f"\n  => Empate. Revisar imagenes individuales.")
    else:
        print(f"\n  => ORIGINAL es mejor. No cambiar.")
    print("=" * 75)

if __name__ == "__main__":
    main()
