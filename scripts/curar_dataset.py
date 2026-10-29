"""
scripts/curar_dataset.py
Curacion del banco de imagenes y generacion de los datasets CSV del primer
avance. SEGURO POR DEFECTO: seleccionar y eliminar son pasos distintos, nada se
borra ni se sobrescribe sin una orden explicita, y los resultados salen en una
carpeta aparte (salida_curacion/), no en datasets/.

Subcomandos
  seleccionar  Evalua TODAS las imagenes con los criterios del primer avance y
               escribe, en --salida:
                 seleccion_curacion.csv        una fila por imagen y etapa: decision
                                                (aceptada / rechazada / excedente) y motivo
                 datasets/dataset_morfologia.csv, datasets/dataset_calidad_real.csv
                                                (60 filas por clase, binarizacion del primer avance)
               No modifica imagenes ni datasets existentes.
  eliminar     Toma seleccion_curacion.csv y lista las imagenes que NINGUNA etapa acepto.
               Por defecto es una SIMULACION (no hace nada). Con --ejecutar las MUEVE a
               --destino (reversible); solo con --borrar-definitivamente las elimina.

Criterios de exclusion (los del primer avance, sin cambios; ver evaluar_imagen_*):
  Morfologia: area de la mascara 8%-70% de la imagen; un objeto dominante (>= 85% del
              area); carozo: aspecto >= 0.76, solidez >= 0.88, textura <= 500;
              citrico: aspecto >= 0.80, desvio de tono <= 25; tropical: aspecto <= 0.65.
  Calidad   : mascara no vacia; premium: defecto <= 0.22 y area >= 10000; comercial:
              defecto <= 0.38; descarte: defecto >= 0.15. Tope de 60 por clase, en orden
              alfabetico de archivo.
ADVERTENCIA: varios criterios usan las mismas medidas que despues alimentan a la red;
curar con ellos hace el banco mas facil que el uso real (ver docs/informe_cambios.md).

Uso:
    python scripts/curar_dataset.py seleccionar
    python scripts/curar_dataset.py eliminar --seleccion salida_curacion/seleccion_curacion.csv
    python scripts/curar_dataset.py eliminar --seleccion ... --ejecutar --destino salida_curacion/descartadas
"""
import argparse
import csv
import os
import random
import shutil
import sys

import cv2

DIR_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DIR_RAIZ)

import configuracion as cfg
from preprocesador import (TAM_CANVAS, binarizar, centrar_y_escalar, extraer_features, segmentar)
from preprocesador_calidad import extraer_crudos_calidad, procesar_lote_calidad
from validacion import COD_ENTRADA, COD_OK, COD_USO

OBJETIVO_POR_CLASE = 60
SALIDA_POR_DEFECTO = os.path.join(DIR_RAIZ, "salida_curacion")
EXT = (".jpg", ".jpeg", ".png")

# (subcarpeta, etapa, clase)
FUENTES = (
    ("Good Quality_Fruits/Apple_Good", "morfologia", "carozo"),
    ("Good Quality_Fruits/Orange_Good", "morfologia", "citrico"),
    ("Good Quality_Fruits/Banana_Good", "morfologia", "tropical"),
    ("Good Quality_Fruits/Apple_Good", "calidad", "premium"),
    ("Mixed Qualit_Fruits/Apple", "calidad", "comercial"),
    ("Bad Quality_Fruits/Apple_Bad", "calidad", "descarte"),
)
COLS_MORF = ["fruta_id", "es_esferica", "cascara_rugosa", "tiene_tallo", "densidad_alta",
             "pigmentacion_oscura", "color_uniforme", "tipo_real"]
COLS_CAL = ["fruta_id", "es_grande", "es_pesada", "textura_firme", "sin_defectos_superficie",
            "color_homogeneo", "calidad_real"]


def evaluar_imagen_morfologia(ruta_img, tipo_esperado):
    """Evalua si una imagen cumple con estandares morfologicos estrictos."""
    img = cv2.imread(ruta_img)
    if img is None:
        return None, "No se pudo leer"

    h, w = img.shape[:2]
    area_total = h * w
    masc = segmentar(img)
    area_masc = cv2.countNonZero(masc)

    ratio_area = area_masc / area_total
    if ratio_area < 0.08 or ratio_area > 0.70:
        return None, f"Ratio de area invalido: {ratio_area:.2f}"

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
    except Exception as e:  # noqa: BLE001
        return None, str(e)

    aspecto = crudos["aspecto_esferico"]
    solidez = crudos["solidez"]
    textura = crudos["textura_laplaciano"]
    desvio_hue = crudos["desvio_hue"]

    if tipo_esperado == "carozo":
        if aspecto < 0.76:
            return None, f"Carozo no esferico: {aspecto:.2f}"
        if solidez < 0.88:
            return None, f"Carozo baja solidez: {solidez:.2f}"
        if textura > 500:
            return None, f"Carozo demasiado rugoso/ruidoso: {textura:.1f}"
    elif tipo_esperado == "citrico":
        if aspecto < 0.80:
            return None, f"Citrico no esferico (posible racimo): {aspecto:.2f}"
        if desvio_hue > 25.0:
            return None, f"Citrico color heterogeneo/sucio: {desvio_hue:.1f}"
    elif tipo_esperado == "tropical":
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
    except Exception as e:  # noqa: BLE001
        return None, str(e)

    defecto = crudos["defecto_score"]
    area = crudos["area"]

    if calidad_esperada == "premium":
        if defecto > 0.22:
            return None, f"Premium con defecto alto/sombras: {defecto:.2f}"
        if area < 10000:
            return None, f"Premium de calibre demasiado chico: {area:.0f}"
    elif calidad_esperada == "comercial":
        if defecto > 0.38:
            return None, f"Comercial con defecto excesivo: {defecto:.2f}"
    elif calidad_esperada == "descarte":
        if defecto < 0.15:
            return None, f"Descarte sin defecto visible: {defecto:.2f}"

    return crudos, "OK"


def _listar(carpeta):
    return [f for f in sorted(os.listdir(carpeta)) if f.lower().endswith(EXT)]


def seleccionar(raiz_imagenes, salida, sobrescribir=False):
    """Evalua todas las imagenes y escribe seleccion_curacion.csv + datasets/*.csv en 'salida'."""
    destino_ds = os.path.join(salida, "datasets")
    rutas_salida = [os.path.join(salida, "seleccion_curacion.csv"),
                    os.path.join(destino_ds, "dataset_morfologia.csv"),
                    os.path.join(destino_ds, "dataset_calidad_real.csv")]
    existentes = [r for r in rutas_salida if os.path.exists(r)]
    if existentes and not sobrescribir:
        raise FileExistsError("ya existen archivos de salida: " + ", ".join(existentes) +
                              ". Use otra --salida o --sobrescribir (los originales de datasets/ no se tocan).")
    decisiones, aceptadas = [], {"morfologia": {}, "calidad": {}}
    for sub, etapa, clase in FUENTES:
        carpeta = os.path.join(raiz_imagenes, *sub.split("/"))
        if not os.path.isdir(carpeta):
            raise FileNotFoundError(f"no existe la carpeta de imagenes '{carpeta}'")
        evaluar = evaluar_imagen_morfologia if etapa == "morfologia" else evaluar_imagen_calidad
        n_ok = 0
        for f in _listar(carpeta):
            ruta = os.path.join(carpeta, f)
            crudos, motivo = evaluar(ruta, clase)
            if crudos is None:
                decision = "rechazada"
            elif n_ok >= OBJETIVO_POR_CLASE:
                decision, motivo = "excedente", f"supera el tope de {OBJETIVO_POR_CLASE} por clase"
            else:
                decision, n_ok = "aceptada", n_ok + 1
                aceptadas[etapa].setdefault(clase, []).append((ruta, crudos))
            decisiones.append({"ruta": sub + "/" + f, "etapa": etapa, "clase": clase,
                               "decision": decision, "motivo": motivo})
    os.makedirs(destino_ds, exist_ok=True)
    with open(rutas_salida[0], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["ruta", "etapa", "clase", "decision", "motivo"])
        w.writeheader()
        w.writerows(decisiones)

    # --- datasets (binarizacion del primer avance) ------------------------------------------------------
    filas_m, i = [], 1
    pref = {"carozo": "FM", "citrico": "FC", "tropical": "FT"}
    for tipo in ("carozo", "citrico", "tropical"):
        for ruta, crudos in aceptadas["morfologia"].get(tipo, []):
            filas_m.append({"fruta_id": f"{pref[tipo]}{i:03d}", **binarizar(crudos), "tipo_real": tipo})
            i += 1
    rutas_cal = [(ruta, q) for q in ("premium", "comercial", "descarte")
                 for ruta, _ in aceptadas["calidad"].get(q, [])]
    filas_c = []
    medianas = None
    if rutas_cal:
        bin_, _, medianas = procesar_lote_calidad(rutas_cal)
        pref_c = {"premium": "PRM", "comercial": "COM", "descarte": "DES"}
        filas_c = [{"fruta_id": f"{pref_c[q]}{idx:03d}", **b, "calidad_real": q}
                   for idx, (b, (_, q)) in enumerate(zip(bin_, rutas_cal), 1)]
    random.seed(42)
    random.shuffle(filas_m)
    random.seed(42)
    random.shuffle(filas_c)
    for ruta, cols, filas in ((rutas_salida[1], COLS_MORF, filas_m), (rutas_salida[2], COLS_CAL, filas_c)):
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(filas)
    return decisiones, filas_m, filas_c, medianas, rutas_salida


def imagenes_sin_uso(ruta_seleccion):
    """Rutas (relativas a la raiz de imagenes) que ninguna etapa acepto."""
    por_ruta = {}
    with open(ruta_seleccion, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            por_ruta.setdefault(r["ruta"], []).append(r["decision"])
    return sorted(r for r, d in por_ruta.items() if "aceptada" not in d)


def cmd_seleccionar(a):
    try:
        decisiones, fm, fc, med, rutas = seleccionar(a.imagenes, a.salida, a.sobrescribir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}\nAyuda: python scripts/curar_dataset.py seleccionar --help", file=sys.stderr)
        return COD_ENTRADA
    except FileExistsError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return COD_USO
    print("Seleccion terminada (no se modifico ninguna imagen ni dataset existente).")
    for etapa in ("morfologia", "calidad"):
        cuenta = {}
        for d in decisiones:
            if d["etapa"] == etapa:
                cuenta.setdefault(d["clase"], {}).setdefault(d["decision"], 0)
                cuenta[d["clase"]][d["decision"]] += 1
        print(f"  {etapa}: " + "; ".join(f"{c}: {dict(v)}" for c, v in cuenta.items()))
    print(f"  Filas escritas: morfologia {len(fm)}, calidad {len(fc)}")
    print(f"  Archivos: {rutas[0]}\n            {rutas[1]}\n            {rutas[2]}")
    print(f"  Imagenes que ninguna etapa acepta: {len(imagenes_sin_uso(rutas[0]))} "
          "(ver 'eliminar' para simular su retiro)")
    return COD_OK


def cmd_eliminar(a):
    if not os.path.isfile(a.seleccion):
        print(f"ERROR: no se encontro '{a.seleccion}'. Genere la seleccion con: "
              "python scripts/curar_dataset.py seleccionar", file=sys.stderr)
        return COD_ENTRADA
    sobran = [r for r in imagenes_sin_uso(a.seleccion) if os.path.isfile(os.path.join(a.imagenes, *r.split("/")))]
    print(f"{len(sobran)} imagen(es) no aceptadas por ninguna etapa.")
    for r in sobran[:15]:
        print(f"  - {r}")
    if len(sobran) > 15:
        print(f"  ... y {len(sobran) - 15} mas")
    if not a.ejecutar:
        print("\nSIMULACION: no se movio ni elimino nada. Para actuar agregue --ejecutar "
              f"(mueve a '{a.destino}'); --borrar-definitivamente elimina sin posibilidad de recuperar.")
        return COD_OK
    if a.borrar_definitivamente and not a.confirmo:
        print("ERROR: --borrar-definitivamente exige ademas --confirmo-que-tengo-copia", file=sys.stderr)
        return COD_USO
    n = 0
    for r in sobran:
        origen = os.path.join(a.imagenes, *r.split("/"))
        if a.borrar_definitivamente:
            os.remove(origen)
        else:
            destino = os.path.join(a.destino, *r.split("/"))
            os.makedirs(os.path.dirname(destino), exist_ok=True)
            if os.path.exists(destino):
                print(f"ERROR: ya existe '{destino}': no se sobrescribe", file=sys.stderr)
                return COD_USO
            shutil.move(origen, destino)
        n += 1
    print(f"{'Eliminadas' if a.borrar_definitivamente else 'Movidas a ' + a.destino}: {n}")
    return COD_OK


def main(argv=None):
    ap = argparse.ArgumentParser(prog="curar_dataset.py", add_help=False,
                                 description="Curacion segura del banco de imagenes (ver docstring del archivo)")
    ap.add_argument("-h", "--help", "--ayuda", action="help", help="muestra esta ayuda y sale")
    sub = ap.add_subparsers(dest="comando", required=True)
    s = sub.add_parser("seleccionar", add_help=True, help="evalua y escribe seleccion + datasets en --salida")
    s.add_argument("--imagenes", default=cfg.DIR_IMAGENES)
    s.add_argument("--salida", default=SALIDA_POR_DEFECTO, help="carpeta de resultados (default salida_curacion/)")
    s.add_argument("--sobrescribir", action="store_true", help="permite pisar archivos de --salida")
    e = sub.add_parser("eliminar", add_help=True, help="simula (o ejecuta) el retiro de imagenes no aceptadas")
    e.add_argument("--seleccion", required=True, help="seleccion_curacion.csv generado por 'seleccionar'")
    e.add_argument("--imagenes", default=cfg.DIR_IMAGENES)
    e.add_argument("--destino", default=os.path.join(SALIDA_POR_DEFECTO, "descartadas"))
    e.add_argument("--ejecutar", action="store_true", help="sin esto es solo una simulacion")
    e.add_argument("--borrar-definitivamente", action="store_true", help="elimina en vez de mover")
    e.add_argument("--confirmo-que-tengo-copia", dest="confirmo", action="store_true")
    a = ap.parse_args(argv)
    return cmd_seleccionar(a) if a.comando == "seleccionar" else cmd_eliminar(a)


if __name__ == "__main__":
    sys.exit(main())
