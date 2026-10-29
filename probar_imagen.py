"""
probar_imagen.py
Clasifica UNA fotografia con la cadena de dos etapas ART1.
Grupo 2 - Redes Neuronales - UADER FCyT

  Etapa 1 (morfologia): tipo de fruta (citrico / tropical / carozo = manzana).
  Etapa 2 (calidad)   : solo si la etapa 1 identifica 'carozo' (manzana),
                        referencias premium / comercial / descarte.

COMPORTAMIENTO NORMAL = INFERENCIA: los pesos de la red NO se modifican.
Con --aprender la red tiene plasticidad EN MEMORIA durante la ejecucion
(prototipo resonante -> interseccion AND; ninguna resonancia -> categoria nueva
SIN etiqueta). Los cambios solo se guardan con --guardar-aprendizaje CARPETA y
siempre en archivos nuevos: el modelo original no se sobrescribe.

Uso:
    python probar_imagen.py --imagen ejemplos/naranja.jpg
    python probar_imagen.py --imagen ejemplos/manzana_premium.jpg --legado     # modelos entregados
    python probar_imagen.py --imagen ejemplos/auto.jpg --aprender --guardar-aprendizaje salida_aprendizaje
    python probar_imagen.py --ayuda

Codigos de salida: 0 ok | 1 corrida infructuosa (limite de categorias) | 2 argumentos |
3 imagen/modelo invalido o ausente | 4 error interno.
"""
import argparse
import os
import sys

import cv2
import numpy as np

DIR_RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR_RAIZ)

import configuracion as cfg
import inferencia
from CarGross import ProliferacionError, cargar_modelo
from validacion import (ALIAS_MANZANA, COD_ENTRADA, COD_ERROR, COD_INFRUCTUOSA, COD_OK, COD_USO,
                        ErrorEntrada)

MODELO_MORF_V2 = os.path.join(cfg.DIR_MODELOS_V2, "principal", "art1_morfologia.json")
MODELO_CAL_V2 = os.path.join(cfg.DIR_MODELOS_V2, "principal", "art1_calidad.json")
MODELO_MORF_ORIG = os.path.join(cfg.DIR_MODELOS_ORIGINAL, "art1_morfologia.json")
MODELO_CAL_ORIG = os.path.join(cfg.DIR_MODELOS_ORIGINAL, "art1_calidad.json")


def banner():
    print("\n" + "=" * 72)
    print("   CLASIFICACION DE FRUTAS CON ART1 - INFERENCIA DE DOS ETAPAS")
    print("=" * 72)


def nombre_tipo(etiqueta):
    return f"{etiqueta.upper()} (manzana)" if etiqueta == ALIAS_MANZANA else etiqueta.upper()


def crear_imagen_diagnostico(img_orig, mascara, img_centrada, texto_etapa1, texto_etapa2, ruta_salida):
    """Compone un PNG con los pasos del preprocesamiento y los resultados."""
    alto_canvas, ancho_canvas = 420, 760
    lienzo = np.full((alto_canvas, ancho_canvas, 3), 30, dtype=np.uint8)
    cv2.putText(lienzo, "DIAGNOSTICO DE PREPROCESAMIENTO Y CLASIFICACION ART1",
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(lienzo, (20, 44), (ancho_canvas - 20, 44), (100, 100, 100), 1)

    h_orig, w_orig = img_orig.shape[:2]
    escala = 190.0 / max(h_orig, w_orig)
    nw, nh = max(1, int(w_orig * escala)), max(1, int(h_orig * escala))
    panel_orig = np.full((200, 200, 3), 45, dtype=np.uint8)
    oy, ox = (200 - nh) // 2, (200 - nw) // 2
    panel_orig[oy:oy + nh, ox:ox + nw] = cv2.resize(img_orig, (nw, nh))

    masc200 = cv2.resize(mascara, (200, 200), interpolation=cv2.INTER_NEAREST)
    panel_masc = cv2.cvtColor(masc200, cv2.COLOR_GRAY2BGR)
    contornos, _ = cv2.findContours(masc200, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        cv2.drawContours(panel_masc, contornos, -1, (0, 255, 0), 2)
    panel_centr = cv2.resize(img_centrada, (200, 200))

    for x, panel, etq in zip([30, 275, 520], [panel_orig, panel_masc, panel_centr],
                             ["1. Foto Original", "2. Mascara Segmentada", "3. Centrado y Escalado"]):
        lienzo[65:265, x:x + 200] = panel
        cv2.rectangle(lienzo, (x, 65), (x + 200, 265), (140, 140, 140), 1)
        cv2.putText(lienzo, etq, (x + 10, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.line(lienzo, (20, 305), (ancho_canvas - 20, 305), (80, 80, 80), 1)
    cv2.putText(lienzo, f"ETAPA 1 (Morfologia): {texto_etapa1}", (30, 340),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 255), 1, cv2.LINE_AA)
    if texto_etapa2:
        cv2.putText(lienzo, f"ETAPA 2 (Calidad):     {texto_etapa2}", (30, 375),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 120), 1, cv2.LINE_AA)
    else:
        cv2.putText(lienzo, "ETAPA 2: omitida (solo se deriva si la etapa 1 identifica 'carozo' = manzana)",
                    (30, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(lienzo, "Coincidencia = |I^T|/|I| de ART1; no es probabilidad de acierto", (30, 405),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.imwrite(ruta_salida, lienzo)


def imprimir_etapa(titulo, r, modelo, aprender):
    print("\n" + "-" * 72)
    print(f"  {titulo}")
    print("-" * 72)
    print("  Medidas continuas -> bit (umbral del modelo):")
    pre = inferencia.pre_de(modelo, r["etapa"])
    for col, bit in zip(r["cols_features"], r["vector"]):
        regla = pre["reglas"][col]
        v = r["crudos"][regla["crudo"]]
        print(f"    - {col:<24} {regla['crudo']:<20} = {v:10.4f}  ({regla['op']} {r['umbrales'][col]:.4g}) -> {bit}")
    print(f"  Vector binario F1: {r['vector']}")
    if r["resuena"]:
        etq = r["etiqueta"]
        print(f"\n  >> RESONANCIA con la categoria {r['cluster']} (vigilancia rho={r['rho']}, "
              f"{r['n_categorias']} categorias)")
        if etq:
            texto = nombre_tipo(etq) if r["etapa"] == "morfologia" else etq.upper()
            print(f"     Etiqueta de la categoria : {texto}")
        else:
            print(f"     Etiqueta de la categoria : (ninguna) - estado '{r['estado_categoria']}': "
                  "la categoria no tiene una etiqueta unica, no se traduce a fruta/calidad")
        print(f"     Coincidencia ART1        : {100 * r['coincidencia']:.1f}%  = |I^T|/|I|: parte del patron contenida en el "
              "prototipo. NO es probabilidad de acierto.")
        if r["pureza_entrenamiento"] is not None:
            print(f"     Pureza de entrenamiento  : {100 * r['pureza_entrenamiento']:.0f}% sobre {r['n_entrenamiento']} muestras: mezcla de "
                  "etiquetas de la categoria al entrenar; NO es la confianza de esta foto.")
    else:
        motivo = {"vigilancia": "ningun prototipo supero la vigilancia", "sin_categorias": "la red no tiene categorias",
                  "vector_nulo_no_aprendido": "vector nulo sin categoria nula aprendida",
                  "vector_nulo_rechazado": "vector nulo rechazado por la politica del modelo"}.get(r["motivo"], r["motivo"])
        print(f"\n  >> NOVEDAD: {motivo}.")
        print(f"     Mejor coincidencia: {100 * r['coincidencia']:.1f}% (se exigia rho={r['rho']}). "
              "La red no asigna etiqueta.")
    for c in r["cambios"]:
        if c["evento"] == "categoria_nueva":
            print(f"     [APRENDIZAJE] Se creo la categoria {c['categoria']} SIN etiqueta (no se inventa un tipo/calidad). "
                  "Existe solo en memoria hasta guardar.")
        elif c["evento"] == "prototipo_actualizado":
            print(f"     [APRENDIZAJE] Prototipo de la categoria {c['categoria']} actualizado (AND): "
                  f"{''.join(map(str, c['prototipo_antes']))} -> {''.join(map(str, c['prototipo_despues']))}. "
                  "Sus 'n' y 'pureza' siguen siendo los del entrenamiento original.")
        else:
            print(f"     [APRENDIZAJE] Resonancia con la categoria {c['categoria']}: el prototipo no cambia.")


def _parser():
    p = argparse.ArgumentParser(
        prog="probar_imagen.py", add_help=False,
        description="Clasifica una fotografia con ART1 en dos etapas (tipo -> calidad de manzanas)",
        epilog="Ejemplos:\n  python probar_imagen.py --imagen ejemplos/naranja.jpg\n"
               "  python probar_imagen.py --imagen ejemplos/manzana_premium.jpg --legado\n"
               "  python probar_imagen.py --imagen ejemplos/auto.jpg --aprender --guardar-aprendizaje salida_aprendizaje",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-h", "--help", "--ayuda", action="help", help="muestra esta ayuda y sale")
    p.add_argument("--imagen", required=True, help="ruta a la fotografia")
    p.add_argument("--modelo-morfologia", help=f"modelo de la etapa 1 (default {os.path.relpath(MODELO_MORF_V2, DIR_RAIZ)})")
    p.add_argument("--modelo-calidad", help=f"modelo de la etapa 2 (default {os.path.relpath(MODELO_CAL_V2, DIR_RAIZ)})")
    p.add_argument("--legado", action="store_true",
                   help="usa los modelos entregados en modelos/ (formato 1) con el preprocesamiento del primer avance")
    p.add_argument("--aprender", action="store_true",
                   help="activa plasticidad EN MEMORIA (actualiza/crea categorias); sin esto los pesos no cambian")
    p.add_argument("--guardar-aprendizaje", metavar="CARPETA",
                   help="con --aprender: guarda los modelos modificados como archivos NUEVOS en CARPETA")
    p.add_argument("--salida-grafico", default="diagnostico_clasificacion.png",
                   help="PNG de diagnostico (default diagnostico_clasificacion.png)")
    p.add_argument("--sin-grafico", action="store_true", help="no genera el PNG")
    p.add_argument("--depurar", action="store_true", help="muestra el traceback de errores internos")
    return p


def _fallo(mensaje, codigo):
    print(f"\n[ERROR] {mensaje}", file=sys.stderr)
    print("\nAyuda: python probar_imagen.py --help   (codigos de salida: 1 infructuosa, 2 argumentos, "
          "3 entrada invalida, 4 error interno)", file=sys.stderr)
    return codigo


def main(argv=None):
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass
    args = _parser().parse_args(argv)
    if args.guardar_aprendizaje and not args.aprender:
        return _fallo("--guardar-aprendizaje requiere --aprender (sin aprendizaje no hay nada que guardar)", COD_USO)
    try:
        banner()
        if not os.path.isfile(args.imagen):
            raise ErrorEntrada("no se encontro el archivo de imagen", archivo=args.imagen)
        ruta_m = args.modelo_morfologia or (MODELO_MORF_ORIG if args.legado else MODELO_MORF_V2)
        ruta_c = args.modelo_calidad or (MODELO_CAL_ORIG if args.legado else MODELO_CAL_V2)
        for ruta in (ruta_m, ruta_c):
            if not os.path.isfile(ruta):
                raise ErrorEntrada(
                    "modelo ausente. Los modelos de la variante principal se generan con 'python scripts/final_ronda2.py' "
                    "(despues de scripts/busqueda.py y scripts/comparar_ronda2.py); para usar los modelos entregados "
                    "agregue --legado. No se entrena nada en silencio.", archivo=ruta)
        m_morf, m_cal = cargar_modelo(ruta_m), cargar_modelo(ruta_c)
        img = cv2.imread(args.imagen)
        if img is None:
            raise ErrorEntrada("OpenCV no pudo decodificar la imagen (formato no soportado o archivo danado)",
                               archivo=args.imagen)
        print(f"\nImagen : {args.imagen}  ({img.shape[1]}x{img.shape[0]} px)")
        for nombre, m in (("Etapa 1", m_morf), ("Etapa 2", m_cal)):
            print(f"{nombre}: {os.path.relpath(m.ruta, DIR_RAIZ)} - formato {m.formato}, rho={m.red.rho}, "
                  f"{len(m.red.top_down)} categorias, politica de vector nulo '{m.red.politica_nulo}'")
        modo = "APRENDIZAJE ACTIVADO (los pesos pueden cambiar en memoria)" if args.aprender else \
            "inferencia (los pesos NO se modifican)"
        print(f"Modo   : {modo}")

        r = inferencia.inferir_imagen(img, m_morf, m_cal, aprender=args.aprender, nombre_imagen=args.imagen)
        e1, e2 = r["etapa1"], r["etapa2"]
        imprimir_etapa("ETAPA 1: CLASIFICACION MORFOLOGICA (tipo de fruta)", e1, m_morf, args.aprender)
        if e2 is not None:
            imprimir_etapa("ETAPA 2: CALIDAD DE LA MANZANA (derivada desde la etapa 1)", e2, m_cal, args.aprender)
        else:
            print("\n" + "-" * 72)
            if e1["etiqueta"] is None:
                print("  ETAPA 2: OMITIDA - la etapa 1 no identifico un tipo (novedad o categoria sin etiqueta).")
            else:
                print(f"  ETAPA 2: OMITIDA - la etapa 1 identifico '{e1['etiqueta']}'; solo se deriva 'carozo' (manzana).")
            print("-" * 72)
        print("\n  Advertencias:")
        for a in r["advertencias"] + [inferencia.AVISO_ALCANCE]:
            print(f"   * {a}")

        if args.aprender:
            print("\n  Estado del aprendizaje:")
            reales = [c for c in r["cambios"] if c["evento"] != "sin_cambio"]
            if not reales:
                print("   - Ningun peso cambio en esta ejecucion: no hay nada que guardar.")
            elif args.guardar_aprendizaje:
                for etapa, modelo, ev in (("morfologia", m_morf, e1), ("calidad", m_cal, e2)):
                    if ev is not None and any(c["evento"] != "sin_cambio" for c in ev["cambios"]):
                        destino = inferencia.guardar_aprendizaje(modelo, etapa, args.guardar_aprendizaje)
                        print(f"   - Guardado (archivo nuevo): {destino}")
            else:
                print("   - Los cambios NO se guardaron: existieron solo en memoria durante esta ejecucion. "
                      "Use --guardar-aprendizaje CARPETA para conservarlos (el modelo original no se sobrescribe).")
        else:
            print(f"\n  Pesos sin cambios (huella etapa 1: {m_morf.red.huella()}, etapa 2: {m_cal.red.huella()}).")

        if not args.sin_grafico:
            med = r["medicion"]
            t1 = (f"categoria {e1['cluster']} -> {(e1['etiqueta'] or 'sin etiqueta').upper()} (coincidencia {100 * e1['coincidencia']:.0f}%)"
                  if e1["resuena"] else f"NOVEDAD (mejor coincidencia {100 * e1['coincidencia']:.0f}%)")
            t2 = None
            if e2 is not None:
                t2 = (f"categoria {e2['cluster']} -> {(e2['etiqueta'] or 'sin etiqueta').upper()} (coincidencia {100 * e2['coincidencia']:.0f}%)"
                      if e2["resuena"] else f"NOVEDAD (mejor coincidencia {100 * e2['coincidencia']:.0f}%)")
            crear_imagen_diagnostico(img, med.mascara, med.img_c, t1, t2, args.salida_grafico)
            print(f"\n  [OK] Imagen de diagnostico guardada en: {args.salida_grafico}")
        print("=" * 72 + "\n")
        return COD_OK
    except ErrorEntrada as e:
        return _fallo(str(e), COD_ENTRADA)
    except ProliferacionError as e:
        return _fallo(f"corrida infructuosa: {e}", COD_INFRUCTUOSA)
    except FileExistsError as e:
        return _fallo(str(e), COD_USO)
    except Exception as e:  # noqa: BLE001
        if args.depurar:
            raise
        return _fallo(f"error interno inesperado ({type(e).__name__}: {e}). Repita con --depurar", COD_ERROR)


if __name__ == "__main__":
    sys.exit(main())
