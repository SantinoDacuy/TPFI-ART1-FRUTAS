"""
demo.py
Demostracion del TFI - Red ART1 para clasificacion de frutas en dos etapas.
Grupo 2 - Redes Neuronales - UADER FCyT

Ejecuta, con los archivos que realmente estan en el repositorio:

  A. Corridas de referencia del primer avance sobre los datasets ENTREGADOS
     (datasets/), una pasada y orden original, para varios valores de rho.
  B. La variante principal: entrenamiento sobre datasets_v2/principal/ con los parametros de
     config/parametros_etapas.csv (misma fuente que usan el entrenamiento y la
     regeneracion de modelos; la inferencia usa los guardados en cada modelo).
  C. Una prediccion de ejemplo por foto incluida en ejemplos/.

La demo NO genera ni borra datasets ni imagenes. Si alguna corrida falla o es
infructuosa lo informa y termina con codigo distinto de cero.

Uso:
    python demo.py
    python demo.py --rho-morf 0.7 0.85 --rho-cal 0.7 0.85 0.95
    python demo.py --sin-fotos          (omite la parte C)
    python demo.py --ayuda
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import configuracion as cfg
from validacion import COD_ENTRADA, COD_OK

PYTHON = sys.executable
CARGROSS = os.path.join(ROOT, "CarGross.py")
PROBAR = os.path.join(ROOT, "probar_imagen.py")
CSV_MORF = os.path.join(cfg.DIR_DATASETS_ORIGINAL, "dataset_morfologia.csv")
CSV_CAL = os.path.join(cfg.DIR_DATASETS_ORIGINAL, "dataset_calidad_real.csv")
CSV_MORF_V2 = os.path.join(cfg.DIR_DATASETS_V2, "principal", "morfologia_entrenamiento.csv")
CSV_CAL_V2 = os.path.join(cfg.DIR_DATASETS_V2, "principal", "calidad_entrenamiento.csv")
EJEMPLOS = ("manzana_premium.jpg", "manzana_comercial_defecto.jpg", "manzana_descarte_severo.jpg",
            "naranja.jpg", "auto.jpg")


def titulo(texto, ancho=70):
    print("\n" + "=" * ancho + f"\n  {texto}\n" + "=" * ancho, flush=True)


def subtitulo(texto, ancho=70):
    print("\n" + "-" * ancho + f"\n  {texto}\n" + "-" * ancho, flush=True)


def correr(cmd, descripcion, resultados):
    """Ejecuta un subproceso mostrando su salida; registra el codigo de salida."""
    print(f"\n  $ {' '.join(os.path.relpath(c, ROOT) if os.path.isabs(c) and c != PYTHON else c for c in cmd)}\n", flush=True)
    t0 = time.time()
    sys.stdout.flush()
    rc = subprocess.run(cmd).returncode
    estado = "OK" if rc == 0 else f"FALLO (codigo {rc})"
    print(f"\n  [{estado}] {descripcion} ({time.time() - t0:.1f}s)")
    resultados.append((descripcion, rc))
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(prog="demo.py", add_help=False, description="Demo de punta a punta del TFI")
    ap.add_argument("-h", "--help", "--ayuda", action="help", help="muestra esta ayuda y sale")
    ap.add_argument("--rho-morf", type=float, nargs="+", default=[0.70, 0.85],
                    help="valores de rho para la etapa 1 (parte A)")
    ap.add_argument("--rho-cal", type=float, nargs="+", default=[0.70, 0.85, 0.95],
                    help="valores de rho para la etapa 2 (parte A)")
    ap.add_argument("--solo-corridas", action="store_true",
                    help="(compatibilidad) la demo ya no genera datasets; no tiene efecto")
    ap.add_argument("--sin-fotos", action="store_true", help="omite la parte C (predicciones sobre fotos)")
    args = ap.parse_args(argv)

    titulo("TFI - RED DE CARPENTER-GROSSBERG (ART1)")
    print("""
  Clasificacion de frutas en dos etapas con aprendizaje no supervisado.
    Etapa 1: tipo (citrico / tropical / carozo = manzana)  - variables morfologicas y cromaticas
    Etapa 2: calidad de manzanas (premium / comercial / descarte)
  ART1 NO conoce las etiquetas al aprender; se usan despues para nombrar las categorias.
    """)
    faltan = [r for r in (CARGROSS, CSV_MORF, CSV_CAL) if not os.path.isfile(r)]
    if faltan:
        print("  [ERROR] Faltan archivos del repositorio:\n    " + "\n    ".join(faltan) +
              "\n  Ayuda: python demo.py --help", file=sys.stderr)
        return COD_ENTRADA
    resultados = []

    titulo("A. CORRIDAS DE REFERENCIA - datasets entregados (180 filas c/u), una pasada")
    for rho in args.rho_morf:
        subtitulo(f"Etapa 1 - morfologia, rho = {rho}")
        correr([PYTHON, CARGROSS, "--csv", CSV_MORF, "--rho", str(rho), "--epocas", "1"],
               f"morfologia entregada rho={rho}", resultados)
    for rho in args.rho_cal:
        subtitulo(f"Etapa 2 - calidad, rho = {rho}")
        correr([PYTHON, CARGROSS, "--csv", CSV_CAL, "--rho", str(rho), "--epocas", "1"],
               f"calidad entregada rho={rho}", resultados)

    titulo("B. VARIANTE PRINCIPAL - datasets_v2/principal con parametros de config/parametros_etapas.csv")
    if not (os.path.isfile(CSV_MORF_V2) and os.path.isfile(CSV_CAL_V2)):
        print("  [ERROR] Faltan datasets_v2/principal/. Se generan con:\n"
              "    python scripts/extraer_mediciones.py && python scripts/busqueda.py buscar && "
              "python scripts/comparar_ronda2.py && python scripts/final_ronda2.py", file=sys.stderr)
        return COD_ENTRADA
    for etapa, ruta in (("morfologia", CSV_MORF_V2), ("calidad", CSV_CAL_V2)):
        p = cfg.parametros_etapa(etapa)
        subtitulo(f"{etapa}: rho={p['rho']}, hasta {p['epocas']} pasadas, desempate '{p['desempate']}'")
        correr([PYTHON, CARGROSS, "--csv", ruta], f"{etapa} variante principal (config)", resultados)

    if not args.sin_fotos:
        titulo("C. PREDICCION SOBRE FOTOS DE EJEMPLO (inferencia: los pesos no cambian)")
        for nombre in EJEMPLOS:
            ruta = os.path.join(ROOT, "ejemplos", nombre)
            if not os.path.isfile(ruta):
                print(f"  [AVISO] falta ejemplos/{nombre}; se omite")
                continue
            subtitulo(nombre)
            correr([PYTHON, PROBAR, "--imagen", ruta, "--sin-grafico"], f"foto {nombre}", resultados)

    titulo("RESUMEN")
    malos = [(d, rc) for d, rc in resultados if rc != 0]
    for d, rc in resultados:
        print(f"  {'OK   ' if rc == 0 else 'FALLO'} {d}")
    if malos:
        print(f"\n  {len(malos)} de {len(resultados)} pasos fallaron o fueron infructuosos (ver arriba).")
        return 1
    print(f"\n  {len(resultados)} pasos completados sin errores.")
    print("""
  Comparacion original vs mejorada, con parametros y datos: resultados/comparacion.md
  Baseline reproducible del primer avance:                  resultados/baseline_original.md
  Foto propia:  python probar_imagen.py --imagen ruta/a/foto.jpg
    """)
    return COD_OK


if __name__ == "__main__":
    sys.exit(main())
