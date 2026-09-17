"""
demo.py
Script de demostracion del TFI — Red ART1 para clasificacion de frutas.
Grupo 2 — Redes Neuronales — UADER FCyT

Ejecuta el pipeline completo en dos etapas:

  ETAPA 1: Clasificacion morfologica (tipo de fruta)
    Corre CarGross.py sobre dataset_morfologia.csv con rho=0.70 y rho=0.85.
    Muestra cuantos clusters se forman y la pureza por cluster.

  ETAPA 2: Clasificacion de calidad (manzanas)
    Si dataset_calidad_real.csv no existe, lo genera automaticamente
    desde las imagenes reales usando preprocesador.py.
    Corre CarGross.py con rho=0.70, rho=0.85 y rho=0.95.
    Muestra clusters y pureza contra etiquetas premium/comercial/descarte.

Uso:
    python demo.py
    python demo.py --solo-corridas      (salta generacion de dataset)
    python demo.py --n-calidad 40       (mas imagenes por clase, default 30)

Requisitos:
    pip install -r requirements.txt
"""

import argparse
import os
import subprocess
import sys
import time

# ── Rutas ─────────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
PYTHON      = sys.executable
CARGROSS    = os.path.join(ROOT, "CarGross.py")
EXTRACTOR   = os.path.join(ROOT, "scripts", "extractor_calidad.py")
CSV_MORF    = os.path.join(ROOT, "datasets", "dataset_morfologia.csv")
CSV_CAL     = os.path.join(ROOT, "datasets", "dataset_calidad_real.csv")


def titulo(texto, ancho=62):
    print("\n" + "=" * ancho)
    print(f"  {texto}")
    print("=" * ancho)


def subtitulo(texto, ancho=62):
    print("\n" + "-" * ancho)
    print(f"  {texto}")
    print("-" * ancho)


def correr(cmd, descripcion):
    """Ejecuta un subproceso y muestra su salida en tiempo real."""
    print(f"\n  $ {' '.join(cmd)}\n")
    t0 = time.time()
    resultado = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - t0
    if resultado.returncode != 0:
        print(f"\n  [ERROR] El comando termino con codigo {resultado.returncode}")
        sys.exit(resultado.returncode)
    print(f"\n  [OK] {descripcion} ({elapsed:.1f}s)")


def verificar_archivo(ruta, nombre):
    if not os.path.isfile(ruta):
        print(f"  [ERROR] No se encontro '{nombre}' en: {ruta}")
        print("          Verificar que los datasets esten en la carpeta 'datasets/'.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Demo completo del TFI — ART1 para clasificacion de frutas"
    )
    parser.add_argument(
        "--solo-corridas", action="store_true",
        help="Salta la generacion del dataset de calidad (usa el existente)"
    )
    parser.add_argument(
        "--n-calidad", type=int, default=30,
        help="Imagenes por clase para el dataset de calidad (default 30)"
    )
    args = parser.parse_args()

    titulo("TFI — RED DE CARPENTER-GROSSBERG (ART1)")
    print("""
  Pipeline de clasificacion automatizada de frutas mediante
  redes neuronales ART1 (aprendizaje no supervisado).

  Implementacion: CarGross.py + preprocesador.py
  Datasets      : dataset_morfologia.csv (150 frutas)
                  dataset_calidad_real.csv (90 manzanas reales)
  Caracteristicas binarias: es_esferica, cascara_rugosa,
    tiene_tallo, densidad_alta, pigmentacion_oscura, color_uniforme
    """)

    # ── Verificaciones previas ────────────────────────────────────────────────
    titulo("VERIFICACION DE ARCHIVOS")
    verificar_archivo(CARGROSS,  "CarGross.py")
    verificar_archivo(CSV_MORF,  "dataset_morfologia.csv")
    print("  [OK] CarGross.py encontrado")
    print("  [OK] dataset_morfologia.csv encontrado")

    # ── Generar dataset de calidad si hace falta ──────────────────────────────
    if not args.solo_corridas:
        titulo("PASO PREVIO — EXTRACCION DE FEATURES DE CALIDAD")
        if os.path.isfile(CSV_CAL):
            print(f"  dataset_calidad_real.csv ya existe. Se usa el existente.")
            print(f"  (Para regenerarlo, eliminalo y volver a correr demo.py)")
        else:
            print("  Generando dataset desde imagenes reales de manzanas...")
            print("  Esto puede tardar 30-60 segundos.\n")
            correr(
                [PYTHON, EXTRACTOR, "--n", str(args.n_calidad), "--rng-seed", "42"],
                "Dataset de calidad generado"
            )

    verificar_archivo(CSV_CAL, "dataset_calidad_real.csv")
    print("  [OK] dataset_calidad_real.csv encontrado")

    # ── ETAPA 1: Morfologia ───────────────────────────────────────────────────
    titulo("ETAPA 1 — CLASIFICACION MORFOLOGICA (tipo de fruta)")
    print("""
  Dataset: dataset_morfologia.csv
  150 frutas de 3 tipos: citrico / tropical / carozo
  La red NO conoce las etiquetas. Solo ve los 6 bits binarios.
  Evaluamos si los clusters que forma coinciden con los tipos reales.
    """)

    subtitulo("Corrida 1a — rho = 0.70 (vigilancia baja, clusters amplios)")
    correr([PYTHON, CARGROSS, "--csv", CSV_MORF, "--rho", "0.70"],
           "Corrida morfologia rho=0.70")

    subtitulo("Corrida 1b — rho = 0.85 (vigilancia alta, clusters finos)")
    correr([PYTHON, CARGROSS, "--csv", CSV_MORF, "--rho", "0.85"],
           "Corrida morfologia rho=0.85")

    # ── ETAPA 2: Calidad ──────────────────────────────────────────────────────
    titulo("ETAPA 2 — CLASIFICACION DE CALIDAD (manzanas)")
    print("""
  Dataset: dataset_calidad_real.csv
  90 manzanas reales: premium / comercial / descarte
  Features extraidos con preprocesador.py (OpenCV + Otsu).
  Pipeline: imagen -> segmentacion -> centrado -> features -> binario.
    """)

    subtitulo("Corrida 2a — rho = 0.70")
    correr([PYTHON, CARGROSS, "--csv", CSV_CAL, "--rho", "0.70"],
           "Corrida calidad rho=0.70")

    subtitulo("Corrida 2b — rho = 0.85")
    correr([PYTHON, CARGROSS, "--csv", CSV_CAL, "--rho", "0.85"],
           "Corrida calidad rho=0.85")

    subtitulo("Corrida 2c — rho = 0.95 (maximo detalle)")
    correr([PYTHON, CARGROSS, "--csv", CSV_CAL, "--rho", "0.95"],
           "Corrida calidad rho=0.95")

    # ── Cierre ────────────────────────────────────────────────────────────────
    titulo("DEMO COMPLETADO")
    print("""
  Todos los pasos ejecutaron sin errores.

  Archivos generados / utilizados:
    datasets/dataset_morfologia.csv     <- Dataset 1 (morfologia)
    datasets/dataset_calidad_real.csv   <- Dataset 2 (calidad, imagenes reales)

  Para probar una fotografia individual en vivo (defensa oral):
    python probar_imagen.py --imagen "ruta/a/foto.jpg"

  Para reproducir solo las corridas sin regenerar el dataset:
    python demo.py --solo-corridas

  Para explorar con otros valores de rho:
    python CarGross.py --csv datasets/dataset_morfologia.csv --rho 0.9
    python CarGross.py --csv datasets/dataset_calidad_real.csv --rho 0.6
    """)


if __name__ == "__main__":
    main()
