"""
generar_dataset_morfologia.py
Recorre carpetas locales de imagenes (formato Fruits-360: una carpeta por
clase) y arma datasets/dataset_morfologia.csv usando preprocesador.py.

USO:
  1. Bajar/clonar Fruits-360 localmente, por ejemplo:
       git clone --filter=blob:none --sparse --depth 1 \
         https://github.com/Horea94/Fruit-Images-Dataset.git fruits-repo
       cd fruits-repo && git sparse-checkout set "Training/Orange" "Training/Banana" "Training/Apple Red 1" "Training/Apple Granny Smith"
  2. Ajustar la variable PLAN de abajo con las rutas reales en tu maquina.
  3. Correr: python scripts/generar_dataset_morfologia.py

No requiere conexion a internet mas alla del paso 1 (ese es un paso manual,
una sola vez).
"""
import os
import sys
import csv
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocesador import procesar_imagen  # noqa: E402

SEMILLA = 7
SALIDA = os.path.join(os.path.dirname(__file__), "..", "datasets", "dataset_morfologia.csv")

# (carpeta_local, tipo_real, cantidad_a_tomar, prefijo_id)
PLAN = [
    ("fruits-repo/Training/Orange", "citrico", 45, "FC"),
    ("fruits-repo/Training/Banana", "tropical", 45, "FT"),
    ("fruits-repo/Training/Apple Red 1", "carozo", 30, "FM"),
    ("fruits-repo/Training/Apple Granny Smith", "carozo", 30, "FM"),
]

COLUMNAS = ["fruta_id", "es_esferica", "cascara_rugosa", "tiene_tallo",
            "densidad_alta", "pigmentacion_oscura", "color_uniforme", "tipo_real"]


def main():
    random.seed(SEMILLA)
    filas, errores = [], []
    idx = 1
    for carpeta, tipo, n, prefijo in PLAN:
        if not os.path.isdir(carpeta):
            print(f"[AVISO] No existe la carpeta '{carpeta}', se salta.")
            continue
        archivos = sorted(os.listdir(carpeta))
        elegidos = random.sample(archivos, min(n, len(archivos)))
        for a in elegidos:
            fruta_id = f"{prefijo}{idx:03d}"
            idx += 1
            ruta = os.path.join(carpeta, a)
            try:
                fila = procesar_imagen(ruta, fruta_id)
                fila["tipo_real"] = tipo
                filas.append(fila)
            except Exception as e:
                # manejo de corridas infructuosas: no corta el proceso entero
                errores.append((ruta, str(e)))

    random.shuffle(filas)
    with open(SALIDA, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        w.writeheader()
        w.writerows(filas)

    print(f"OK: {len(filas)} filas escritas en {SALIDA}")
    if errores:
        print(f"AVISO: {len(errores)} imagenes fallaron y se omitieron:")
        for ruta, err in errores[:10]:
            print(f"  - {ruta}: {err}")


if __name__ == "__main__":
    main()
