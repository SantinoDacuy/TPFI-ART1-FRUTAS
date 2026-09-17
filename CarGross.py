"""
CarGross.py
Red de Carpenter-Grossberg (ART1) para clasificacion no supervisada de
frutas. TFI Redes Neuronales - Grupo 2 - UADER FCyT.

Formulacion clasica de ART1 con aprendizaje rapido (L=2), siguiendo la
notacion estandar de Carpenter & Grossberg (1987):
  - Capa F1 (comparacion) <-> Capa F2 (reconocimiento/categorias)
  - Pesos top-down t_j (vector binario por categoria)
  - Pesos bottom-up b_j, derivados de t_j: b_ij = L*t_ji / (L - 1 + |t_j|)
  - Competencia: gana el nodo con mayor entrada neta bottom-up
  - Test de vigilancia: |I ^ t_j| / |I| >= rho ?
      si pasa  -> resonancia: t_j se actualiza como AND(I, t_j)
      si no pasa -> reset de ese nodo, se prueba el siguiente candidato
  - Si ningun nodo existente resuena -> se compromete una categoria nueva

NOTA: implementado con la formulacion estandar de la bibliografia (Fausett /
Carpenter-Grossberg). Si suben Lau.pp5.a.11 / Lau.pp12.a.14 con una notacion
distinta, se puede ajustar sin cambiar la logica de fondo.

USO:
    python CarGross.py --csv datasets/dataset_morfologia.csv --rho 0.8
    python CarGross.py --csv datasets/dataset_calidad.csv --rho 0.7
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter


class ProliferacionError(Exception):
    """Se levanta cuando la red no logra estabilizar sus categorias.
    Es el mecanismo de 'corrida infructuosa' que pide la consigna: no
    cuelga el programa, corta con un mensaje explicando la causa probable."""
    pass


class RedART1:
    """
    Parametros
    ----------
    n_features : int
        Cantidad de variables binarias de entrada.
    rho : float
        Parametro de vigilancia, en (0, 1]. Mas alto = clusters mas finos.
    L : float
        Parametro de la regla de aprendizaje rapido (L > 1). Default 2.0.
    max_categorias : int
        Tope de categorias antes de declarar la corrida infructuosa por
        proliferacion (ver seccion de metricas del informe: si con 3 tipos
        reales aparecen 15+ clusters, el umbral esta mal calibrado).
    """

    def __init__(self, n_features, rho, L=2.0, max_categorias=50):
        if not (0 < rho <= 1):
            raise ValueError(f"rho debe estar en (0, 1], se recibio {rho}")
        if L <= 1:
            raise ValueError(f"L debe ser > 1, se recibio {L}")
        self.n = n_features
        self.rho = rho
        self.L = L
        self.max_categorias = max_categorias
        self.top_down = []  # un vector binario (lista de 0/1) por categoria comprometida

    def _bottom_up(self, t_j):
        norma = sum(t_j)
        denom = self.L - 1 + norma
        return [(self.L * b) / denom for b in t_j]

    @staticmethod
    def _interseccion(entrada, t_j):
        return [int(a and b) for a, b in zip(entrada, t_j)]

    def evaluar(self, entrada):
        """Evalua un patron contra todos los clusters comprometidos sin modificar pesos.
        Devuelve el cluster ganador, el ratio de similitud |I ^ T_j| / |I|,
        si hubo resonancia (ratio >= rho), y detalles de candidatos.
        """
        if len(entrada) != self.n:
            raise ValueError(f"Se esperaban {self.n} variables, llegaron {len(entrada)}")
        if any(v not in (0, 1) for v in entrada):
            raise ValueError(f"La entrada debe ser binaria (0/1): {entrada}")
        norma_entrada = sum(entrada)
        if norma_entrada == 0:
            cat_nula = [0] * self.n
            for j, t in enumerate(self.top_down):
                if t == cat_nula:
                    return {"ganador": j, "ratio": 1.0, "resuena": True, "candidatos": []}
            return {"ganador": None, "ratio": 0.0, "resuena": False, "motivo": "vector nulo no aprendido"}

        if not self.top_down:
            return {"ganador": None, "ratio": 0.0, "resuena": False, "motivo": "sin categorias"}

        candidatos = list(range(len(self.top_down)))
        detalles = []
        while candidatos:
            netos = [(j, sum(a * b for a, b in zip(entrada, self._bottom_up(self.top_down[j]))))
                     for j in candidatos]
            ganador, neto = max(netos, key=lambda par: par[1])
            interseccion = self._interseccion(entrada, self.top_down[ganador])
            ratio = sum(interseccion) / norma_entrada
            resuena = ratio >= self.rho
            detalles.append({
                "cluster": ganador, "neto": neto, "ratio": ratio, "resuena": resuena
            })
            if resuena:
                return {
                    "ganador": ganador, "ratio": ratio, "resuena": True,
                    "candidatos": detalles
                }
            candidatos.remove(ganador)

        mejor = max(detalles, key=lambda d: d["ratio"]) if detalles else None
        return {
            "ganador": None,
            "ratio": mejor["ratio"] if mejor else 0.0,
            "resuena": False,
            "mejor_candidato": mejor["cluster"] if mejor else None,
            "candidatos": detalles
        }

    def clasificar(self, entrada, aprender=True):
        """Presenta un patron binario a la red.
        Si aprender=True: actualiza el prototipo resonante o compromete una nueva categoria.
        Si aprender=False (modo inferencia): devuelve el cluster resonante, o None si es NOVEDAD.
        """
        if len(entrada) != self.n:
            raise ValueError(f"Se esperaban {self.n} variables, llegaron {len(entrada)}")
        if any(v not in (0, 1) for v in entrada):
            raise ValueError(f"La entrada debe ser binaria (0/1): {entrada}")
        norma_entrada = sum(entrada)
        if norma_entrada == 0:
            # Caso limite en ART1: patron sin caracteristicas activas (|I| = 0).
            # Si existe una categoria nula comprometida, resuena con ella; si no, se compromete.
            cat_nula = [0] * self.n
            for j, t in enumerate(self.top_down):
                if t == cat_nula:
                    return j
            if not aprender:
                return None
            if len(self.top_down) >= self.max_categorias:
                raise ProliferacionError("Limite de categorias alcanzado al registrar categoria para patron nulo.")
            self.top_down.append(cat_nula)
            return len(self.top_down) - 1

        candidatos = list(range(len(self.top_down)))

        while candidatos:
            netos = [(j, sum(a * b for a, b in zip(entrada, self._bottom_up(self.top_down[j]))))
                     for j in candidatos]
            ganador, _ = max(netos, key=lambda par: par[1])

            interseccion = self._interseccion(entrada, self.top_down[ganador])
            ratio = sum(interseccion) / norma_entrada
            if ratio >= self.rho:
                if aprender:
                    self.top_down[ganador] = interseccion  # aprendizaje: AND logico
                return ganador
            candidatos.remove(ganador)  # reset: esta categoria no sirve, probar otra

        if not aprender:
            # Modo inferencia: ningun cluster paso la vigilancia -> es NOVEDAD
            return None

        if len(self.top_down) >= self.max_categorias:
            raise ProliferacionError(
                f"Se alcanzo el limite de {self.max_categorias} categorias sin resonancia. "
                f"Esto indica proliferacion descontrolada: rho={self.rho} es demasiado alto "
                f"para el ruido de los datos. Sugerencia: bajar rho."
            )
        self.top_down.append(list(entrada))
        return len(self.top_down) - 1

    def guardar(self, ruta_json, metadata_extra=None):
        """Guarda la red y sus categorias aprendidas en un archivo JSON."""
        datos = {
            "n_features": self.n,
            "rho": self.rho,
            "L": self.L,
            "max_categorias": self.max_categorias,
            "top_down": self.top_down,
            "metadata_extra": metadata_extra or {},
        }
        os.makedirs(os.path.dirname(os.path.abspath(ruta_json)), exist_ok=True)
        with open(ruta_json, "w", encoding="utf-8") as f:
            json.dump(datos, f, indent=2)

    @classmethod
    def cargar(cls, ruta_json):
        """Carga una red ART1 desde un archivo JSON."""
        with open(ruta_json, "r", encoding="utf-8") as f:
            datos = json.load(f)
        red = cls(
            n_features=datos["n_features"],
            rho=datos["rho"],
            L=datos.get("L", 2.0),
            max_categorias=datos.get("max_categorias", 50),
        )
        red.top_down = datos["top_down"]
        return red, datos.get("metadata_extra", {})


def leer_csv(ruta, columna_id="fruta_id"):
    """Lee el CSV y separa: id, columnas binarias (features -> entrada de
    la red) y columnas no binarias (metadata, ej. tipo_real / calidad_real
    - se detectan solas y NUNCA se pasan a la red, solo sirven para el
    reporte de pureza)."""
    try:
        with open(ruta, newline="") as f:
            lector = csv.DictReader(f)
            filas = list(lector)
            columnas = lector.fieldnames
    except FileNotFoundError:
        raise SystemExit(f"ERROR: no se encontro el archivo '{ruta}'")
    except csv.Error as e:
        raise SystemExit(f"ERROR: el CSV esta mal formado ({e})")

    if not filas:
        raise SystemExit(f"ERROR: '{ruta}' no tiene filas de datos")
    if columna_id not in columnas:
        raise SystemExit(f"ERROR: falta la columna '{columna_id}' en el CSV")

    cols_features, cols_metadata = [], []
    for col in columnas:
        if col == columna_id:
            continue
        valores = {fila[col] for fila in filas}
        (cols_features if valores <= {"0", "1"} else cols_metadata).append(col)

    if not cols_features:
        raise SystemExit("ERROR: no se detecto ninguna columna binaria (0/1) para usar como entrada")

    ids, entradas, metadata = [], [], []
    for fila in filas:
        ids.append(fila[columna_id])
        entradas.append([int(fila[c]) for c in cols_features])
        metadata.append({c: fila[c] for c in cols_metadata})

    return ids, entradas, metadata, cols_features, cols_metadata


def ejecutar(ruta_csv, rho, L=2.0, max_categorias=50, columna_id="fruta_id"):
    ids, entradas, metadata, cols_features, cols_metadata = leer_csv(ruta_csv, columna_id)
    red = RedART1(n_features=len(cols_features), rho=rho, L=L, max_categorias=max_categorias)

    asignaciones = []
    try:
        for entrada in entradas:
            asignaciones.append(red.clasificar(entrada))
    except ProliferacionError as e:
        return {
            "exito": False, "motivo": str(e), "n_categorias": len(red.top_down),
            "n_procesados": len(asignaciones), "n_total": len(entradas), "rho": rho,
        }

    return {
        "exito": True, "n_categorias": len(red.top_down), "ids": ids,
        "asignaciones": asignaciones, "metadata": metadata,
        "cols_metadata": cols_metadata, "cols_features": cols_features, "rho": rho,
    }


def imprimir_reporte(resultado):
    if not resultado["exito"]:
        print(f"\nCORRIDA INFRUCTUOSA (rho={resultado['rho']}): {resultado['motivo']}")
        print(f"Se procesaron {resultado['n_procesados']}/{resultado['n_total']} registros antes de cortar.\n")
        return

    print(f"\nCorrida exitosa - rho={resultado['rho']} - {resultado['n_categorias']} clusters formados\n")
    conteo = Counter(resultado["asignaciones"])
    for cluster in sorted(conteo):
        n = conteo[cluster]
        linea = f"  Cluster {cluster}: {n} frutas"
        if resultado["cols_metadata"]:
            col_ref = resultado["cols_metadata"][0]
            reales = [resultado["metadata"][i][col_ref]
                      for i, c in enumerate(resultado["asignaciones"]) if c == cluster]
            mayoria, n_mayoria = Counter(reales).most_common(1)[0]
            linea += f" | mayoria real: '{mayoria}' ({100 * n_mayoria / n:.0f}% pureza)"
        print(linea)


def entrenar_modelo(ruta_csv, rho, L=2.0, max_categorias=50, columna_id="fruta_id"):
    """Entrena la red ART1 sobre un CSV y genera un mapeo semantico de cada cluster
    hacia la etiqueta real mayoritaria (ej. 'citrico', 'premium')."""
    ids, entradas, metadata, cols_features, cols_metadata = leer_csv(ruta_csv, columna_id)
    red = RedART1(n_features=len(cols_features), rho=rho, L=L, max_categorias=max_categorias)
    asignaciones = []
    for entrada in entradas:
        asignaciones.append(red.clasificar(entrada, aprender=True))

    mapeo_clusters = {}
    conteo = Counter(asignaciones)
    for cluster in sorted(conteo):
        n = conteo[cluster]
        info = {"n": n, "etiqueta": f"Cluster_{cluster}", "pureza": 1.0}
        if cols_metadata:
            col_ref = cols_metadata[0]
            reales = [metadata[i][col_ref] for i, c in enumerate(asignaciones) if c == cluster]
            if reales:
                mayoria, n_mayoria = Counter(reales).most_common(1)[0]
                info["etiqueta"] = mayoria
                info["pureza"] = round(n_mayoria / n, 4)
                info["col_referencia"] = col_ref
        mapeo_clusters[str(cluster)] = info

    metadata_extra = {
        "cols_features": cols_features,
        "cols_metadata": cols_metadata,
        "mapeo_clusters": mapeo_clusters,
        "n_muestras_entrenamiento": len(entradas),
    }
    return red, metadata_extra


def main():
    parser = argparse.ArgumentParser(description="Red ART1 (Carpenter-Grossberg) para clasificacion de frutas")
    parser.add_argument("--csv", required=True, help="ruta al dataset CSV de entrada")
    parser.add_argument("--rho", type=float, required=True, help="parametro de vigilancia (0,1]")
    parser.add_argument("--L", type=float, default=2.0, help="parametro de aprendizaje rapido (default 2.0)")
    parser.add_argument("--max-categorias", type=int, default=50, help="tope antes de declarar corrida infructuosa")
    parser.add_argument("--id-col", default="fruta_id", help="columna identificadora (default fruta_id)")
    parser.add_argument("--guardar-modelo", help="ruta JSON donde guardar el modelo entrenado y sus clusters")
    args = parser.parse_args()

    if args.guardar-modelo if hasattr(args, "guardar-modelo") else args.guardar_modelo:
        ruta_salida = args.guardar_modelo
        try:
            red, meta = entrenar_modelo(args.csv, args.rho, L=args.L,
                                       max_categorias=args.max_categorias, columna_id=args.id_col)
            red.guardar(ruta_salida, metadata_extra=meta)
            print(f"\nModelo entrenado y guardado exitosamente en: {ruta_salida}")
            print(f"Total de clusters comprometidos: {len(red.top_down)}")
            for c_id, info in meta["mapeo_clusters"].items():
                print(f"  Cluster {c_id}: {info['etiqueta']} ({info['n']} frutas, {info['pureza']*100:.0f}% pureza)")
        except Exception as e:
            print(f"ERROR al entrenar/guardar modelo: {e}")
            sys.exit(1)
        return

    try:
        resultado = ejecutar(args.csv, args.rho, L=args.L, max_categorias=args.max_categorias, columna_id=args.id_col)
    except ValueError as e:
        print(f"ERROR de configuracion: {e}")
        sys.exit(1)

    imprimir_reporte(resultado)



if __name__ == "__main__":
    main()
