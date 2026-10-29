"""
evaluacion/banco.py
Inventario del banco fotografico, grupos de fotos probablemente repetidas y
particiones entrenamiento / seleccion de parametros / prueba.

HECHOS DEL BANCO (verificados en scripts/baseline_original.py):
  - Hay 339 fotos: 219 manzanas (Apple_Good 99, Apple 60 "mixed", Apple_Bad 60),
    60 naranjas (Orange_Good) y 60 bananas (Banana_Good). Son las que quedaron
    despues de la curacion del primer avance (el curador borro el resto).
  - NO existe un identificador de fruta fisica ni de sesion de captura.
  - Lo unico disponible es el NOMBRE DE ARCHIVO: IMGAAAAMMDDhhmmss.jpg y
    AAAAMMDD_hhmmss.jpg traen fecha y hora del telefono; IMG_nnnn.JPG trae un
    contador de camara. El sufijo _01/_02 indica copias editadas de la misma toma.

GRUPO (proxy de "misma fruta / misma rafaga"): fotos de la misma carpeta cuya
hora de toma difiere en <= GAP_RAFAGA_S segundos (o cuyo contador de camara
difiere en <= GAP_CONTADOR). Con 3 s quedan 166 grupos en las 339 fotos; con 10 s
quedarian solo 4 grupos de manzanas "mixed", imposibles de repartir en tres
particiones. Es un compromiso: cuanto menor el intervalo, mas fuga posible. Es una HEURISTICA: reduce la fuga entre
particiones (varias fotos de una misma fruta en entrenamiento y prueba) pero
no demuestra independencia: dos frutas distintas fotografiadas en la misma
sesion (misma luz y fondo) pueden quedar en particiones distintas.
"""
import csv
import os
import random
import re
from collections import defaultdict
from datetime import datetime

GAP_RAFAGA_S = 3
GAP_CONTADOR = 1

# (subcarpeta relativa a la raiz de imagenes, tipo_real, calidad_real, estrato)
# 'carozo' es el alias historico de manzana. La calidad solo existe para manzanas.
FUENTES = (
    ("Good Quality_Fruits/Apple_Good", "carozo", "premium", "manzana_good"),
    ("Mixed Qualit_Fruits/Apple", "carozo", "comercial", "manzana_mixed"),
    ("Bad Quality_Fruits/Apple_Bad", "carozo", "descarte", "manzana_bad"),
    ("Good Quality_Fruits/Orange_Good", "citrico", "", "naranja"),
    ("Good Quality_Fruits/Banana_Good", "tropical", "", "banana"),
)
PARTICIONES = ("entrenamiento", "validacion", "prueba")
COLUMNAS_MANIFIESTO = ("foto_id", "ruta", "estrato", "tipo_real", "calidad_real",
                       "marca_tiempo", "grupo", "particion", "id_orig_morfologia",
                       "id_orig_calidad")

_RE_TS_A = re.compile(r"^IMG(\d{8})(\d{6})")      # IMG20200728175856
_RE_TS_B = re.compile(r"^(\d{8})_(\d{6})")        # 20190809_115439
_RE_CONTADOR = re.compile(r"^IMG_(\d+)", re.I)     # IMG_8536


def _clave_tiempo(nombre):
    """('t', segundos) | ('c', contador) | None segun el nombre de archivo."""
    for rx in (_RE_TS_A, _RE_TS_B):
        m = rx.match(nombre)
        if m:
            dt = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            return ("t", int(dt.timestamp()))
    m = _RE_CONTADOR.match(nombre)
    if m:
        return ("c", int(m.group(1)))
    return None


def inventariar(raiz_imagenes):
    """Lista de dicts (uno por foto) con foto_id estable (orden de FUENTES y
    nombre de archivo), etiquetas de referencia y grupo. No lee pixeles."""
    filas = []
    for sub, tipo, calidad, estrato in FUENTES:
        carpeta = os.path.join(raiz_imagenes, *sub.split("/"))
        if not os.path.isdir(carpeta):
            raise FileNotFoundError(f"no existe la carpeta de imagenes '{carpeta}'")
        for nombre in sorted(os.listdir(carpeta)):
            if nombre.lower().endswith((".jpg", ".jpeg", ".png")):
                filas.append({"ruta": sub + "/" + nombre, "estrato": estrato,
                              "tipo_real": tipo, "calidad_real": calidad,
                              "_clave": _clave_tiempo(nombre)})
    for i, f in enumerate(filas, 1):
        f["foto_id"] = f"F{i:03d}"
    asignar_grupos(filas)
    return filas


def asignar_grupos(filas, gap_s=GAP_RAFAGA_S, gap_contador=GAP_CONTADOR):
    """Agrega 'grupo' (G001...) y 'marca_tiempo' a cada fila. Sin hora ni
    contador en el nombre, la foto forma su propio grupo."""
    por_estrato = defaultdict(list)
    for f in filas:
        por_estrato[f["estrato"]].append(f)
    n = 0
    for estrato, lista in por_estrato.items():
        con_clave = sorted((f for f in lista if f["_clave"]), key=lambda f: f["_clave"])
        sin_clave = [f for f in lista if not f["_clave"]]
        previo = None
        for f in con_clave:
            tipo, valor = f["_clave"]
            tope = gap_s if tipo == "t" else gap_contador
            if previo is None or previo[0] != tipo or valor - previo[1] > tope:
                n += 1
            f["grupo"] = f"G{n:03d}"
            previo = (tipo, valor)
            f["marca_tiempo"] = (datetime.fromtimestamp(valor).strftime("%Y-%m-%dT%H:%M:%S")
                                 if tipo == "t" else f"contador:{valor}")
        for f in sin_clave:
            n += 1
            f["grupo"] = f"G{n:03d}"
            f["marca_tiempo"] = ""
    return filas


def particionar(filas, semilla, proporciones=(0.6, 0.2, 0.2)):
    """Asigna 'particion' por GRUPO (un grupo entero va a una sola particion),
    estratificado por estrato: en cada estrato los grupos se reparten, en orden
    aleatorio con semilla, hacia la particion mas atrasada respecto de su cuota
    de fotos. Los grupos se procesan de mayor a menor (los grandes primero, con un
    pequeno desempate aleatorio) porque con pocos grupos grandes el reparto 60/20/20
    solo es aproximado: el tamano real de cada particion se informa en los
    resultados. Determinista dada la semilla."""
    rng = random.Random(semilla)
    por_estrato = defaultdict(lambda: defaultdict(list))
    for f in filas:
        por_estrato[f["estrato"]][f["grupo"]].append(f)
    for estrato in sorted(por_estrato):
        grupos = sorted(por_estrato[estrato].items())
        rng.shuffle(grupos)
        grupos.sort(key=lambda kv: -len(kv[1]))  # estable: empates en orden aleatorio
        total = sum(len(g) for _, g in grupos)
        cuota = [p * total for p in proporciones]
        cargado = [0, 0, 0]
        for _, fotos in grupos:
            # particion con mayor deficit relativo
            k = max(range(3), key=lambda i: (cuota[i] - cargado[i]) / max(cuota[i], 1e-9)
                    + rng.uniform(0, 0.3))
            for f in fotos:
                f["particion"] = PARTICIONES[k]
            cargado[k] += len(fotos)
    return filas


def escribir_manifiesto(filas, ruta):
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS_MANIFIESTO, extrasaction="ignore")
        w.writeheader()
        for fila in filas:
            w.writerow({c: fila.get(c, "") for c in COLUMNAS_MANIFIESTO})


def leer_manifiesto(ruta):
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        filas = list(csv.DictReader(f))
    comunes = [c for c in COLUMNAS_MANIFIESTO if c != "particion"]       # la ronda 1 trae 'particion'; la 2, 'bloque'
    faltan = [c for c in comunes if filas and c not in filas[0]]
    if filas and "particion" not in filas[0] and "bloque" not in filas[0]:
        faltan.append("particion|bloque")
    if faltan:
        raise ValueError(f"manifiesto '{ruta}' sin columnas {faltan}")
    return filas


# ---------------------------------------------------------------------------
# Protocolo con PARTICION DE INFORME reservada (segunda ronda)
# ---------------------------------------------------------------------------
# Se reserva una fraccion de los GRUPOS (estratificada por origen) que no interviene en ninguna decision:
# elegir variantes, umbrales ni rho se hace solo con el resto ("desarrollo"), mediante particiones agrupadas
# repetidas entrenamiento/validacion. Advertencia: las 339 fotos ya fueron inspeccionadas y usadas en la primera
# ronda de experimentos; la particion de informe NO es un conjunto externo intacto, solo evita que esta
# ronda la use para elegir.
BLOQUES = ("desarrollo", "informe")


def _repartir_grupos(filas, semilla, proporciones, nombres, campo):
    rng = random.Random(semilla)
    por_estrato = defaultdict(lambda: defaultdict(list))
    for f in filas:
        por_estrato[f["estrato"]][f["grupo"]].append(f)
    for estrato in sorted(por_estrato):
        grupos = sorted(por_estrato[estrato].items())
        rng.shuffle(grupos)
        grupos.sort(key=lambda kv: -len(kv[1]))
        total = sum(len(g) for _, g in grupos)
        cuota = [p * total for p in proporciones]
        cargado = [0] * len(nombres)
        for _, fotos in grupos:
            k = max(range(len(nombres)), key=lambda i: (cuota[i] - cargado[i]) / max(cuota[i], 1e-9)
                    + rng.uniform(0, 0.3))
            for f in fotos:
                f[campo] = nombres[k]
            cargado[k] += len(fotos)
    return filas


def reservar_informe(filas, semilla, fraccion=0.3):
    """Marca cada foto con 'bloque' = 'desarrollo' | 'informe' (por grupo, estratificado por origen)."""
    return _repartir_grupos(filas, semilla, (1 - fraccion, fraccion), BLOQUES, "bloque")


COLUMNAS_MANIFIESTO_RONDA2 = ("foto_id", "ruta", "estrato", "tipo_real", "calidad_real", "marca_tiempo", "grupo",
                              "bloque", "id_orig_morfologia", "id_orig_calidad")


def escribir_manifiesto_ronda2(filas, ruta):
    """Manifiesto de la ronda 2: igual al anterior pero con 'bloque' (desarrollo / informe) en lugar de 'particion'."""
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS_MANIFIESTO_RONDA2, extrasaction="ignore")
        w.writeheader()
        for fila in filas:
            w.writerow({c: fila.get(c, "") for c in COLUMNAS_MANIFIESTO_RONDA2})


def particionar_desarrollo(filas, semilla, prop_validacion=0.3):
    """Con 'bloque' ya asignado: las fotos de desarrollo se reparten por grupo en 'entrenamiento' y
    'validacion'; las de informe quedan como 'prueba' (la misma en todas las semillas)."""
    dev = [f for f in filas if f["bloque"] == "desarrollo"]
    _repartir_grupos(dev, semilla, (1 - prop_validacion, prop_validacion), ("entrenamiento", "validacion"), "particion")
    for f in filas:
        if f["bloque"] == "informe":
            f["particion"] = "prueba"
    return filas
