"""
validacion.py
Esquemas de los datasets CSV conocidos, lectura validada de CSV, validacion de
modelos persistidos y codigos de salida comunes a todos los scripts.

Por que existe: la lectura original (CarGross.leer_csv) deducia que columnas
eran caracteristicas mirando si sus valores eran {0,1}. Con esa regla una
caracteristica con un valor invalido (por ejemplo un "2" o un celda vacia)
pasaba en silencio a "metadato" y la red se entrenaba con menos variables.
Aqui los esquemas son explicitos: cada columna esta declarada y cualquier
desvio es un error con fila y columna.

Modo generico (otros CSV): solo con esquema="generico" y declarando de forma
explicita la columna de etiqueta (y las de metadatos). Toda columna restante
debe ser binaria 0/1 o se rechaza el archivo. Las etiquetas de referencia
(columnas *_real) jamas se toman como caracteristicas.
"""
import csv
import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Codigos de salida (compartidos por CarGross.py, probar_imagen.py, demo.py y
# scripts/): 0 solo cuando la tarea se completo.
# ---------------------------------------------------------------------------
COD_OK = 0
COD_INFRUCTUOSA = 1   # corrida infructuosa: proliferacion, sin resultado util
COD_USO = 2           # argumentos invalidos (es el codigo que usa argparse)
COD_ENTRADA = 3       # archivo/imagen/modelo de entrada invalido o ausente
COD_ERROR = 4         # error interno inesperado

MAX_PROBLEMAS_MOSTRADOS = 20
ALIAS_MANZANA = "carozo"  # alias historico: en este proyecto "carozo" = manzana


class ErrorEntrada(Exception):
    """Entrada invalida (dataset, imagen o modelo). Lleva todos los problemas
    encontrados, no solo el primero, para corregirlos de una vez."""

    def __init__(self, problemas, archivo=None):
        if isinstance(problemas, str):
            problemas = [problemas]
        self.problemas = list(problemas)
        self.archivo = archivo
        super().__init__(self._formatear())

    def _formatear(self):
        cab = f"'{self.archivo}': " if self.archivo else ""
        visibles = self.problemas[:MAX_PROBLEMAS_MOSTRADOS]
        resto = len(self.problemas) - len(visibles)
        texto = "\n  - ".join(visibles)
        if len(self.problemas) == 1:
            return cab + self.problemas[0]
        extra = f"\n  ... y {resto} problema(s) mas" if resto > 0 else ""
        return f"{cab}{len(self.problemas)} problemas:\n  - {texto}{extra}"


class ErrorModelo(ErrorEntrada):
    """Archivo de modelo ART1 invalido o incompatible."""


# ---------------------------------------------------------------------------
# Esquemas
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Esquema:
    nombre: str
    columna_id: str
    features_base: tuple          # comprometidas en el primer avance, en orden
    features_extra: tuple         # descriptores adicionales documentados (opcionales)
    columna_etiqueta: str
    etiquetas_validas: tuple
    descripcion: str = ""

    @property
    def todas_las_features(self):
        return self.features_base + self.features_extra


ESQUEMA_MORFOLOGIA = Esquema(
    nombre="morfologia",
    columna_id="fruta_id",
    features_base=("es_esferica", "cascara_rugosa", "tiene_tallo",
                   "densidad_alta", "pigmentacion_oscura", "color_uniforme"),
    # Descriptores cromaticos adicionales (sugerencia docente). Ver docs/.
    features_extra=("tono_calido", "saturacion_alta", "croma_alto", "color_con_gradiente",
                    "tono_verdoso", "tono_rojizo", "tono_amarillento", "rojo_dominante", "naranja_dominante",
                    "amarillo_dominante", "verde_dominante"),
    columna_etiqueta="tipo_real",
    etiquetas_validas=("citrico", "tropical", "carozo"),
    descripcion="Etapa 1: tipo de fruta. 'carozo' es el alias historico de manzana.",
)

ESQUEMA_CALIDAD = Esquema(
    nombre="calidad",
    columna_id="fruta_id",
    features_base=("es_grande", "es_pesada", "textura_firme",
                   "sin_defectos_superficie", "color_homogeneo"),
    features_extra=("sin_oscurecimiento", "color_sin_gradiente", "croma_alto"),
    columna_etiqueta="calidad_real",
    etiquetas_validas=("premium", "comercial", "descarte"),
    descripcion="Etapa 2: calidad de manzanas (referencias premium/comercial/descarte).",
)

ESQUEMAS = {e.nombre: e for e in (ESQUEMA_MORFOLOGIA, ESQUEMA_CALIDAD)}
NOMBRES_ESQUEMA = tuple(ESQUEMAS) + ("generico",)


def esquema_por_features(cols_features):
    """Devuelve el esquema conocido cuyo conjunto de variables contiene a
    cols_features (util para modelos legados). None si no coincide."""
    cols = set(cols_features)
    for esq in ESQUEMAS.values():
        if set(esq.features_base) <= cols <= set(esq.todas_las_features):
            return esq
    return None


@dataclass
class Dataset:
    ruta: str
    esquema: str
    columna_id: str
    ids: list
    entradas: list                 # lista de vectores 0/1 (orden = cols_features)
    cols_features: list
    columna_etiqueta: str = None
    etiquetas: list = None         # None si el CSV no trae etiqueta de referencia
    cols_metadata: list = field(default_factory=list)
    metadata: list = field(default_factory=list)   # una dict por fila (sin features)
    advertencias: list = field(default_factory=list)

    def __len__(self):
        return len(self.ids)


# ---------------------------------------------------------------------------
# Lectura validada
# ---------------------------------------------------------------------------
def _leer_crudo(ruta):
    """Devuelve (encabezado, filas[(nro_linea, lista_de_celdas)]). Detecta
    archivo ausente, vacio y CSV mal formado."""
    if not os.path.isfile(ruta):
        raise ErrorEntrada("no se encontro el archivo", archivo=ruta)
    if os.path.getsize(ruta) == 0:
        raise ErrorEntrada("el archivo esta vacio (0 bytes)", archivo=ruta)
    try:
        with open(ruta, newline="", encoding="utf-8-sig") as f:
            lector = csv.reader(f, strict=True)
            filas = [(lector.line_num, fila) for fila in lector if fila]
    except UnicodeDecodeError as e:
        raise ErrorEntrada(f"no es texto UTF-8 valido ({e})", archivo=ruta)
    except csv.Error as e:
        raise ErrorEntrada(f"CSV mal formado ({e})", archivo=ruta)
    if not filas:
        raise ErrorEntrada("el archivo no contiene encabezado ni datos", archivo=ruta)
    encabezado = [c.strip() for c in filas[0][1]]
    return encabezado, filas[1:]


def _resolver_esquema(nombre, encabezado):
    if nombre in ESQUEMAS:
        return ESQUEMAS[nombre]
    if nombre != "auto":
        raise ErrorEntrada(f"esquema desconocido '{nombre}'. Opciones: auto, "
                           f"{', '.join(NOMBRES_ESQUEMA)}")
    for esq in ESQUEMAS.values():
        if esq.columna_etiqueta in encabezado:
            return esq
    cols = set(encabezado)
    for esq in ESQUEMAS.values():
        if set(esq.features_base) <= cols:
            return esq
    raise ErrorEntrada(
        "el encabezado no coincide con ningun esquema conocido "
        f"({', '.join(ESQUEMAS)}). Columnas recibidas: {encabezado}. Para un CSV "
        "propio use esquema 'generico' declarando la etiqueta, por ejemplo: "
        "--esquema generico --col-etiqueta NOMBRE")


def leer_dataset(ruta, esquema="auto", columna_id=None, col_etiqueta=None,
                 cols_metadata=None):
    """Lee y valida un CSV. Devuelve un Dataset o levanta ErrorEntrada con TODOS
    los problemas encontrados.

    esquema: 'auto' (deduce por encabezado entre los conocidos), un nombre de
    ESQUEMAS, o 'generico' (requiere declarar col_etiqueta/cols_metadata; las
    demas columnas deben ser binarias).
    """
    encabezado, filas = _leer_crudo(ruta)
    esq = None if esquema == "generico" else _resolver_esquema(esquema, encabezado)
    problemas = []

    id_col = columna_id or (esq.columna_id if esq else "fruta_id")
    if "" in encabezado:
        problemas.append("el encabezado tiene una columna sin nombre")
    repetidas = sorted({c for c in encabezado if encabezado.count(c) > 1 and c})
    if repetidas:
        problemas.append(f"columnas repetidas en el encabezado: {repetidas}")
    if id_col not in encabezado:
        problemas.append(f"falta la columna identificadora '{id_col}'")

    cols_meta = list(cols_metadata or [])
    if esq is not None:
        col_et = esq.columna_etiqueta
        if col_et not in encabezado:
            # Dataset sin etiquetas de referencia (p. ej. a clasificar): permitido,
            # pero no habra pureza ni matriz de confusion.
            col_et = None
        faltantes = [c for c in esq.features_base if c not in encabezado]
        if faltantes:
            problemas.append(f"faltan caracteristicas obligatorias del esquema "
                             f"'{esq.nombre}': {faltantes}")
        extras_presentes = [c for c in esq.features_extra if c in encabezado]
        cols_features = [c for c in esq.features_base if c in encabezado] + extras_presentes
        permitidas = {id_col, *cols_features, *( [col_et] if col_et else [] ), *cols_meta}
        desconocidas = [c for c in encabezado if c and c not in permitidas]
        if desconocidas:
            problemas.append(
                f"columnas que el esquema '{esq.nombre}' no define: {desconocidas}. "
                "Una columna desconocida no se toma como caracteristica ni como "
                "metadato en forma automatica; si es un CSV propio use --esquema generico")
    else:
        col_et = col_etiqueta
        if col_et and col_et not in encabezado:
            problemas.append(f"la columna de etiqueta declarada '{col_et}' no existe")
        for c in cols_meta:
            if c not in encabezado:
                problemas.append(f"la columna de metadatos declarada '{c}' no existe")
        declaradas = {id_col, *( [col_et] if col_et else [] ), *cols_meta}
        sospechosas = [c for c in encabezado
                       if c and c not in declaradas and c.endswith("_real")]
        if sospechosas:
            problemas.append(
                f"las columnas {sospechosas} parecen etiquetas de referencia y no estan "
                "declaradas: indiquelas con --col-etiqueta/--col-meta para que no "
                "ingresen como caracteristicas")
        cols_features = [c for c in encabezado if c and c not in declaradas]
        if not cols_features:
            problemas.append("no queda ninguna columna para usar como caracteristica")

    if problemas:
        raise ErrorEntrada(problemas, archivo=ruta)

    indice = {c: i for i, c in enumerate(encabezado)}
    n_cols = len(encabezado)
    ids, entradas, etiquetas, metadata = [], [], [], []
    vistos = {}
    advertencias = []
    for nro, celdas in filas:
        if len(celdas) != n_cols:
            problemas.append(f"linea {nro}: tiene {len(celdas)} campos y el encabezado "
                             f"{n_cols}")
            continue
        celdas = [c.strip() for c in celdas]
        fid = celdas[indice[id_col]]
        if fid == "":
            problemas.append(f"linea {nro}: identificador vacio")
        elif fid in vistos:
            problemas.append(f"linea {nro}: identificador duplicado '{fid}' "
                             f"(ya aparecia en la linea {vistos[fid]})")
        else:
            vistos[fid] = nro
        vec = []
        for c in cols_features:
            v = celdas[indice[c]]
            if v == "":
                problemas.append(f"linea {nro}, columna '{c}': valor ausente")
            elif v not in ("0", "1"):
                problemas.append(f"linea {nro}, columna '{c}': valor no binario '{v}' "
                                 "(se esperaba 0 o 1)")
            else:
                vec.append(int(v))
        et = None
        if col_et:
            et = celdas[indice[col_et]]
            if et == "":
                problemas.append(f"linea {nro}, columna '{col_et}': etiqueta ausente")
            elif esq is not None and et not in esq.etiquetas_validas:
                problemas.append(f"linea {nro}, columna '{col_et}': etiqueta '{et}' "
                                 f"fuera de {list(esq.etiquetas_validas)}")
        ids.append(fid)
        entradas.append(vec)
        etiquetas.append(et)
        metadata.append({c: celdas[indice[c]] for c in cols_meta})

    if not filas:
        problemas.append("el archivo tiene encabezado pero ninguna fila de datos")
    if problemas:
        raise ErrorEntrada(problemas, archivo=ruta)

    n_nulos = sum(1 for v in entradas if not any(v))
    if n_nulos:
        advertencias.append(
            f"{n_nulos} fila(s) con vector nulo (todas las caracteristicas en 0). ART1 "
            "no define el match |I^T|/|I| para |I|=0; se aplica la politica de vector "
            "nulo de la red (ver CarGross.RedART1)")
    return Dataset(
        ruta=ruta, esquema=esq.nombre if esq else "generico", columna_id=id_col,
        ids=ids, entradas=entradas, cols_features=cols_features,
        columna_etiqueta=col_et, etiquetas=etiquetas if col_et else None,
        cols_metadata=cols_meta, metadata=metadata, advertencias=advertencias)


def validar_compatibilidad(ds, cols_features_modelo):
    """Comprueba que el dataset tenga exactamente las variables del modelo
    (mismos nombres; el orden lo resuelve el nombre, no la posicion)."""
    p = []
    if len(ds.cols_features) != len(cols_features_modelo):
        p.append(f"dimensiones incompatibles: el dataset tiene {len(ds.cols_features)} "
                 f"caracteristicas y el modelo {len(cols_features_modelo)}")
    falta = [c for c in cols_features_modelo if c not in ds.cols_features]
    sobra = [c for c in ds.cols_features if c not in cols_features_modelo]
    if falta:
        p.append(f"al dataset le faltan las variables del modelo: {falta}")
    if sobra:
        p.append(f"el dataset trae variables que el modelo no usa: {sobra}")
    if p:
        raise ErrorEntrada(p, archivo=ds.ruta)


def reordenar_a_modelo(ds, cols_features_modelo):
    """Devuelve las entradas del dataset en el orden de variables del modelo."""
    validar_compatibilidad(ds, cols_features_modelo)
    pos = [ds.cols_features.index(c) for c in cols_features_modelo]
    return [[fila[i] for i in pos] for fila in ds.entradas]


# ---------------------------------------------------------------------------
# Validacion del preprocesamiento (reglas y umbrales que convierten una foto en el vector)
# ---------------------------------------------------------------------------
SEGMENTADORES = ("v1", "v2")
NORMALIZACIONES = ("ninguna", "estirar", "gris_mundo", "gamma_media")
TIPOS_MODELO = ("fotografias", "vectores")
_COLOR_BASE = ("tono_medio", "desvio_tono", "saturacion_media", "frac_oscuros", "dist_tono_naranja",
               "dist_tono_verde", "dist_tono_rojo", "dist_tono_amarillo", "croma_medio", "gradiente_color",
               "frac_rojo", "frac_naranja", "frac_amarillo", "frac_verde")
# Medidas continuas que produce extraccion.medir_foto (una prueba comprueba que coincidan).
MEDIDAS_CONOCIDAS = frozenset(
    {"aspecto_esferico", "textura_laplaciano", "profundidad_defecto", "solidez", "gris_promedio", "desvio_hue",
     "area", "volumen_proxy", "defecto_score", "tallo_relativo", "defecto_convexidad", "textura_norm"}
    | {b + s for b in _COLOR_BASE for s in ("", "_n", "_wb", "_nwb")})


def validar_preprocesamiento_dict(pre, cols_features=None):
    """Comprueba que un preprocesamiento este COMPLETO y sea coherente: version, etapa conocida, segmentador,
    tamano de canvas, variables en el orden del esquema (base + adicionales canonicas), una regla (medida
    existente y operador) y un umbral numerico finito por variable, y ninguna regla/umbral sobrante.
    Si se da cols_features, deben coincidir EXACTAMENTE (nombres y orden). Levanta ErrorEntrada."""
    p = []
    if not isinstance(pre, dict):
        raise ErrorEntrada("el preprocesamiento no es un objeto")
    for clave in ("version", "etapa", "segmentador", "tam_canvas", "cols_features", "reglas", "umbrales"):
        if clave not in pre:
            p.append(f"preprocesamiento.{clave} ausente")
    if p:
        raise ErrorEntrada(p)
    if pre["version"] not in ("v1", "v2"):
        p.append(f"version de preprocesamiento desconocida: {pre['version']!r}")
    if pre["segmentador"] not in SEGMENTADORES:
        p.append(f"segmentador desconocido: {pre['segmentador']!r} (conocidos: {SEGMENTADORES})")
    if pre.get("normalizacion", "ninguna") not in NORMALIZACIONES:
        p.append(f"normalizacion de iluminacion desconocida: {pre.get('normalizacion')!r} (conocidas: {NORMALIZACIONES})")
    tam = pre["tam_canvas"]
    if not (isinstance(tam, int) and not isinstance(tam, bool) and tam > 0):
        p.append(f"tam_canvas invalido: {tam!r}")
    esq = ESQUEMAS.get(pre["etapa"])
    cols = pre["cols_features"]
    if esq is None:
        p.append(f"etapa desconocida: {pre['etapa']!r} (conocidas: {list(ESQUEMAS)})")
    elif not isinstance(cols, list):
        p.append("cols_features no es una lista")
    else:
        extras = [c for c in esq.features_extra if c in cols]
        esperadas = list(esq.features_base) + extras
        if cols != esperadas:
            p.append(f"cols_features {cols} no es el orden esperado para la etapa '{pre['etapa']}' "
                     f"(variables comprometidas + descriptores adicionales canonicos): {esperadas}")
    if cols_features is not None and cols != list(cols_features):
        p.append(f"las variables del preprocesamiento {cols} no coinciden (nombres y orden) con las del "
                 f"dataset/modelo {list(cols_features)}")
    reglas, umbrales = pre["reglas"], pre["umbrales"]
    if isinstance(cols, list) and isinstance(reglas, dict) and isinstance(umbrales, dict):
        for c in cols:
            r = reglas.get(c)
            if not isinstance(r, dict):
                p.append(f"falta la regla de '{c}'")
            else:
                if r.get("crudo") not in MEDIDAS_CONOCIDAS:
                    p.append(f"'{c}': medida continua desconocida {r.get('crudo')!r}")
                if r.get("op") not in (">=", "<="):
                    p.append(f"'{c}': operador desconocido {r.get('op')!r}")
            u = umbrales.get(c)
            if c not in umbrales:
                p.append(f"falta el umbral de '{c}'")
            elif isinstance(u, bool) or not isinstance(u, (int, float)) or u != u or abs(u) == float("inf"):
                p.append(f"'{c}': umbral no numerico o no finito {u!r}")
        sobran = sorted((set(reglas) | set(umbrales)) - set(cols))
        if sobran:
            p.append(f"reglas/umbrales de variables que no estan en el vector: {sobran}")
    else:
        p.append("reglas y umbrales deben ser objetos")
    if p:
        raise ErrorEntrada(p)


# ---------------------------------------------------------------------------
# Validacion de modelos persistidos
# ---------------------------------------------------------------------------
def validar_modelo_dict(datos, ruta=None):
    """Valida estructura y coherencia de un modelo cargado desde JSON.
    Soporta formato 1 (entregado, sin clave 'formato_version') y formato 2."""
    p = []
    if not isinstance(datos, dict):
        raise ErrorModelo("el JSON del modelo no es un objeto", archivo=ruta)
    version = datos.get("formato_version", 1)
    if version not in (1, 2):
        p.append(f"formato_version {version!r} no soportado (conocidos: 1 y 2)")
    for clave in ("n_features", "rho", "top_down"):
        if clave not in datos:
            p.append(f"falta la clave '{clave}'")
    if p:
        raise ErrorModelo(p, archivo=ruta)
    n, rho, L = datos["n_features"], datos["rho"], datos.get("L", 2.0)
    if not (isinstance(n, int) and not isinstance(n, bool) and n > 0):
        p.append(f"n_features invalido: {n!r}")
    if not (isinstance(rho, (int, float)) and 0 < rho <= 1):
        p.append(f"rho fuera de (0, 1]: {rho!r}")
    if not (isinstance(L, (int, float)) and L > 1):
        p.append(f"L debe ser > 1: {L!r}")
    top = datos["top_down"]
    if not isinstance(top, list):
        p.append("top_down no es una lista")
        top = []
    maxc = datos.get("max_categorias", 50)
    if len(top) > maxc:
        p.append(f"hay {len(top)} categorias y max_categorias={maxc}")
    for j, t in enumerate(top):
        if not (isinstance(t, list) and len(t) == n and all(v in (0, 1) and not isinstance(v, bool) for v in t)):
            p.append(f"top_down[{j}] no es un vector binario de largo {n}")
    meta = datos.get("metadata_extra", {})
    cols = meta.get("cols_features")
    if cols is None:
        p.append("metadata_extra.cols_features ausente: no se puede alinear la entrada")
    elif len(cols) != n or len(set(cols)) != n:
        p.append(f"cols_features ({len(cols)}) no coincide con n_features ({n}) o tiene repetidas")
    for cid in (meta.get("mapeo_clusters") or {}):
        if not (cid.isdigit() and int(cid) < len(top)):
            p.append(f"mapeo_clusters referencia una categoria inexistente: {cid!r}")
    if datos.get("politica_nulo", "categoria_propia") not in ("categoria_propia", "rechazar"):
        p.append(f"politica_nulo desconocida: {datos.get('politica_nulo')!r}")
    if version == 2:
        pre = datos.get("preprocesamiento")
        tipo = datos.get("tipo_modelo", "fotografias" if isinstance(pre, dict) else "vectores")
        if tipo not in TIPOS_MODELO:
            p.append(f"tipo_modelo desconocido: {tipo!r} (conocidos: {TIPOS_MODELO})")
        elif tipo == "fotografias":
            if not isinstance(pre, dict):
                p.append("un modelo para fotografias requiere la seccion 'preprocesamiento'")
            else:
                try:
                    validar_preprocesamiento_dict(pre, cols)
                except ErrorEntrada as e:
                    p.extend(f"preprocesamiento: {x}" for x in e.problemas)
        elif pre is not None:
            p.append("un modelo 'vectores' no debe traer 'preprocesamiento' (use tipo_modelo 'fotografias')")
    if p:
        raise ErrorModelo(p, archivo=ruta)
    return version
