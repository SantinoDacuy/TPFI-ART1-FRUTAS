"""
extraccion.py
Pipeline imagen -> vector binario, DESCRITO POR DATOS: la configuracion de
preprocesamiento (segmentador, tamano de canvas, reglas y umbrales exactos,
orden de las variables) vive dentro del modelo y se puede escribir/leer como CSV,
de modo que entrenamiento, regeneracion, demo e inferencia aplican exactamente
lo mismo.

  medir_foto()      imagen -> valores continuos (superset de todas las medidas)
  vectorizar()      medicion + configuracion -> vector binario en el orden del modelo
  preprocesamiento_legado()  configuracion equivalente a la del primer avance
  configuracion()   arma una configuracion nueva
  umbrales_desde_entrenamiento()  calcula umbrales SOLO con datos de entrenamiento
  escribir/leer_preprocesamiento_csv()  la misma configuracion como CSV documentado

Las variables comprometidas en el primer avance se conservan con sus nombres:
  morfologia: es_esferica, cascara_rugosa, tiene_tallo, densidad_alta,
              pigmentacion_oscura, color_uniforme
  calidad   : es_grande, es_pesada, textura_firme, sin_defectos_superficie,
              color_homogeneo
Aclaracion de alcance: densidad_alta, es_pesada y textura_firme son
APROXIMACIONES VISUALES (solidez del contorno, area x solidez, varianza del
laplaciano). No miden densidad fisica, masa en gramos ni firmeza mecanica.
"""
import csv
import os
from dataclasses import dataclass, field

import numpy as np

import preprocesador as P
import preprocesador_calidad as PC
import validacion as val
from validacion import ESQUEMAS, ErrorEntrada

VERSION_V1 = "v1"
VERSION_V2 = "v2"
OPERADORES = (">=", "<=")

# Reglas v1: variable binaria -> (valor continuo, operador, umbral por defecto)
REGLAS_V1 = {
    "morfologia": {
        "es_esferica": ("aspecto_esferico", ">=", P.UMBRALES["aspecto_esferico"]),
        "cascara_rugosa": ("textura_laplaciano", ">=", P.UMBRALES["textura_laplaciano"]),
        "tiene_tallo": ("profundidad_defecto", ">=", P.UMBRALES["profundidad_defecto"]),
        "densidad_alta": ("solidez", ">=", P.UMBRALES["solidez"]),
        "pigmentacion_oscura": ("gris_promedio", "<=", P.UMBRALES["gris_promedio"]),
        "color_uniforme": ("desvio_hue", "<=", P.UMBRALES["desvio_hue"]),
    },
    "calidad": {
        "es_grande": ("area", ">=", PC.UMBRALES_REFERENCIA_CALIDAD["area"]),
        "es_pesada": ("volumen_proxy", ">=", PC.UMBRALES_REFERENCIA_CALIDAD["volumen_proxy"]),
        "textura_firme": ("textura_laplaciano", "<=", PC.UMBRALES_REFERENCIA_CALIDAD["textura_laplaciano"]),
        "sin_defectos_superficie": ("defecto_score", "<=", PC.UMBRALES_REFERENCIA_CALIDAD["defecto_score"]),
        "color_homogeneo": ("desvio_hue", "<=", PC.UMBRALES_REFERENCIA_CALIDAD["desvio_hue"]),
    },
}

# Opciones de medicion de las variables comprometidas (mismo nombre, otra medida)
TALLOS = {"v1": ("profundidad_defecto", ">="),          # primer avance (0 con OpenCV 5)
          "convexidad": ("defecto_convexidad", ">="),    # MISMO criterio del primer avance, compatible con OpenCV 5
          "protuberancia": ("tallo_relativo", ">=")}     # otro detector (experimental, sin anotaciones de referencia)
TONOS = {"lineal": "desvio_hue", "circular": "desvio_tono"}

# Descriptores cromaticos ADICIONALES (sugerencia docente): variable -> (medida base, operador).
# La medida efectiva lleva el sufijo de la opcion 'color' (ver COLOR_SUFIJOS).
EXTRAS = {
    "morfologia": {
        "tono_calido": ("dist_tono_naranja", "<="),
        "saturacion_alta": ("saturacion_media", ">="),
        "croma_alto": ("croma_medio", ">="),
        "color_con_gradiente": ("gradiente_color", ">="),
        "tono_verdoso": ("dist_tono_verde", "<="),
        "tono_rojizo": ("dist_tono_rojo", "<="),
        "tono_amarillento": ("dist_tono_amarillo", "<="),
        "rojo_dominante": ("frac_rojo", ">="),
        "naranja_dominante": ("frac_naranja", ">="),
        "amarillo_dominante": ("frac_amarillo", ">="),
        "verde_dominante": ("frac_verde", ">="),
    },
    "calidad": {
        "sin_oscurecimiento": ("frac_oscuros", "<="),
        "color_sin_gradiente": ("gradiente_color", "<="),
        "croma_alto": ("croma_medio", ">="),
    },
}
# 'mascara': toda la fruta segmentada; 'nucleo': mascara erosionada (sin el borde fruta/fondo);
# 'balanceado': tras un balance de blancos "mundo gris" de la foto; 'nucleo_balanceado': ambas.
COLOR_SUFIJOS = {"mascara": "", "nucleo": "_n", "balanceado": "_wb", "nucleo_balanceado": "_nwb"}
COLOR_BASE = ("tono_medio", "desvio_tono", "saturacion_media", "frac_oscuros", "dist_tono_naranja",
              "dist_tono_verde", "dist_tono_rojo", "dist_tono_amarillo", "croma_medio", "gradiente_color",
              "frac_rojo", "frac_naranja", "frac_amarillo", "frac_verde")

UMBRAL_FIJO_EXTRAS = {"tono_calido": 30.0}   # grados de distancia al naranja (30 deg)
UMBRAL_FIJO_TALLO = {"v1": 0.06, "convexidad": 0.06, "protuberancia": 0.12}
# Cortes originales del primer avance para las variables morfologicas comprometidas
CORTES_ORIGINALES_MORFOLOGIA = {c: r[2] for c, r in REGLAS_V1["morfologia"].items()}
ESTRATEGIAS_UMBRAL = ("compromiso", "fijo", "mediana", "otsu")


@dataclass
class Medicion:
    crudos: dict
    mascara: np.ndarray
    img_c: np.ndarray
    masc_c: np.ndarray
    info_seg: dict = field(default_factory=dict)
    img: np.ndarray = None


def _dist_circular(a, b):
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def medir_foto(img_bgr, segmentador=VERSION_V2, tam=P.TAM_CANVAS, normalizacion="ninguna"):
    """Valores continuos de TODAS las medidas (morfologia, calidad y cromaticas).
    segmentador 'v1' reproduce exactamente el primer avance (mascara con
    interpolacion lineal incluida); 'v2' usa segmentar_v2 y mascara binaria.
    normalizacion: paso previo de iluminacion (preprocesador.NORMALIZACIONES); 'ninguna' = primer avance."""
    img_bgr = P.normalizar_iluminacion(img_bgr, normalizacion)
    if segmentador == VERSION_V1:
        mascara, info = P.segmentar(img_bgr), {"candidato": "v1", "advertencia": None}
        img_c, masc_c = P.centrar_y_escalar(img_bgr, mascara, tam)
    elif segmentador == VERSION_V2:
        mascara, info = P.segmentar_v2(img_bgr, devolver_info=True)
        img_c, masc_c = P.centrar_y_escalar_v2(img_bgr, mascara, tam)
    else:
        raise ValueError(f"segmentador desconocido: {segmentador!r}")
    crudos = dict(P.extraer_features(img_c, masc_c))
    crudos.update(PC.extraer_crudos_calidad(img_bgr, mascara, img_c, masc_c))
    crudos["tallo_relativo"] = P.medir_tallo(masc_c)
    crudos["defecto_convexidad"] = P.profundidad_defecto_v2(masc_c)
    crudos["textura_norm"] = P.textura_normalizada(img_c, masc_c)
    nucleo = P.nucleo_mascara(masc_c)
    img_wb = P.aplicar_factores(img_c, P.factores_gris(img_bgr))
    for opcion, (im, mk) in {"mascara": (img_c, masc_c), "nucleo": (img_c, nucleo),
                             "balanceado": (img_wb, masc_c), "nucleo_balanceado": (img_wb, nucleo)}.items():
        est = P.estadisticas_color(im, mk)
        for k in COLOR_BASE:
            crudos[k + COLOR_SUFIJOS[opcion]] = est[k]
    return Medicion(crudos=crudos, mascara=mascara, img_c=img_c, masc_c=masc_c, info_seg=info, img=img_bgr)


# ---------------------------------------------------------------------------
# Configuracion de preprocesamiento
# ---------------------------------------------------------------------------
def preprocesamiento_legado(etapa):
    """Configuracion que reproduce el primer avance (formato 1)."""
    if etapa not in REGLAS_V1:
        raise ErrorEntrada(f"no hay preprocesamiento legado para la etapa '{etapa}'")
    reglas = REGLAS_V1[etapa]
    return {
        "version": VERSION_V1, "etapa": etapa, "segmentador": VERSION_V1, "tam_canvas": P.TAM_CANVAS,
        "cols_features": list(ESQUEMAS[etapa].features_base),
        "reglas": {c: {"crudo": r[0], "op": r[1]} for c, r in reglas.items()},
        "umbrales": {c: r[2] for c, r in reglas.items()},
        "origen_umbrales": "constantes del primer avance (preprocesador.UMBRALES / "
                           "preprocesador_calidad.UMBRALES_REFERENCIA_CALIDAD)",
    }


def _reglas_base(etapa, tallo, tono):
    """{variable: (crudo, operador)} de las variables comprometidas segun las
    opciones de medicion de tiene_tallo y del desvio de tono."""
    reglas = {c: (r[0], r[1]) for c, r in REGLAS_V1[etapa].items()}
    if etapa == "morfologia":
        reglas["tiene_tallo"] = TALLOS[tallo]
        reglas["color_uniforme"] = (TONOS[tono], "<=")
    else:
        reglas["color_homogeneo"] = (TONOS[tono], "<=")
    return reglas


def _regla_extra(etapa, col, color):
    if color not in COLOR_SUFIJOS:
        raise ValueError(f"opcion de color desconocida: {color!r} (opciones: {tuple(COLOR_SUFIJOS)})")
    if col not in EXTRAS[etapa]:
        raise ValueError(f"descriptor adicional desconocido para '{etapa}': {col!r}")
    medida, op = EXTRAS[etapa][col]
    return medida + COLOR_SUFIJOS[color], op


def configuracion(etapa, umbrales, segmentador=VERSION_V2, tallo="protuberancia", tono="circular",
                  extras=(), origen_umbrales="", tam=P.TAM_CANVAS, color="mascara", normalizacion="ninguna"):
    """Arma una configuracion de preprocesamiento (se guarda dentro del modelo).
    umbrales: {variable_binaria: valor} para TODAS las variables elegidas."""
    reglas = {c: {"crudo": r[0], "op": r[1]} for c, r in _reglas_base(etapa, tallo, tono).items()}
    for col in extras:
        crudo, op = _regla_extra(etapa, col, color)
        reglas[col] = {"crudo": crudo, "op": op}
    faltan = [c for c in reglas if c not in umbrales]
    if faltan:
        raise ValueError(f"faltan umbrales para {faltan}")
    extras = [c for c in ESQUEMAS[etapa].features_extra if c in extras]   # orden canonico del esquema
    cols = list(ESQUEMAS[etapa].features_base) + extras
    pre = {"version": VERSION_V2, "etapa": etapa, "segmentador": segmentador, "tam_canvas": tam,
           "normalizacion": normalizacion,
           "opciones": {"tallo": tallo, "tono": tono, "extras": extras, "color": color},
           "cols_features": cols, "reglas": reglas,
           "umbrales": {c: float(umbrales[c]) for c in reglas}, "origen_umbrales": origen_umbrales}
    val.validar_preprocesamiento_dict(pre, cols)
    return pre


def validar_preprocesamiento(pre, cols_features=None):
    """Validacion completa (delegada en validacion.py): cada variable tiene regla y umbral, las medidas
    existen, el orden coincide con el del esquema/modelo. Levanta ErrorEntrada."""
    val.validar_preprocesamiento_dict(pre, cols_features)


def binarizar(crudos, pre):
    """Valores continuos -> {variable: 0/1} segun las reglas y umbrales del modelo."""
    out = {}
    for col in pre["cols_features"]:
        r, u = pre["reglas"][col], pre["umbrales"][col]
        v = crudos[r["crudo"]]
        out[col] = int(v >= u) if r["op"] == ">=" else int(v <= u)
    return out


def vectorizar(medicion_o_crudos, pre):
    """Vector binario en el orden exacto de pre['cols_features']."""
    crudos = medicion_o_crudos.crudos if isinstance(medicion_o_crudos, Medicion) else medicion_o_crudos
    b = binarizar(crudos, pre)
    return [b[c] for c in pre["cols_features"]]


def _corte_otsu(vals, bins=64):
    """Umbral de Otsu 1-D (maximiza la varianza entre grupos) sobre una muestra."""
    v = np.asarray(vals, dtype=np.float64)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return lo
    hist, bordes = np.histogram(v, bins=bins, range=(lo, hi))
    p = hist / hist.sum()
    centros = (bordes[:-1] + bordes[1:]) / 2
    w0 = np.cumsum(p)
    m0 = np.cumsum(p * centros)
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (m0[-1] * w0 - m0) ** 2 / (w0 * (1 - w0))
    return float(bordes[int(np.argmax(np.nan_to_num(sb))) + 1])


def umbrales_desde_entrenamiento(etapa, crudos_entrenamiento, estrategia="mediana",
                                 tallo="protuberancia", tono="circular", extras=(), color="mascara"):
    """Calcula umbrales SOLO con las mediciones de entrenamiento.
      'compromiso': variante PRINCIPAL compatible con el primer avance. Variables morfologicas
                 comprometidas: los cortes ORIGINALES (aspecto 0.80, rugosidad 150, solidez 0.90, gris 130,
                 desvio de tono 12 y, para tiene_tallo, 0.06 si se mide por defectos de convexidad). Variables
                 de calidad y descriptores adicionales (que el primer avance ya definia o define relativos al
                 lote): mediana del entrenamiento.
      'fijo'   : constantes por defecto del primer avance (las de calidad son las medianas redondeadas del
                 lote original); no depende de los datos. Lo usa la receta original reentrenada.
      'mediana': mediana de cada valor continuo en el entrenamiento para TODAS las variables (EXPERIMENTAL:
                 cambia el significado de variables como es_esferica).
      'otsu'   : corte natural de la distribucion en el entrenamiento (EXPERIMENTAL, no supervisado).
    crudos_entrenamiento: lista de dicts de medidas continuas. Devuelve {variable_binaria: umbral}."""
    if estrategia not in ESTRATEGIAS_UMBRAL:
        raise ValueError(f"estrategia desconocida: {estrategia!r} (opciones: {ESTRATEGIAS_UMBRAL})")
    reglas = _reglas_base(etapa, tallo, tono)
    defecto = {c: r[2] for c, r in REGLAS_V1[etapa].items()}
    if etapa == "morfologia":
        defecto["tiene_tallo"] = UMBRAL_FIJO_TALLO[tallo]
    if estrategia == "fijo" and tono == "circular":   # otra medida: la constante lineal no es transferible
        defecto.pop("color_uniforme" if etapa == "morfologia" else "color_homogeneo", None)
    if estrategia == "compromiso":
        defecto = {}
        if etapa == "morfologia":
            defecto = dict(CORTES_ORIGINALES_MORFOLOGIA)
            if tallo == "protuberancia":
                defecto.pop("tiene_tallo")            # detector nuevo: sin corte original
            elif tallo == "convexidad":
                defecto["tiene_tallo"] = UMBRAL_FIJO_TALLO["convexidad"]
    for col in extras:
        reglas[col] = _regla_extra(etapa, col, color)
        if estrategia == "fijo" and col in UMBRAL_FIJO_EXTRAS:
            defecto[col] = UMBRAL_FIJO_EXTRAS[col]
    out = {}
    for col, (crudo, op) in reglas.items():
        vals = sorted(c[crudo] for c in crudos_entrenamiento)
        if estrategia in ("fijo", "compromiso") and col in defecto:
            out[col] = defecto[col]
        elif estrategia == "otsu":
            out[col] = _corte_otsu(vals)
        else:
            out[col] = vals[len(vals) // 2]     # misma convencion que preprocesador_calidad
    return out


# ---------------------------------------------------------------------------
# El preprocesamiento como CSV documentado (metadato de entrada de la red)
# ---------------------------------------------------------------------------
COLUMNAS_PREPROC_CSV = ["seccion", "clave", "medida_continua", "operador", "valor"]
_CLAVES_GLOBALES = ("version", "etapa", "segmentador", "tam_canvas", "normalizacion", "origen_umbrales")


def preprocesamiento_a_filas(pre):
    filas = [["global", k, "", "", str(pre[k])] for k in _CLAVES_GLOBALES if k in pre]
    for k, v in (pre.get("opciones") or {}).items():
        filas.append(["global", f"opciones.{k}", "", "", ";".join(v) if isinstance(v, (list, tuple)) else str(v)])
    for c in pre["cols_features"]:
        r = pre["reglas"][c]
        filas.append(["variable", c, r["crudo"], r["op"], repr(float(pre["umbrales"][c]))])
    return filas


def escribir_preprocesamiento_csv(pre, ruta):
    """Escribe el preprocesamiento (validado) como CSV: filas 'global' (version, etapa, segmentador, tam_canvas,
    origen_umbrales, opciones.*) y una fila 'variable' por cada variable del vector, en el orden del modelo."""
    validar_preprocesamiento(pre)
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNAS_PREPROC_CSV)
        w.writerows(preprocesamiento_a_filas(pre))


def leer_preprocesamiento_csv(ruta, cols_features=None):
    """Lee y valida un CSV de preprocesamiento. Si se da cols_features (las del dataset), comprueba que
    coincidan EXACTAMENTE (nombres y orden). Levanta ErrorEntrada con las filas problematicas."""
    if not os.path.isfile(ruta):
        raise ErrorEntrada("no se encontro el archivo de preprocesamiento", archivo=ruta)
    with open(ruta, newline="", encoding="utf-8-sig") as f:
        lector = csv.DictReader(f)
        if lector.fieldnames != COLUMNAS_PREPROC_CSV:
            raise ErrorEntrada(f"el encabezado debe ser {COLUMNAS_PREPROC_CSV}, se recibio {lector.fieldnames}",
                               archivo=ruta)
        filas = [(lector.line_num, r) for r in lector]
    pre, opciones, variables, problemas = {}, {}, [], []
    vistos = set()
    for nro, r in filas:
        sec, clave = r["seccion"].strip(), r["clave"].strip()
        if (sec, clave) in vistos:
            problemas.append(f"linea {nro}: '{sec}/{clave}' repetida")
        vistos.add((sec, clave))
        if sec == "global":
            if clave.startswith("opciones."):
                v = r["valor"]
                opciones[clave[len("opciones."):]] = [x for x in v.split(";") if x] if clave == "opciones.extras" else v
            elif clave in _CLAVES_GLOBALES:
                pre[clave] = r["valor"]
            else:
                problemas.append(f"linea {nro}: clave global desconocida '{clave}'")
        elif sec == "variable":
            try:
                variables.append((clave, r["medida_continua"].strip(), r["operador"].strip(), float(r["valor"])))
            except ValueError:
                problemas.append(f"linea {nro}: umbral no numerico '{r['valor']}' para '{clave}'")
        else:
            problemas.append(f"linea {nro}: seccion desconocida '{sec}' (esperadas: global, variable)")
    if problemas:
        raise ErrorEntrada(problemas, archivo=ruta)
    try:
        pre["tam_canvas"] = int(pre["tam_canvas"])
    except (KeyError, ValueError):
        raise ErrorEntrada("falta 'global/tam_canvas' o no es un entero", archivo=ruta)
    if opciones:
        pre["opciones"] = opciones
    pre["cols_features"] = [v[0] for v in variables]
    pre["reglas"] = {v[0]: {"crudo": v[1], "op": v[2]} for v in variables}
    pre["umbrales"] = {v[0]: v[3] for v in variables}
    try:
        val.validar_preprocesamiento_dict(pre, cols_features)
    except ErrorEntrada as e:
        raise ErrorEntrada(e.problemas, archivo=ruta)
    return pre
