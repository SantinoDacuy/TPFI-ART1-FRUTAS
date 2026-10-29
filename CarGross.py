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
    (empate exacto -> gana la categoria de menor indice)
  - Test de vigilancia: |I ^ t_j| / |I| >= rho ?
      si pasa  -> resonancia: t_j se actualiza como AND(I, t_j)
      si no pasa -> reset de ese nodo, se prueba el siguiente candidato
  - Si ningun nodo existente resuena -> se compromete una categoria nueva

Nodos no comprometidos: no compiten. Equivale a la formulacion de la
bibliografia (Fausett) con pesos bottom-up iniciales b_ij(0) arbitrariamente
chicos, que esa formulacion permite (0 < b_ij(0) < L/(L-1+n)): una categoria
nueva se crea solo si ninguna comprometida pasa la vigilancia.

El nucleo (competencia, vigilancia, reset, resonancia, aprendizaje) es el del
primer avance. Lo agregado alrededor: validacion de entradas (validacion.py),
politica explicita de vector nulo, etiquetado posterior con los prototipos
finales, modelo persistido con version/preprocesamiento y codigos de salida.

Limitacion de verificacion: la bibliografia de catedra de Lau (Lau.pp5.a.11 /
Lau.pp12.a.14) no esta en el repositorio; se contrasto con Carpenter &
Grossberg (1987) y Fausett, no con Lau. No se afirma equivalencia con Lau.

QUE APRENDE LA RED: prototipos binarios (interseccion de los patrones que
resonaron en cada categoria) sobre las variables de entrada. Las etiquetas
(citrico/tropical/carozo, premium/comercial/descarte) NO intervienen en el
aprendizaje: se asignan DESPUES, por mayoria entre las muestras de
entrenamiento, y solo sirven para nombrar categorias.

"carozo" es el alias historico de manzana en este proyecto.

USO:
    python CarGross.py --csv datasets/dataset_morfologia.csv --rho 0.8
    python CarGross.py --csv datasets/dataset_calidad_real.csv --rho 0.7
    python CarGross.py --csv datasets/dataset_morfologia.csv --guardar-modelo modelos_v2/prueba.json
    python CarGross.py --ayuda   (o --help)

Codigos de salida: 0 ok | 1 corrida infructuosa | 2 argumentos | 3 entrada
invalida (CSV/modelo) | 4 error interno. Ver validacion.py.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
from collections import Counter

import validacion as val
from validacion import (COD_ENTRADA, COD_ERROR, COD_INFRUCTUOSA, COD_OK, COD_USO,
                        ErrorEntrada, ErrorModelo)

FORMATO_MODELO = 2
POLITICAS_NULO = ("categoria_propia", "rechazar")
DESEMPATES = ("indeterminada", "mayoritaria_global")
MODOS_MAPEO = ("prototipos_finales", "online")


class ProliferacionError(Exception):
    """Se levanta cuando la red no logra estabilizar sus categorias.
    Es el mecanismo de 'corrida infructuosa' que pide la consigna: no
    cuelga el programa, corta con un mensaje explicando la causa probable."""
    pass


class VectorNuloError(ValueError):
    """Se intento APRENDER un vector nulo con politica_nulo='rechazar'."""
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
    politica_nulo : str
        Tratamiento del patron nulo (|I| = 0), para el que |I^T|/|I| no esta
        definido:
          'categoria_propia' (default, comportamiento del primer avance): el
              vector nulo solo resuena con un prototipo nulo (match 1.0 por
              convencion) y, al aprender, compromete uno. Un prototipo nulo
              nunca absorbe patrones no nulos porque |I^0| = 0 < rho*|I|.
          'rechazar': el vector nulo nunca resuena; aprenderlo levanta
              VectorNuloError e inferirlo devuelve novedad (motivo
              'vector_nulo_rechazado').
    """

    def __init__(self, n_features, rho, L=2.0, max_categorias=50,
                 politica_nulo="categoria_propia"):
        if not (0 < rho <= 1):
            raise ValueError(f"rho debe estar en (0, 1], se recibio {rho}")
        if L <= 1:
            raise ValueError(f"L debe ser > 1, se recibio {L}")
        if politica_nulo not in POLITICAS_NULO:
            raise ValueError(f"politica_nulo debe ser una de {POLITICAS_NULO}")
        self.n = n_features
        self.rho = rho
        self.L = L
        self.max_categorias = max_categorias
        self.politica_nulo = politica_nulo
        self.top_down = []  # un vector binario (lista de 0/1) por categoria comprometida

    # ---- utilidades ------------------------------------------------------
    def _bottom_up(self, t_j):
        norma = sum(t_j)
        denom = self.L - 1 + norma
        return [(self.L * b) / denom for b in t_j]

    @staticmethod
    def _interseccion(entrada, t_j):
        return [int(a and b) for a, b in zip(entrada, t_j)]

    def _neto(self, entrada, t_j):
        """Entrada neta bottom-up sum_i I_i * b_ij, calculada como una unica
        division (L*|I^t| / (L-1+|t|)): dos categorias con el mismo valor
        racional dan exactamente el mismo flotante, asi el desempate es
        determinista y no depende del orden de suma."""
        solape = sum(a & b for a, b in zip(entrada, t_j))
        return (self.L * solape) / (self.L - 1 + sum(t_j))

    def _validar(self, entrada):
        if len(entrada) != self.n:
            raise ValueError(f"Se esperaban {self.n} variables, llegaron {len(entrada)}")
        if any(v not in (0, 1) for v in entrada):
            raise ValueError(f"La entrada debe ser binaria (0/1): {list(entrada)}")
        return [int(v) for v in entrada]

    def huella(self):
        """Hash de los pesos: permite comprobar que inferir no los modifica."""
        return hashlib.sha1(json.dumps(self.top_down).encode()).hexdigest()[:12]

    def copia(self):
        r = RedART1(self.n, self.rho, self.L, self.max_categorias, self.politica_nulo)
        r.top_down = [list(t) for t in self.top_down]
        return r

    # ---- ciclo ART1 ------------------------------------------------------
    def evaluar(self, entrada):
        """Evalua un patron contra todos los clusters comprometidos sin modificar pesos.
        Devuelve el cluster ganador, el ratio de similitud |I ^ T_j| / |I|,
        si hubo resonancia (ratio >= rho), y detalles de candidatos.
        'ratio' mide cuanto del patron de entrada esta contenido en el
        prototipo: NO es simetrico, NO es una probabilidad de acierto y no
        implica que patron y prototipo sean iguales.
        """
        entrada = self._validar(entrada)
        norma_entrada = sum(entrada)
        if norma_entrada == 0:
            if self.politica_nulo == "rechazar":
                return {"ganador": None, "ratio": 0.0, "resuena": False,
                        "motivo": "vector_nulo_rechazado"}
            cat_nula = [0] * self.n
            for j, t in enumerate(self.top_down):
                if t == cat_nula:
                    return {"ganador": j, "ratio": 1.0, "resuena": True, "candidatos": []}
            return {"ganador": None, "ratio": 0.0, "resuena": False,
                    "motivo": "vector_nulo_no_aprendido"}

        if not self.top_down:
            return {"ganador": None, "ratio": 0.0, "resuena": False, "motivo": "sin_categorias"}

        candidatos = list(range(len(self.top_down)))
        detalles = []
        while candidatos:
            netos = [(j, self._neto(entrada, self.top_down[j])) for j in candidatos]
            ganador, neto = max(netos, key=lambda par: par[1])  # empate -> menor indice
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
            candidatos.remove(ganador)  # reset

        mejor = max(detalles, key=lambda d: d["ratio"])
        return {
            "ganador": None,
            "ratio": mejor["ratio"],
            "resuena": False,
            "motivo": "vigilancia",
            "mejor_candidato": mejor["cluster"],
            "candidatos": detalles
        }

    def clasificar(self, entrada, aprender=True):
        """Presenta un patron binario a la red.
        Si aprender=True: actualiza el prototipo resonante o compromete una nueva categoria.
        Si aprender=False (modo inferencia): devuelve el cluster resonante, o None si es NOVEDAD.
        La competencia es la misma de evaluar(): una sola implementacion del ciclo.
        """
        entrada = self._validar(entrada)
        res = self.evaluar(entrada)
        if res["resuena"]:
            if aprender and any(entrada):
                self.top_down[res["ganador"]] = self._interseccion(entrada, self.top_down[res["ganador"]])
            return res["ganador"]
        if not aprender:
            return None  # modo inferencia: ningun cluster paso la vigilancia -> NOVEDAD

        if not any(entrada) and self.politica_nulo == "rechazar":
            raise VectorNuloError("Vector nulo: con politica_nulo='rechazar' no se aprende")
        if len(self.top_down) >= self.max_categorias:
            if not any(entrada):
                raise ProliferacionError("Limite de categorias alcanzado al registrar categoria para patron nulo.")
            raise ProliferacionError(
                f"Se alcanzo el limite de {self.max_categorias} categorias sin resonancia. "
                f"Esto indica proliferacion descontrolada: rho={self.rho} es demasiado alto "
                f"para el ruido de los datos. Sugerencia: bajar rho."
            )
        self.top_down.append(list(entrada))
        return len(self.top_down) - 1

    # ---- persistencia ----------------------------------------------------
    def guardar(self, ruta_json, metadata_extra=None, preprocesamiento=None, sobrescribir=False):
        """Guarda la red (formato 2) con escritura atomica.
        No pisa un archivo existente salvo sobrescribir=True.
        preprocesamiento: dict COMPLETO con lo necesario para reproducir la entrada (version, etapa,
        segmentador, cols_features en orden, reglas, umbrales exactos, tam_canvas...). Con el se guarda un
        modelo para FOTOGRAFIAS; con None, un modelo de VECTORES: sirve para clasificar filas de un CSV
        binario pero no sabe procesar imagenes (probar_imagen.py lo rechaza). Un preprocesamiento
        incompleto o incompatible con las variables del modelo se rechaza ANTES de escribir."""
        if os.path.exists(ruta_json) and not sobrescribir:
            raise FileExistsError(
                f"'{ruta_json}' ya existe; no se sobrescribe sin autorizacion explicita")
        meta = dict(metadata_extra or {})
        datos = {
            "formato_version": FORMATO_MODELO,
            "n_features": self.n,
            "rho": self.rho,
            "L": self.L,
            "max_categorias": self.max_categorias,
            "politica_nulo": self.politica_nulo,
            "tipo_modelo": "fotografias" if preprocesamiento is not None else "vectores",
            "top_down": self.top_down,
            "preprocesamiento": preprocesamiento,
            "metadata_extra": meta,
        }
        val.validar_modelo_dict(json.loads(json.dumps(datos)), ruta=ruta_json)
        carpeta = os.path.dirname(os.path.abspath(ruta_json))
        os.makedirs(carpeta, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=carpeta, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(datos, f, indent=2, ensure_ascii=False)
            os.replace(tmp, ruta_json)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    @classmethod
    def cargar(cls, ruta_json):
        """Carga una red ART1 desde JSON (formato 1 entregado o formato 2).
        Devuelve (red, metadata_extra), como en el primer avance. Para acceder
        al preprocesamiento y a las advertencias use cargar_modelo()."""
        m = cargar_modelo(ruta_json)
        return m.red, m.meta


class Modelo:
    """Modelo cargado: red + metadatos + descripcion del preprocesamiento."""

    def __init__(self, red, meta, preprocesamiento, formato, advertencias, ruta, tipo="fotografias"):
        self.tipo = tipo                          # 'fotografias' | 'vectores' (el formato 1 es de fotografias)
        self.red = red
        self.meta = meta
        self.preprocesamiento = preprocesamiento  # None en formato 1 (legado)
        self.formato = formato
        self.advertencias = advertencias
        self.ruta = ruta

    @property
    def legado(self):
        return self.formato == 1

    @property
    def cols_features(self):
        return self.meta["cols_features"]

    def categoria(self, cid):
        """Informacion semantica de una categoria: dict con etiqueta (None si no
        tiene), estado, n y pureza_entrenamiento."""
        info = (self.meta.get("mapeo_clusters") or {}).get(str(cid))
        if info is None:
            return {"etiqueta": None, "estado": "sin_informacion", "n": 0, "pureza": None}
        etiqueta = info.get("etiqueta")
        if isinstance(etiqueta, str) and etiqueta.startswith("Cluster_"):
            etiqueta = None  # marcador generico del formato 1 cuando no habia etiquetas
        return {"etiqueta": etiqueta, "estado": info.get("estado", "etiquetada_legado"),
                "n": info.get("n", 0), "pureza": info.get("pureza"),
                "distribucion": info.get("distribucion")}


def cargar_modelo(ruta_json):
    """Carga y VALIDA un modelo. Estrategia de compatibilidad:
      - formato 2: se valida completo (incluye preprocesamiento).
      - formato 1 (modelos entregados, sin 'formato_version'): se carga en modo
        legado con el comportamiento original (politica_nulo='categoria_propia',
        etiquetas tal como fueron guardadas) y una advertencia explicita; el
        archivo no se modifica. Para pasarlo a formato 2 sin reentrenar:
        --migrar-modelo (genera un archivo nuevo)."""
    if not os.path.isfile(ruta_json):
        raise ErrorModelo("no se encontro el archivo de modelo", archivo=ruta_json)
    try:
        with open(ruta_json, "r", encoding="utf-8") as f:
            datos = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ErrorModelo(f"JSON invalido ({e})", archivo=ruta_json)
    version = val.validar_modelo_dict(datos, ruta=ruta_json)
    red = RedART1(
        n_features=datos["n_features"], rho=datos["rho"], L=datos.get("L", 2.0),
        max_categorias=datos.get("max_categorias", 50),
        politica_nulo=datos.get("politica_nulo", "categoria_propia"))
    red.top_down = [list(t) for t in datos["top_down"]]
    meta = datos.get("metadata_extra", {})
    adv = []
    if version == 1:
        adv.append("modelo en formato 1 (entregado): no registra el preprocesamiento ni "
                   "las versiones; se usan las constantes del primer avance y el etiquetado "
                   "de categorias tal como se guardo (asignaciones online). Ver "
                   "docs/informe_cambios.md, seccion compatibilidad")
    pre = datos.get("preprocesamiento")
    tipo = datos.get("tipo_modelo", "fotografias" if (version == 1 or isinstance(pre, dict)) else "vectores")
    return Modelo(red, meta, pre, version, adv, ruta_json, tipo)


# ---------------------------------------------------------------------------
# Entrenamiento y etiquetado posterior
# ---------------------------------------------------------------------------
def entrenar(red, entradas, epocas=1, orden="original", semilla=None):
    """Presenta las entradas a la red en aprendizaje rapido.
    epocas: maximo de pasadas; se corta antes si una pasada completa no cambia
    ningun prototipo ni crea categorias ('estable'). Con epocas=1 es la corrida
    de una pasada del primer avance.
    orden: 'original' (el del CSV) o 'aleatorio' (permutacion con semilla).
    Devuelve dict con el orden usado, las asignaciones online de la primera
    pasada (indexadas por posicion en 'entradas') y datos de convergencia."""
    indices = list(range(len(entradas)))
    if orden == "aleatorio":
        random.Random(semilla).shuffle(indices)
    elif orden != "original":
        raise ValueError("orden debe ser 'original' o 'aleatorio'")
    online = [None] * len(entradas)
    estable, realizadas = False, 0
    for ep in range(max(1, epocas)):
        antes = [list(t) for t in red.top_down]
        for i in indices:
            c = red.clasificar(entradas[i], aprender=True)
            if ep == 0:
                online[i] = c
        realizadas = ep + 1
        if red.top_down == antes:
            estable = True
            break
    return {"orden": indices, "asignaciones_online": online, "epocas_realizadas": realizadas,
            "estable": estable}


def construir_mapeo(red, entradas, etiquetas, desempate="indeterminada", soporte_min=1):
    """Etiquetado posterior de categorias con los PROTOTIPOS FINALES.

    Cada muestra de entrenamiento se vuelve a presentar en modo inferencia
    (red.evaluar, sin tocar pesos): la misma competencia y vigilancia que usara
    el programa. La etiqueta de una categoria es la mayoritaria entre las
    muestras que resuenan con ella. Casos explicitos:
      - muestra sin resonancia con los prototipos finales: no vota por ninguna
        categoria (se cuenta en 'sin_resonancia');
      - categoria sin muestras: etiqueta None, estado 'sin_muestras';
      - empate de etiquetas: 'indeterminada' (default) deja la categoria sin
        etiqueta (estado 'empate'); 'mayoritaria_global' desempata por la
        frecuencia de la etiqueta en todo el entrenamiento (y luego por orden
        alfabetico).
      - soporte_min: una categoria con menos de soporte_min muestras (con los prototipos finales) NO lleva etiqueta
        (estado 'sin_soporte'): evita nombrar con una sola muestra. Por defecto 1 = sin restriccion.
    Devuelve (mapeo_clusters, reinferencia)."""
    por_cat = {j: Counter() for j in range(len(red.top_down))}
    sin_res = Counter()
    for x, et in zip(entradas, etiquetas):
        r = red.evaluar(x)
        if r["resuena"]:
            por_cat[r["ganador"]][et] += 1
        else:
            sin_res[et] += 1
    global_ = Counter(etiquetas)
    mapeo = {}
    for j, cnt in por_cat.items():
        n = sum(cnt.values())
        info = {"n": n, "etiqueta": None, "estado": "sin_muestras", "pureza": None,
                "distribucion": dict(sorted(cnt.items()))}
        if n:
            maximo = max(cnt.values())
            empatadas = sorted(e for e, c in cnt.items() if c == maximo)
            info["pureza"] = round(maximo / n, 4)
            if n < soporte_min:
                info.update(estado="sin_soporte", etiqueta_candidata=empatadas[0] if len(empatadas) == 1 else None)
            elif len(empatadas) == 1:
                info.update(etiqueta=empatadas[0], estado="etiquetada")
            elif desempate == "mayoritaria_global":
                elegida = min(empatadas, key=lambda e: (-global_[e], e))
                info.update(etiqueta=elegida, estado="etiquetada_por_desempate",
                            empatadas=empatadas)
            else:
                info.update(estado="empate", empatadas=empatadas)
        mapeo[str(j)] = info
    n_res = sum(sum(c.values()) for c in por_cat.values())
    reinf = {"n": len(etiquetas), "n_resuenan": n_res,
             "n_sin_resonancia": len(etiquetas) - n_res,
             "distribucion_sin_resonancia": dict(sorted(sin_res.items()))}
    return mapeo, reinf


def mapeo_online(asignaciones, etiquetas):
    """Etiquetado del primer avance (se conserva SOLO para comparar/ablar):
    mayoria entre las asignaciones ONLINE, hechas mientras los prototipos
    todavia cambiaban. No coincide con lo que hace la inferencia final."""
    mapeo = {}
    por_cat = {}
    for c, et in zip(asignaciones, etiquetas):
        por_cat.setdefault(c, []).append(et)
    for c in sorted(por_cat):
        mayoria, k = Counter(por_cat[c]).most_common(1)[0]
        mapeo[str(c)] = {"n": len(por_cat[c]), "etiqueta": mayoria,
                         "estado": "etiquetada_online", "pureza": round(k / len(por_cat[c]), 4),
                         "distribucion": dict(sorted(Counter(por_cat[c]).items()))}
    return mapeo


def pureza(asignaciones, etiquetas):
    """Pureza global = suma de la clase mayoritaria de cada categoria / N.
    Las asignaciones None (sin resonancia) se excluyen; devuelve (pureza, cobertura)."""
    por_cat = {}
    for c, et in zip(asignaciones, etiquetas):
        if c is not None:
            por_cat.setdefault(c, Counter())[et] += 1
    n = sum(sum(c.values()) for c in por_cat.values())
    if not n:
        return 0.0, 0.0
    return sum(max(c.values()) for c in por_cat.values()) / n, n / len(etiquetas)


def hash_archivo(ruta):
    """SHA-256 del contenido con saltos de linea normalizados (estable ante
    CRLF/LF por autocrlf de git)."""
    with open(ruta, "rb") as f:
        contenido = f.read().replace(b"\r\n", b"\n")
    return hashlib.sha256(contenido).hexdigest()


def entrenar_desde_dataset(ds, rho, L=2.0, max_categorias=50, epocas=1, orden="original",
                           semilla=None, politica_nulo="categoria_propia",
                           desempate="indeterminada", modo_mapeo="prototipos_finales", soporte_min=1):
    """Entrena sobre un Dataset (validacion.Dataset). Devuelve (red, metadata_extra)."""
    if modo_mapeo not in MODOS_MAPEO:
        raise ValueError(f"modo_mapeo debe ser uno de {MODOS_MAPEO}")
    if desempate not in DESEMPATES:
        raise ValueError(f"desempate debe ser uno de {DESEMPATES}")
    red = RedART1(len(ds.cols_features), rho, L=L, max_categorias=max_categorias,
                  politica_nulo=politica_nulo)
    info = entrenar(red, ds.entradas, epocas=epocas, orden=orden, semilla=semilla)
    etiquetas = ds.etiquetas
    meta = {
        "cols_features": list(ds.cols_features),
        "cols_metadata": [ds.columna_etiqueta] if ds.columna_etiqueta else [],
        "n_muestras_entrenamiento": len(ds),
    }
    entren = {"csv": os.path.basename(ds.ruta) if ds.ruta else None,
              "csv_sha256": hash_archivo(ds.ruta) if ds.ruta and os.path.isfile(ds.ruta) else None,
              "esquema": ds.esquema, "orden": orden, "semilla": semilla,
              "epocas_pedidas": epocas, "epocas_realizadas": info["epocas_realizadas"],
              "estable": info["estable"], "modo_mapeo": modo_mapeo, "desempate": desempate}
    if etiquetas is not None:
        if modo_mapeo == "online":
            meta["mapeo_clusters"] = mapeo_online(info["asignaciones_online"], etiquetas)
            reinf = None
        else:
            meta["mapeo_clusters"], reinf = construir_mapeo(red, ds.entradas, etiquetas, desempate, soporte_min)
        if reinf is not None:
            meta["reinferencia_entrenamiento"] = reinf
        p_on, cob_on = pureza(info["asignaciones_online"], etiquetas)
        finales = [r["ganador"] if r["resuena"] else None for r in map(red.evaluar, ds.entradas)]
        p_fin, cob_fin = pureza(finales, etiquetas)
        entren.update(pureza_online=round(p_on, 4), pureza_final=round(p_fin, 4),
                      cobertura_final=round(cob_fin, 4),
                      n_por_etiqueta=dict(sorted(Counter(etiquetas).items())),
                      col_referencia=ds.columna_etiqueta)
    meta["entrenamiento"] = entren
    return red, meta


def entrenar_modelo(ruta_csv, rho, L=2.0, max_categorias=50, columna_id="fruta_id",
                    esquema="auto", col_etiqueta=None, cols_metadata=None, **kwargs):
    """Entrena la red ART1 sobre un CSV validado y genera el mapeo semantico de
    cada cluster hacia la etiqueta de referencia (ej. 'citrico', 'premium').
    Devuelve (red, metadata_extra). kwargs: epocas, orden, semilla,
    politica_nulo, desempate, modo_mapeo (ver entrenar_desde_dataset)."""
    ds = val.leer_dataset(ruta_csv, esquema=esquema, columna_id=columna_id,
                          col_etiqueta=col_etiqueta, cols_metadata=cols_metadata)
    return entrenar_desde_dataset(ds, rho, L=L, max_categorias=max_categorias, **kwargs)


def leer_csv(ruta, columna_id="fruta_id", esquema="auto", col_etiqueta=None, cols_metadata=None):
    """Compatibilidad con el primer avance: devuelve (ids, entradas, metadata,
    cols_features, cols_metadata). Ahora valida con esquema; ver
    validacion.leer_dataset. metadata incluye la columna de etiqueta."""
    ds = val.leer_dataset(ruta, esquema=esquema, columna_id=columna_id,
                          col_etiqueta=col_etiqueta, cols_metadata=cols_metadata)
    cols_meta = ([ds.columna_etiqueta] if ds.columna_etiqueta else []) + list(ds.cols_metadata)
    metadata = []
    for i in range(len(ds)):
        fila = dict(ds.metadata[i])
        if ds.columna_etiqueta:
            fila[ds.columna_etiqueta] = ds.etiquetas[i]
        metadata.append(fila)
    return ds.ids, ds.entradas, metadata, ds.cols_features, cols_meta


def ejecutar(ruta_csv, rho, L=2.0, max_categorias=50, columna_id="fruta_id", esquema="auto",
             col_etiqueta=None, cols_metadata=None, epocas=1, orden="original", semilla=None,
             politica_nulo="categoria_propia", desempate="indeterminada", dataset=None):
    """Corrida de agrupamiento (sin guardar modelo). Devuelve un dict; con
    exito False la corrida fue infructuosa (proliferacion)."""
    ds = dataset or val.leer_dataset(ruta_csv, esquema=esquema, columna_id=columna_id,
                                     col_etiqueta=col_etiqueta, cols_metadata=cols_metadata)
    red = RedART1(len(ds.cols_features), rho, L=L, max_categorias=max_categorias,
                  politica_nulo=politica_nulo)
    try:
        info = entrenar(red, ds.entradas, epocas=epocas, orden=orden, semilla=semilla)
    except ProliferacionError as e:
        return {
            "exito": False, "motivo": str(e), "n_categorias": len(red.top_down),
            "n_total": len(ds), "rho": rho,
            "advertencias": ds.advertencias,
        }
    # Las dos asignaciones: ONLINE (durante el aprendizaje) y FINAL (re-evaluando
    # con los prototipos finales); la pureza online es la del primer avance.
    resultado = {
        "exito": True, "n_categorias": len(red.top_down), "ids": ds.ids,
        "asignaciones": info["asignaciones_online"], "red": red,
        "etiquetas": ds.etiquetas, "col_etiqueta": ds.columna_etiqueta,
        "cols_features": ds.cols_features, "rho": rho, "estable": info["estable"],
        "epocas_realizadas": info["epocas_realizadas"], "advertencias": ds.advertencias,
        "cols_metadata": ([ds.columna_etiqueta] if ds.columna_etiqueta else []) + list(ds.cols_metadata),
    }
    if ds.etiquetas is not None:
        finales = [r["ganador"] if r["resuena"] else None for r in map(red.evaluar, ds.entradas)]
        resultado["asignaciones_finales"] = finales
        resultado["pureza_online"], _ = pureza(info["asignaciones_online"], ds.etiquetas)
        resultado["pureza_final"], resultado["cobertura_final"] = pureza(finales, ds.etiquetas)
    return resultado


def imprimir_reporte(resultado, archivo=None):
    out = archivo or sys.stdout
    if not resultado["exito"]:
        print(f"\nCORRIDA INFRUCTUOSA (rho={resultado['rho']}): {resultado['motivo']}", file=out)
        print(f"Se crearon {resultado['n_categorias']} categorias antes de cortar "
              f"(registros: {resultado['n_total']}).\n", file=out)
        return

    print(f"\nCorrida exitosa - rho={resultado['rho']} - {resultado['n_categorias']} clusters formados"
          f" ({resultado['epocas_realizadas']} pasada(s))\n", file=out)
    for adv in resultado.get("advertencias", []):
        print(f"  AVISO: {adv}", file=out)
    etiquetas = resultado["etiquetas"]
    asign = resultado["asignaciones"]
    conteo = Counter(asign)
    for cluster in sorted(conteo):
        n = conteo[cluster]
        linea = f"  Cluster {cluster}: {n} frutas"
        if etiquetas is not None:
            reales = [etiquetas[i] for i, c in enumerate(asign) if c == cluster]
            mayoria, n_mayoria = Counter(reales).most_common(1)[0]
            linea += f" | mayoria real: '{mayoria}' ({100 * n_mayoria / n:.0f}% pureza online)"
        print(linea, file=out)
    if etiquetas is not None:
        print(f"\n  Pureza online (asignaciones durante el aprendizaje): "
              f"{100 * resultado['pureza_online']:.2f}%", file=out)
        print(f"  Pureza con prototipos finales (reinferencia): "
              f"{100 * resultado['pureza_final']:.2f}% sobre "
              f"{100 * resultado['cobertura_final']:.1f}% de las muestras que resuenan", file=out)
        print("  (La pureza describe la mezcla de etiquetas dentro de cada categoria en "
              "ENTRENAMIENTO; no es la exactitud de prueba ni una confianza por fruta.)", file=out)


# ---------------------------------------------------------------------------
# Migracion de modelos de formato 1 (sin reentrenar)
# ---------------------------------------------------------------------------
def migrar_modelo_v1(ruta_v1, ruta_csv, ruta_salida, preprocesamiento, desempate="indeterminada"):
    """Convierte un modelo de formato 1 a formato 2 SIN reentrenar: los pesos
    (prototipos) se conservan tal cual; las etiquetas se recalculan con esos
    prototipos finales usando el CSV de entrenamiento, y se registra el
    preprocesamiento. Escribe un archivo nuevo (nunca pisa el original)."""
    m = cargar_modelo(ruta_v1)
    if not m.legado:
        raise ErrorModelo("el modelo ya esta en formato 2", archivo=ruta_v1)
    ds = val.leer_dataset(ruta_csv)
    entradas = val.reordenar_a_modelo(ds, m.cols_features)
    meta = dict(m.meta)
    meta["mapeo_clusters"], meta["reinferencia_entrenamiento"] = construir_mapeo(
        m.red, entradas, ds.etiquetas, desempate)
    meta["migracion"] = {"desde_formato": 1, "origen": os.path.basename(ruta_v1),
                         "csv": os.path.basename(ruta_csv), "csv_sha256": hash_archivo(ruta_csv),
                         "nota": "pesos conservados; etiquetas recalculadas con prototipos finales"}
    pre = dict(preprocesamiento)
    pre["cols_features"] = list(m.cols_features)
    m.red.guardar(ruta_salida, meta, preprocesamiento=pre, sobrescribir=False)


def resolver_preprocesamiento(opcion, ds, ruta_csv, archivo=None):
    """Preprocesamiento que se adjunta al modelo guardado desde un CSV. Devuelve (pre|None, origen).
      - ruta a un CSV de preprocesamiento (extraccion.leer_preprocesamiento_csv): se usa y se valida contra
        las variables del dataset (nombres y orden);
      - 'legado': el del primer avance; SOLO si el dataset tiene exactamente las variables comprometidas;
      - 'ninguno': modelo de vectores, sin preprocesamiento;
      - omitido: se busca preprocesamiento_<esquema>.csv junto al dataset; si no esta, modelo de VECTORES.
    Nunca se supone que un CSV fue generado con las constantes del primer avance."""
    out = archivo or sys.stdout
    from extraccion import leer_preprocesamiento_csv, preprocesamiento_legado
    if opcion == "ninguno":
        return None, "ninguno (explicito)"
    if opcion == "legado":
        if ds.esquema not in val.ESQUEMAS:
            raise ErrorEntrada("'legado' solo existe para los esquemas conocidos")
        base = list(val.ESQUEMAS[ds.esquema].features_base)
        if list(ds.cols_features) != base:
            raise ErrorEntrada(f"'legado' (preprocesamiento del primer avance) define {base}, pero el dataset trae "
                               f"{list(ds.cols_features)}: indique el CSV de preprocesamiento que lo genero",
                               archivo=ruta_csv)
        return preprocesamiento_legado(ds.esquema), "legado (declarado con --preprocesamiento legado)"
    ruta = opcion
    if ruta is None and ds.esquema in val.ESQUEMAS:
        candidata = os.path.join(os.path.dirname(os.path.abspath(ruta_csv)), f"preprocesamiento_{ds.esquema}.csv")
        if os.path.isfile(candidata):
            ruta = candidata
            print(f"(preprocesamiento tomado de {os.path.relpath(candidata)})", file=out)
    if ruta is None:
        print("AVISO: no se indico --preprocesamiento ni existe preprocesamiento_<esquema>.csv junto al dataset: "
              "se guarda un modelo de VECTORES (no procesa fotografias).", file=out)
        return None, "ninguno (no se indico)"
    pre = leer_preprocesamiento_csv(ruta, cols_features=ds.cols_features)
    if ds.esquema in val.ESQUEMAS and pre["etapa"] != ds.esquema:
        raise ErrorEntrada(f"el preprocesamiento es de la etapa '{pre['etapa']}' y el dataset del esquema "
                           f"'{ds.esquema}'", archivo=ruta)
    return pre, os.path.basename(ruta)


def inferir_csv(ruta_modelo, ds, ruta_salida=None, archivo=None):
    """Inferencia sobre las filas de un dataset con un modelo guardado (los pesos NO cambian).
    Levanta ErrorEntrada si las variables del CSV no coinciden con las del modelo. Una muestra sin
    resonancia, o cuya categoria no tiene etiqueta, cuenta como 'sin identificar' (error en la exactitud)."""
    out = archivo or sys.stdout
    m = cargar_modelo(ruta_modelo)
    entradas = val.reordenar_a_modelo(ds, m.cols_features)
    huella = m.red.huella()
    for adv in m.advertencias:
        print(f"AVISO: {adv}", file=out)
    filas, sin_res, sin_etiqueta = [], 0, 0
    for i, x in enumerate(entradas):
        ev = m.red.evaluar(x)
        info = m.categoria(ev["ganador"]) if ev["resuena"] else None
        pred = info["etiqueta"] if info else None
        sin_res += not ev["resuena"]
        sin_etiqueta += bool(ev["resuena"] and pred is None)
        filas.append({"fruta_id": ds.ids[i], "categoria": ev["ganador"] if ev["resuena"] else "",
                      "etiqueta_predicha": pred or "", "coincidencia": round(ev["ratio"], 4),
                      "estado": ("novedad" if not ev["resuena"] else info["estado"]),
                      "etiqueta_real": ds.etiquetas[i] if ds.etiquetas else ""})
    assert m.red.huella() == huella, "la inferencia modifico los pesos"
    print(f"\nInferencia con {os.path.basename(ruta_modelo)} (formato {m.formato}, rho={m.red.rho}, "
          f"{len(m.red.top_down)} categorias) sobre {len(ds)} muestras; pesos sin cambios ({huella}).", file=out)
    print(f"  Sin resonancia (novedad): {sin_res}; con resonancia pero categoria sin etiqueta: {sin_etiqueta}", file=out)
    if ds.etiquetas:
        from evaluacion import metricas
        clases = sorted(set(ds.etiquetas))
        pred = [f["etiqueta_predicha"] or None for f in filas]
        print(f"  Exactitud: {100 * metricas.exactitud(ds.etiquetas, pred):.1f}% "
              f"(novedades y categorias sin etiqueta cuentan como error)  F1 macro: "
              f"{metricas.f1_macro(ds.etiquetas, pred, clases):.3f}", file=out)
        conf = metricas.matriz_confusion(ds.etiquetas, pred, clases)
        cols = clases + [metricas.SIN_ETIQUETA]
        print("  Matriz de confusion (filas = real, columnas = predicha):", file=out)
        print("    " + " | ".join(f"{c:>14}" for c in [""] + cols), file=out)
        for c in clases:
            print("    " + " | ".join(f"{v:>14}" for v in [c] + [conf[c].get(k, 0) for k in cols]), file=out)
        print("  (Si el CSV es de entrenamiento esto es REINFERENCIA, no una evaluacion de prueba.)", file=out)
    if ruta_salida:
        os.makedirs(os.path.dirname(os.path.abspath(ruta_salida)), exist_ok=True)
        with open(ruta_salida, "w", newline="", encoding="utf-8") as f:
            import csv
            w = csv.DictWriter(f, fieldnames=list(filas[0]))
            w.writeheader()
            w.writerows(filas)
        print(f"  Predicciones escritas en: {ruta_salida}", file=out)
    return COD_OK


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _fallo(mensaje, codigo, mostrar_ayuda=True):
    print(f"\nERROR: {mensaje}", file=sys.stderr)
    if mostrar_ayuda:
        print("\nAyuda: python CarGross.py --help   (opciones y ejemplos)\n"
              "       codigos de salida: 1 corrida infructuosa, 2 argumentos, "
              "3 entrada invalida, 4 error interno", file=sys.stderr)
    return codigo


def _parser():
    p = argparse.ArgumentParser(
        prog="CarGross.py", add_help=False,
        description="Red ART1 (Carpenter-Grossberg) para clasificacion de frutas",
        epilog="Ejemplos:\n  python CarGross.py --csv datasets/dataset_morfologia.csv --rho 0.8\n"
               "  python CarGross.py --csv datasets/dataset_calidad_real.csv\n"
               "      (sin --rho: usa config/parametros_etapas.csv)\n"
               "  python CarGross.py --csv mis_datos.csv --esquema generico --col-etiqueta clase",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-h", "--help", "--ayuda", action="help", help="muestra esta ayuda y sale")
    p.add_argument("--csv", required=True, help="ruta al dataset CSV de entrada")
    p.add_argument("--rho", type=float, default=None,
                   help="vigilancia (0,1]; si se omite se toma de config/parametros_etapas.csv "
                        "(solo con esquemas conocidos)")
    p.add_argument("--L", type=float, default=None, help="parametro de aprendizaje rapido (default: config o 2.0)")
    p.add_argument("--max-categorias", type=int, default=None,
                   help="tope antes de declarar corrida infructuosa (default: config o 50)")
    p.add_argument("--id-col", default=None, help="columna identificadora (default fruta_id)")
    p.add_argument("--esquema", default="auto", choices=("auto",) + val.NOMBRES_ESQUEMA,
                   help="esquema del CSV: auto deduce entre los conocidos; 'generico' acepta "
                        "otros CSV y exige --col-etiqueta si hay etiqueta de referencia")
    p.add_argument("--col-etiqueta", help="(solo generico) columna de etiqueta de referencia")
    p.add_argument("--col-meta", action="append", default=[],
                   help="(solo generico) columna de metadatos, repetible; nunca entra a la red")
    p.add_argument("--epocas", type=int, default=None, help="maximo de pasadas (default: config o 1)")
    p.add_argument("--orden", choices=("original", "aleatorio"), default="original")
    p.add_argument("--semilla", type=int, default=None, help="semilla para --orden aleatorio")
    p.add_argument("--politica-nulo", choices=POLITICAS_NULO, default=None,
                   help="tratamiento del vector nulo (default: config o categoria_propia)")
    p.add_argument("--desempate", choices=DESEMPATES, default=None,
                   help="desempate de etiquetas al guardar el modelo (default: config o indeterminada)")
    p.add_argument("--mapeo", choices=MODOS_MAPEO, default="prototipos_finales",
                   help="'online' reproduce el etiquetado del primer avance (solo comparacion)")
    p.add_argument("--guardar-modelo", help="ruta JSON donde guardar el modelo entrenado (formato 2)")
    p.add_argument("--sobrescribir", action="store_true",
                   help="permite pisar --guardar-modelo si ya existe (por defecto se rechaza)")
    p.add_argument("--inferir-con", metavar="MODELO",
                   help="no entrena: clasifica las filas del CSV con un modelo guardado, SIN modificar sus pesos; "
                        "valida que las variables del CSV coincidan con las del modelo")
    p.add_argument("--salida-predicciones", metavar="CSV",
                   help="con --inferir-con: escribe una fila por muestra (categoria, etiqueta, coincidencia)")
    p.add_argument("--preprocesamiento", metavar="CSV|legado|ninguno",
                   help="con --guardar-modelo: preprocesamiento que produjo el CSV (ruta a un "
                        "preprocesamiento_<etapa>.csv; 'legado' = el del primer avance, solo si el dataset trae "
                        "exactamente sus variables; 'ninguno' = modelo de vectores). Si se omite se busca "
                        "preprocesamiento_<esquema>.csv junto al dataset; si no existe, se guarda un modelo de "
                        "vectores (nunca se supone v1)")
    p.add_argument("--migrar-modelo", metavar="MODELO_V1",
                   help="convierte un modelo de formato 1 usando --csv y --guardar-modelo, sin reentrenar")
    p.add_argument("--depurar", action="store_true", help="muestra el traceback de errores internos")
    return p


def main(argv=None):
    try:  # consola de Windows (cp1252): no fallar por caracteres no ASCII
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass
    parser = _parser()
    args = parser.parse_args(argv)  # argparse sale con codigo 2 ante argumentos invalidos

    try:
        import configuracion
        ds = val.leer_dataset(args.csv, esquema=args.esquema, columna_id=args.id_col,
                              col_etiqueta=args.col_etiqueta, cols_metadata=args.col_meta)
        for adv in ds.advertencias:
            print(f"AVISO: {adv}")

        if args.inferir_con:
            return inferir_csv(args.inferir_con, ds, args.salida_predicciones)

        base = {}
        if ds.esquema in val.ESQUEMAS:
            base = configuracion.parametros_etapa(ds.esquema)
        elif args.rho is None:
            return _fallo("con --esquema generico hay que indicar --rho explicitamente", COD_USO)
        rho = args.rho if args.rho is not None else base["rho"]
        if args.rho is None:
            print(f"(rho={rho} tomado de config/parametros_etapas.csv, etapa '{ds.esquema}')")
        L = args.L if args.L is not None else base.get("L", 2.0)
        tope = args.max_categorias if args.max_categorias is not None else base.get("max_categorias", 50)
        epocas = args.epocas if args.epocas is not None else base.get("epocas", 1)
        nulo = args.politica_nulo or base.get("politica_nulo", "categoria_propia")
        desempate = args.desempate or base.get("desempate", "indeterminada")

        if args.migrar_modelo:
            if not args.guardar_modelo:
                return _fallo("--migrar-modelo requiere --guardar-modelo con la ruta del archivo nuevo", COD_USO)
            from extraccion import preprocesamiento_legado
            pre = preprocesamiento_legado(ds.esquema)
            migrar_modelo_v1(args.migrar_modelo, args.csv, args.guardar_modelo, pre, desempate)
            print(f"Modelo migrado a formato 2 (pesos conservados): {args.guardar_modelo}")
            return COD_OK

        if args.guardar_modelo and os.path.exists(args.guardar_modelo) and not args.sobrescribir:
            return _fallo(f"'{args.guardar_modelo}' ya existe. Use otra ruta o --sobrescribir "
                          "(no se pisa un modelo en silencio)", COD_USO)
        pre, pre_origen = None, None
        if args.guardar_modelo:   # se resuelve y VALIDA el preprocesamiento antes de entrenar o escribir nada
            pre, pre_origen = resolver_preprocesamiento(args.preprocesamiento, ds, args.csv)

        resultado = ejecutar(args.csv, rho, L=L, max_categorias=tope, epocas=epocas, orden=args.orden,
                             semilla=args.semilla, politica_nulo=nulo, desempate=desempate, dataset=ds)
        imprimir_reporte(resultado)
        if not resultado["exito"]:
            return _fallo("corrida infructuosa: " + resultado["motivo"], COD_INFRUCTUOSA)

        if args.guardar_modelo:
            red, meta = entrenar_desde_dataset(ds, rho, L=L, max_categorias=tope, epocas=epocas,
                                               orden=args.orden, semilla=args.semilla,
                                               politica_nulo=nulo, desempate=desempate,
                                               modo_mapeo=args.mapeo)
            if pre is not None:
                meta["preprocesamiento_origen"] = pre_origen
            red.guardar(args.guardar_modelo, meta, preprocesamiento=pre, sobrescribir=args.sobrescribir)
            tipo = "fotografias" if pre is not None else "vectores (sin preprocesamiento: no procesa fotos)"
            print(f"\nModelo guardado en: {args.guardar_modelo} (formato {FORMATO_MODELO}, tipo {tipo})")
            print(f"Total de clusters comprometidos: {len(red.top_down)}")
            for c_id, info in meta.get("mapeo_clusters", {}).items():
                et = info["etiqueta"] or f"({info['estado']})"
                pur = "-" if info["pureza"] is None else f"{info['pureza'] * 100:.0f}%"
                print(f"  Cluster {c_id}: {et} ({info['n']} muestras con prototipos finales, "
                      f"{pur} pureza de entrenamiento)")
        return COD_OK
    except ErrorEntrada as e:
        return _fallo(str(e), COD_ENTRADA)
    except FileExistsError as e:
        return _fallo(str(e), COD_USO)
    except VectorNuloError as e:
        return _fallo(f"el dataset contiene vectores nulos incompatibles con la politica elegida: {e}", COD_ENTRADA)
    except ValueError as e:
        return _fallo(f"configuracion invalida: {e}", COD_USO)
    except ProliferacionError as e:
        return _fallo(f"corrida infructuosa: {e}", COD_INFRUCTUOSA)
    except Exception as e:  # noqa: BLE001 - ultimo recurso: codigo distinto de cero + diagnostico
        if args.depurar:
            raise
        return _fallo(f"error interno inesperado ({type(e).__name__}: {e}). "
                      "Repita con --depurar para ver el traceback", COD_ERROR)


if __name__ == "__main__":
    sys.exit(main())
