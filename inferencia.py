"""
inferencia.py
Inferencia de dos etapas sobre UNA imagen, con modelos de formato 1 (entregados)
o formato 2. Devuelve estructuras de datos; probar_imagen.py las imprime.

Reglas de este modulo:
  - Inferir (aprender=False) NUNCA modifica pesos: solo se llama a RedART1.evaluar.
  - El preprocesamiento sale del propio modelo (segmentador, umbrales, orden de
    variables), no de constantes del codigo.
  - 'coincidencia' es |I ^ T| / |I| (cuanto del patron esta contenido en el
    prototipo). No es una probabilidad de acierto ni una confianza individual.
    'pureza_entrenamiento' describe la mezcla de etiquetas de la categoria en el
    entrenamiento; tampoco es una confianza por foto.
  - Una categoria sin etiqueta (empate, sin muestras o creada por aprendizaje)
    no se traduce a ninguna fruta o calidad: no se inventa un nombre.
  - Con aprender=True los pesos cambian EN MEMORIA y los eventos quedan en
    resultado['cambios']; guardar es una decision aparte (guardar_aprendizaje).
"""
import datetime
import os

import extraccion as X
import preprocesador as P
from validacion import ALIAS_MANZANA, ErrorEntrada

AVISO_ALCANCE = ("ART1 solo compara vectores binarios: un match alto no demuestra que la imagen sea una "
                 "fruta ni que la etiqueta sea correcta; no hay rechazo de objetos ajenos al dominio.")


def pre_de(modelo, etapa):
    """Configuracion de preprocesamiento del modelo (la legada si es formato 1)."""
    return modelo.preprocesamiento or X.preprocesamiento_legado(etapa)


def _medir(img, pre, cache):
    clave = (pre["segmentador"], pre.get("normalizacion", "ninguna"))
    if clave not in cache:
        cache[clave] = X.medir_foto(img, segmentador=clave[0], tam=pre.get("tam_canvas", P.TAM_CANVAS),
                                    normalizacion=clave[1])
    return cache[clave]


def _advertencias_segmentacion(med):
    adv = []
    if med.info_seg.get("advertencia"):
        adv.append(med.info_seg["advertencia"])
    elif P.puntaje_plausibilidad(med.mascara) < 0.35:
        adv.append("la mascara de segmentacion es poco plausible (muy grande, pegada al borde o irregular): "
                   "revise la imagen de diagnostico; los valores medidos pueden no corresponder a la fruta")
    return adv


def _describir_etapa(nombre, modelo, pre, med, aprender, nombre_imagen):
    vec = X.vectorizar(med, pre)
    red = modelo.red
    huella_antes = red.huella()
    ev = red.evaluar(vec)                               # competencia + vigilancia, sin tocar pesos
    assert red.huella() == huella_antes, "la evaluacion modifico los pesos"
    info = modelo.categoria(ev["ganador"]) if ev["resuena"] else None
    res = {
        "etapa": nombre, "vector": vec, "cols_features": list(pre["cols_features"]),
        "crudos": {pre["reglas"][c]["crudo"]: med.crudos[pre["reglas"][c]["crudo"]] for c in pre["cols_features"]},
        "umbrales": {c: pre["umbrales"][c] for c in pre["cols_features"]},
        "resuena": ev["resuena"], "cluster": ev["ganador"], "coincidencia": ev["ratio"],
        "motivo": ev.get("motivo"), "rho": red.rho, "n_categorias": len(red.top_down),
        "candidatos": ev.get("candidatos", []), "etiqueta": None, "estado_categoria": None,
        "pureza_entrenamiento": None, "n_entrenamiento": None, "cambios": [],
        "nulo": not any(vec),
    }
    if info is not None:
        res.update(etiqueta=info["etiqueta"], estado_categoria=info["estado"],
                   pureza_entrenamiento=info["pureza"], n_entrenamiento=info["n"])
    if aprender:
        antes = [list(t) for t in red.top_down]
        cid = red.clasificar(vec, aprender=True)       # ProliferacionError sube al llamador (codigo de salida 1)
        if cid < len(antes):
            evento = "prototipo_actualizado" if red.top_down[cid] != antes[cid] else "sin_cambio"
            cambio = {"evento": evento, "categoria": cid, "prototipo_antes": antes[cid],
                      "prototipo_despues": list(red.top_down[cid])}
        else:
            cambio = {"evento": "categoria_nueva", "categoria": cid, "prototipo_antes": None,
                      "prototipo_despues": list(red.top_down[cid])}
            modelo.meta.setdefault("mapeo_clusters", {})[str(cid)] = {
                "n": 0, "etiqueta": None, "estado": "nueva_sin_etiqueta", "pureza": None,
                "origen": f"aprendida en linea con '{os.path.basename(nombre_imagen)}'"}
        cambio.update(etapa=nombre, imagen=os.path.basename(nombre_imagen),
                      fecha=datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        res["cambios"].append(cambio)
        modelo.meta.setdefault("historial_aprendizaje", []).append(cambio)
        if cambio["evento"] == "categoria_nueva":
            res["cluster_nuevo"] = cid
    return res


def inferir_imagen(img, modelo_morf, modelo_cal, aprender=False, nombre_imagen=""):
    """Pipeline de dos etapas. Devuelve dict con 'etapa1', 'etapa2' (None si no se
    deriva), 'advertencias', 'cambios' y 'derivada'."""
    cache = {}
    for m, etapa in ((modelo_morf, "morfologia"), (modelo_cal, "calidad")):
        if m.tipo != "fotografias" or (m.preprocesamiento is None and not m.legado):
            raise ErrorEntrada(f"el modelo de {etapa} es de VECTORES (guardado desde un CSV sin preprocesamiento): "
                               "no sabe convertir una fotografia en su vector. Guardelo con "
                               "--preprocesamiento <preprocesamiento_etapa.csv>", archivo=m.ruta)
    pm, pc = pre_de(modelo_morf, "morfologia"), pre_de(modelo_cal, "calidad")
    med1 = _medir(img, pm, cache)
    adv = _advertencias_segmentacion(med1)
    if med1.mascara is None or not med1.mascara.any():
        raise ErrorEntrada("no se detecto ninguna region de fruta en la imagen (mascara vacia)")
    e1 = _describir_etapa("morfologia", modelo_morf, pm, med1, aprender, nombre_imagen)
    derivada = e1["etiqueta"] == ALIAS_MANZANA
    e2 = None
    if derivada:
        med2 = _medir(img, pc, cache)
        e2 = _describir_etapa("calidad", modelo_cal, pc, med2, aprender, nombre_imagen)
    if e1["nulo"]:
        adv.append("vector nulo en etapa 1: todas las variables en 0 (ver politica de vector nulo del modelo)")
    for m in (modelo_morf, modelo_cal):
        adv.extend(m.advertencias)
    return {"etapa1": e1, "etapa2": e2, "derivada": derivada, "advertencias": list(dict.fromkeys(adv)),
            "cambios": e1["cambios"] + (e2["cambios"] if e2 else []), "medicion": med1}


def guardar_aprendizaje(modelo, etapa, carpeta_destino, sobrescribir=False):
    """Guarda un modelo modificado por aprendizaje en un archivo NUEVO dentro de
    carpeta_destino. Nunca pisa el modelo de origen ni un archivo existente salvo
    sobrescribir=True. Los modelos de formato 1 deben migrarse antes."""
    if modelo.legado:
        raise ErrorEntrada("el modelo es de formato 1 (entregado); migrelo primero con "
                           "CarGross.py --migrar-modelo y vuelva a intentar")
    destino = os.path.join(carpeta_destino, f"art1_{etapa}.json")
    if os.path.abspath(destino) == os.path.abspath(modelo.ruta or ""):
        raise ErrorEntrada(f"'{destino}' es el modelo de origen: elija otra carpeta (el original no se sobrescribe)")
    modelo.red.guardar(destino, modelo.meta, preprocesamiento=modelo.preprocesamiento, sobrescribir=sobrescribir)
    return destino
