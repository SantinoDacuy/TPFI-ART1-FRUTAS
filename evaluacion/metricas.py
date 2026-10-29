"""
evaluacion/metricas.py
Metricas de clasificacion con tratamiento explicito de las "no respuestas".

Convencion: una prediccion None (novedad: ninguna categoria resuena) o una
categoria sin etiqueta cuentan como ERROR en exactitud y F1 (la fruta no fue
identificada) y se informan aparte como tasa de novedad. No se las oculta ni se
las excluye del denominador.
"""
import math
import random
from collections import Counter

SIN_ETIQUETA = "(sin etiqueta)"


def matriz_confusion(verdad, pred, clases):
    """{verdadera: {predicha: n}}, con la columna SIN_ETIQUETA para las no respuestas."""
    m = {c: Counter() for c in clases}
    for v, p in zip(verdad, pred):
        if v in m:
            m[v][p if p in clases else SIN_ETIQUETA] += 1
    return {c: dict(m[c]) for c in clases}


def exactitud(verdad, pred):
    return sum(v == p for v, p in zip(verdad, pred)) / len(verdad) if verdad else float("nan")


def f1_por_clase(verdad, pred, clases):
    out = {}
    for c in clases:
        tp = sum(v == c and p == c for v, p in zip(verdad, pred))
        fp = sum(v != c and p == c for v, p in zip(verdad, pred))
        fn = sum(v == c and p != c for v, p in zip(verdad, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[c] = {"precision": prec, "recall": rec,
                  "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                  "soporte": sum(v == c for v in verdad)}
    return out


def f1_macro(verdad, pred, clases):
    """Promedio simple de F1 sobre las clases que tienen soporte en 'verdad'."""
    por = f1_por_clase(verdad, pred, clases)
    usadas = [c for c in clases if por[c]["soporte"] > 0]
    return sum(por[c]["f1"] for c in usadas) / len(usadas) if usadas else float("nan")


def clase_mayoritaria(etiquetas):
    cnt = Counter(etiquetas)
    mayor = max(cnt.values())
    return sorted(c for c, k in cnt.items() if k == mayor)[0]  # empate: orden alfabetico


def linea_base_mayoritaria(verdad, clase, clases):
    """Exactitud y F1 macro de predecir SIEMPRE 'clase' (referencia sencilla)."""
    pred = [clase] * len(verdad)
    return {"clase": clase, "exactitud": exactitud(verdad, pred),
            "f1_macro": f1_macro(verdad, pred, clases)}


def ic_bootstrap_por_grupos(aciertos, grupos, repeticiones=2000, semilla=0, nivel=0.95):
    """Intervalo de confianza percentil de la exactitud remuestreando GRUPOS
    completos (las fotos de un grupo no son independientes). Devuelve
    (bajo, alto). Con pocos grupos el intervalo es ancho: es lo correcto."""
    por_grupo = {}
    for a, g in zip(aciertos, grupos):
        por_grupo.setdefault(g, []).append(a)
    claves = sorted(por_grupo)
    if len(claves) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(semilla)
    vals = []
    for _ in range(repeticiones):
        tot = ok = 0
        for _ in claves:
            lista = por_grupo[claves[rng.randrange(len(claves))]]
            tot += len(lista)
            ok += sum(lista)
        vals.append(ok / tot)
    vals.sort()
    k = int((1 - nivel) / 2 * repeticiones)
    return (vals[k], vals[repeticiones - 1 - k])


def media_y_desvio(valores):
    v = [x for x in valores if not (isinstance(x, float) and math.isnan(x))]
    if not v:
        return float("nan"), float("nan")
    m = sum(v) / len(v)
    return m, (math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else 0.0)
