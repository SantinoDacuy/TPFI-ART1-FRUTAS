"""
evaluacion/cadena.py
Evaluacion de las DOS ETAPAS ACOPLADAS sobre fotografias.

Etapa 1 (morfologia) decide el tipo: citrico / tropical / carozo (= manzana).
Solo si el tipo predicho es 'carozo' la foto se DERIVA a la etapa 2 (calidad),
igual que en probar_imagen.py. Se evalua tambien la etapa 2 aislada ("oraculo":
todas las manzanas reales, derivadas o no) para separar el error de cada etapa.

Una novedad (ninguna categoria resuena) o una categoria sin etiqueta cuentan
como "sin identificar": error en exactitud/F1, e informadas por separado.
"""
from validacion import ALIAS_MANZANA
from evaluacion import metricas

TIPOS = ["citrico", "tropical", "carozo"]
CALIDADES = ["premium", "comercial", "descarte"]


def _clasificar(modelo, vector):
    """(cluster|None, etiqueta|None, estado, ratio) en modo inferencia."""
    ev = modelo.red.evaluar(vector)
    if not ev["resuena"]:
        return None, None, "novedad", ev["ratio"]
    info = modelo.categoria(ev["ganador"])
    return ev["ganador"], info["etiqueta"], info["estado"], ev["ratio"]


def evaluar_cadena(fotos, vecs_morf, vecs_cal, modelo_morf, modelo_cal, derivacion_min=0.0):
    """fotos: dicts del manifiesto; vecs_*: {foto_id: vector binario en el orden
    de variables del modelo}. Devuelve un registro por foto (inferencia pura:
    no modifica pesos). derivacion_min: coincidencia |I^T|/|I| minima de la etapa 1 para derivar a calidad
    (0 = basta con resonar con una categoria etiquetada 'carozo')."""
    antes = (modelo_morf.red.huella(), modelo_cal.red.huella())
    regs = []
    for f in fotos:
        fid = f["foto_id"]
        c1, e1, s1, r1 = _clasificar(modelo_morf, vecs_morf[fid])
        reg = {"foto_id": fid, "estrato": f["estrato"], "grupo": f.get("grupo", ""),
               "particion": f.get("particion", ""), "tipo_real": f["tipo_real"],
               "calidad_real": f["calidad_real"], "morf_cluster": c1, "morf_etiqueta": e1,
               "morf_estado": s1, "morf_match": round(r1, 4),
               "derivada": e1 == ALIAS_MANZANA and r1 >= derivacion_min}
        if f["tipo_real"] == ALIAS_MANZANA:           # etapa 2 aislada ("oraculo")
            c2, e2, s2, r2 = _clasificar(modelo_cal, vecs_cal[fid])
            reg.update(orac_cluster=c2, orac_etiqueta=e2, orac_estado=s2, orac_match=round(r2, 4))
        if reg["derivada"]:                            # etapa 2 acoplada
            c2, e2, s2, r2 = _clasificar(modelo_cal, vecs_cal[fid])
            reg.update(cal_cluster=c2, cal_etiqueta=e2, cal_estado=s2, cal_match=round(r2, 4))
        regs.append(reg)
    assert antes == (modelo_morf.red.huella(), modelo_cal.red.huella()), "la evaluacion modifico pesos"
    return regs


def resumir_cadena(regs, mayoritaria_tipo=ALIAS_MANZANA, mayoritaria_calidad=None):
    """Metricas de las dos etapas y de la cadena. mayoritaria_*: clases mayoritarias
    del ENTRENAMIENTO (referencia sencilla de 'predecir siempre la mayoritaria')."""
    n = len(regs)
    if n == 0:
        return {"n": 0}
    verdad = [r["tipo_real"] for r in regs]
    pred = [r["morf_etiqueta"] for r in regs]
    manz = [r for r in regs if r["tipo_real"] == ALIAS_MANZANA]
    nom = [r for r in regs if r["tipo_real"] != ALIAS_MANZANA]
    deriv_ok = [r for r in manz if r["derivada"]]
    cadena_manz = [r for r in deriv_ok if r.get("cal_etiqueta") == r["calidad_real"]]
    cadena_todo = len(cadena_manz) + sum(r["morf_etiqueta"] == r["tipo_real"] for r in nom)
    res = {
        "n": n, "n_manzanas": len(manz), "n_no_manzanas": len(nom),
        # etapa 1
        "e1_exactitud": metricas.exactitud(verdad, pred),
        "e1_f1_macro": metricas.f1_macro(verdad, pred, TIPOS),
        "e1_tipo_correcto": sum(v == p for v, p in zip(verdad, pred)),
        "e1_novedades": sum(r["morf_estado"] == "novedad" for r in regs),
        "e1_sin_etiqueta": sum(r["morf_estado"] not in ("novedad",) and r["morf_etiqueta"] is None for r in regs),
        "e1_confusion": metricas.matriz_confusion(verdad, pred, TIPOS),
        "manzanas_derivadas": len(deriv_ok),
        "no_manzanas_derivadas": sum(r["derivada"] for r in nom),
        # cadena
        "cadena_manzanas_correctas": len(cadena_manz),
        "cadena_total_correcta": cadena_todo,
        "cadena_total_exactitud": cadena_todo / n,
        "cadena_manzanas_exactitud": len(cadena_manz) / len(manz) if manz else float("nan"),
        # etapa 2 acoplada (solo sobre manzanas derivadas)
        "e2_acoplada_n": len(deriv_ok),
        "e2_acoplada_correctas": len(cadena_manz),
        # etapa 2 aislada
        "e2_aislada_n": len(manz),
    }
    if manz:
        v2 = [r["calidad_real"] for r in manz]
        p2 = [r["orac_etiqueta"] for r in manz]
        res.update(e2_aislada_exactitud=metricas.exactitud(v2, p2),
                   e2_aislada_f1_macro=metricas.f1_macro(v2, p2, CALIDADES),
                   e2_aislada_novedades=sum(r["orac_estado"] == "novedad" for r in manz),
                   e2_aislada_sin_etiqueta=sum(r["orac_estado"] != "novedad" and r["orac_etiqueta"] is None for r in manz),
                   e2_aislada_confusion=metricas.matriz_confusion(v2, p2, CALIDADES))
        if mayoritaria_calidad:
            res["linea_base_calidad"] = metricas.linea_base_mayoritaria(v2, mayoritaria_calidad, CALIDADES)
            res["linea_base_cadena_manzanas"] = sum(r["calidad_real"] == mayoritaria_calidad for r in manz) / len(manz)
    res["linea_base_tipo"] = metricas.linea_base_mayoritaria(verdad, mayoritaria_tipo, TIPOS)
    return res
