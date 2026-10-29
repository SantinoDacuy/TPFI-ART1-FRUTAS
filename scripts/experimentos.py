"""
scripts/experimentos.py
Ablacion secuencial, barrido de rho / sensibilidad al orden y construccion de
los entregables finales (datasets v2, modelos v2 y resultados), todo sobre las
mediciones de resultados/mediciones/ (generadas por scripts/extraer_mediciones.py).

Subcomandos:
  ablacion   cadena de decisiones sobre VALIDACION (promedio de N semillas de particion)
  final      entrena los modelos finales con la configuracion elegida y evalua en prueba
  barrido    barrido de rho y sensibilidad al orden de presentacion
  fondo      diagnostico de confusion fondo-clase (clasificar mirando solo el entorno)
  todo       los cuatro, en ese orden

Regla de decision (declarada antes de mirar resultados): en cada paso se prueban las
opciones sobre la configuracion ya adoptada; la mejor se adopta si mejora la
exactitud de la cadena completa en VALIDACION (media entre semillas) en al menos
MARGEN (0.01 absoluto) y esa mejora supera 1 error estandar de la diferencia
pareada. Si no, se conserva la configuracion vigente ("sin evidencia suficiente").
Tres pasos son correcciones de metodo exigidas por el pedido y se adoptan siempre,
informando su efecto: mapeo con prototipos finales, cobertura de manzanas de todas
las calidades en la etapa 1 y eleccion de rho en validacion.
La prueba no interviene en ninguna decision.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import configuracion as cfg
from evaluacion import banco, experimentos as E
from validacion import COD_ENTRADA, COD_OK

SEMILLAS = tuple(range(30))
SEMILLA_PRINCIPAL = 0      # particion usada para matrices de confusion y modelos entregados
MARGEN = 0.01
CRITERIO = "cadena_total_exactitud"

_FOTOS = None
_MED = None


def _init(dir_med, dir_img):
    global _FOTOS, _MED
    _FOTOS = banco.inventariar(dir_img)
    _MED = {s: E.leer_mediciones(os.path.join(dir_med, f"mediciones_{s}.csv")) for s in ("v1", "v2")}


def _correr(args):
    var, semilla = args
    r = E.correr_split(var, _FOTOS, _MED[var.segmentador], semilla)
    for k in ("_modelos", "_pre"):
        r.pop(k, None)
    for p in banco.PARTICIONES:
        if p in r:
            r[p].pop("_registros", None)
    return r


def _ejecutar(ex, var, semillas):
    return list(ex.map(_correr, [(var, s) for s in semillas], chunksize=2))


# ---------------------------------------------------------------------------
def _pasos():
    """Lista de pasos: (id, descripcion, tipo, {opcion: funcion(config)->config}).
    La primera opcion de cada paso es siempre la configuracion vigente. Tipos:
      'obligatoria': correccion de metodo exigida por el pedido; se adopta la 2da opcion;
      'correccion' : arregla un defecto demostrado; se adopta la mejor opcion de las
                     restantes si no es peor que la vigente por mas que MARGEN (no inferior);
      'mejora'     : se adopta la mejor solo con evidencia (>= MARGEN y > 1 error estandar)."""
    return [
        ("S1", "mapeo de categorias a etiquetas", "obligatoria",
         {"online (original)": lambda v: v,
          "prototipos finales": lambda v: replace(v, mapeo="prototipos_finales")}),
        ("S2", "desempate de etiquetas", "mejora",
         {"indeterminada": lambda v: v,
          "mayoritaria_global": lambda v: replace(v, desempate="mayoritaria_global")}),
        ("S3", "cobertura de la etapa 1", "obligatoria",
         {"solo manzanas good (original)": lambda v: v,
          "manzanas de todas las calidades": lambda v: replace(v, cobertura="todas")}),
        ("S4", "eleccion de rho", "obligatoria",
         {"rho fijo del entregado": lambda v: v,
          "rho elegido en validacion": lambda v: replace(v, rho_morf=None, rho_cal=None)}),
        ("S5", "umbrales de calidad (calculados solo con entrenamiento)", "obligatoria",
         {"constantes de referencia redondeadas (del lote original)": lambda v: v,
          "medianas exactas del entrenamiento": lambda v: replace(v, umbral_cal="mediana")}),
        ("S6", "segmentacion", "mejora",
         {"v1 (saturacion + Otsu)": lambda v: v,
          "v2 (candidatos + puntaje)": lambda v: replace(v, segmentador="v2")}),
        ("S7", "medicion de tiene_tallo", "correccion",
         {"v1 (siempre 0 con OpenCV 5)": lambda v: v,
          "convexidad corregida": lambda v: replace(v, tallo="convexidad"),
          "protuberancia": lambda v: replace(v, tallo="protuberancia")}),
        ("S8", "desvio de tono (color_uniforme / color_homogeneo)", "correccion",
         {"lineal (v1)": lambda v: v,
          "circular ponderado": lambda v: replace(v, tono="circular")}),
        ("S9", "umbrales de morfologia", "mejora",
         {"fijos del primer avance": lambda v: v,
          "mediana del entrenamiento": lambda v: replace(v, umbral_morf="mediana"),
          "corte de Otsu del entrenamiento": lambda v: replace(v, umbral_morf="otsu")}),
        ("S10", "descriptor cromatico adicional en morfologia", "mejora",
         {"ninguno": lambda v: v,
          "tono_calido": lambda v: replace(v, extras_morf=("tono_calido",)),
          "saturacion_alta": lambda v: replace(v, extras_morf=("saturacion_alta",)),
          "ambos": lambda v: replace(v, extras_morf=("tono_calido", "saturacion_alta"))}),
        ("S11", "descriptor cromatico adicional en calidad", "mejora",
         {"ninguno": lambda v: v,
          "sin_oscurecimiento": lambda v: replace(v, extras_cal=("sin_oscurecimiento",))}),
        ("S12", "pasadas de entrenamiento", "mejora",
         {"una pasada": lambda v: v,
          "hasta estabilizar (max 10)": lambda v: replace(v, epocas=10)}),
        ("S13", "segmentacion (reconsideracion sobre la configuracion final)", "mejora",
         {"v1 (saturacion + Otsu)": lambda v: v,
          "v2 (candidatos + puntaje)": lambda v: replace(v, segmentador="v2")}),
    ]


def _fila(paso, opcion, res, base_res, adoptada, nota=""):
    f = {"paso": paso, "opcion": opcion, "adoptada": "si" if adoptada else "no", "nota": nota}
    for part in banco.PARTICIONES:
        ag = E.agregar(res, part)
        for k in ("e1_exactitud", "e1_f1_macro", "manzanas_derivadas", "no_manzanas_derivadas",
                  "e2_aislada_exactitud", "e2_aislada_f1_macro", "cadena_manzanas_exactitud",
                  "cadena_total_exactitud"):
            f[f"{part}_{k}"] = round(ag[k][0], 4)
            f[f"{part}_{k}_sd"] = round(ag[k][1], 4)
    ag = E.agregar(res, "validacion")
    f["rho_morfologia"], f["rho_calidad"] = round(ag["rho_morfologia"][0], 3), round(ag["rho_calidad"][0], 3)
    f["k_morfologia"], f["k_calidad"] = round(ag["k_morfologia"][0], 2), round(ag["k_calidad"][0], 2)
    for part in ("validacion", "prueba"):
        d, se = E.diferencia_pareada(base_res, res, part, CRITERIO)
        f[f"delta_{part}"], f[f"se_delta_{part}"] = round(d, 4), round(se, 4)
    return f


def _equivalentes(r1, r2):
    """Dos corridas por semilla con los mismos resultados (comprobacion de equivalencia)."""
    return all(
        a["validacion"]["cadena_total_exactitud"] == b["validacion"]["cadena_total_exactitud"]
        and a["prueba"]["e1_exactitud"] == b["prueba"]["e1_exactitud"] and a["rho_calidad"] == b["rho_calidad"]
        for a, b in zip(r1, r2))


def cmd_ablacion(args):
    t0 = time.time()
    filas, log = [], []
    with ProcessPoolExecutor(args.procesos, initializer=_init, initargs=(args.mediciones, args.imagenes)) as ex:
        config = E.ORIGINAL
        base = _ejecutar(ex, config, SEMILLAS)
        filas.append(_fila("S0", "original reentrenado (referencia)", base, base, True,
                           "primer avance, misma receta, particion limpia"))
        # Paso estructural: pasar de 'reglas legadas' a reglas parametrizables sin cambiar nada
        # (mismas reglas y constantes) para poder variar cada medicion por separado.
        desligado = replace(config, legado=False, umbral_cal="fijo", tallo="v1", tono="lineal")
        r0 = _ejecutar(ex, desligado, SEMILLAS)
        if not _equivalentes(base, r0):
            print("ERROR: las reglas parametrizables no reproducen las legadas", file=sys.stderr)
            return 4
        config, base = desligado, r0
        for pid, desc, tipo, opciones in _pasos():
            nombres = list(opciones)
            cands = [(nom, opciones[nom](config)) for nom in nombres]
            resultados = {nom: (base if cand == config else _ejecutar(ex, cand, SEMILLAS)) for nom, cand in cands}
            deltas = {nom: E.diferencia_pareada(base, resultados[nom], "validacion", CRITERIO) for nom in nombres}
            mejor = max(nombres[1:], key=lambda n: deltas[n][0])
            d, se = deltas[mejor]
            if tipo == "obligatoria":
                elegido = nombres[1]
            elif tipo == "correccion":
                elegido = mejor if d >= -MARGEN else None
            else:
                elegido = mejor if (d >= MARGEN and d >= se) else None
            for nom, cand in cands:
                if elegido is not None:
                    adoptada = nom == elegido
                    nota = {"obligatoria": "obligatoria (correccion de metodo)",
                            "correccion": "correccion de defecto, no inferior (>= -margen)",
                            "mejora": "mejora >= margen y > 1 SE"}[tipo] if adoptada else ""
                else:
                    adoptada = cand == config
                    nota = "" if adoptada else "sin evidencia suficiente"
                filas.append(_fila(f"{pid} {desc}", nom, resultados[nom], base, adoptada, nota))
            if elegido is not None:
                config = dict(cands)[elegido]
                base = resultados[elegido]
            d_el = deltas[elegido][0] if elegido else max(deltas[n][0] for n in nombres[1:])
            log.append({"paso": pid, "descripcion": desc, "elegida": elegido or nombres[0],
                        "delta_validacion": round(d_el, 4),
                        "mejor_alternativa_se": round(deltas[elegido][1] if elegido else deltas[max(nombres[1:], key=lambda n: deltas[n][0])][1], 4)})
            print(f"{pid} {desc}: -> {log[-1]['elegida']}  (delta val {d_el:+.4f})", flush=True)
    os.makedirs(args.salida, exist_ok=True)
    ruta = os.path.join(args.salida, "ablacion.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    with open(os.path.join(args.salida, "config_final.json"), "w", encoding="utf-8") as f:
        json.dump({"variante": asdict(config), "decisiones": log, "semillas": list(SEMILLAS),
                   "criterio": CRITERIO, "margen": MARGEN}, f, indent=1)
    print(f"\nAblacion terminada en {time.time() - t0:.0f}s -> {ruta}")
    return COD_OK


def _variante_final(ruta):
    with open(ruta, encoding="utf-8") as f:
        d = json.load(f)["variante"]
    return E.Variante(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})


def _msd(x, pct=True):
    m, sd = x
    return f"{100 * m:.1f}% +- {100 * sd:.1f}" if pct else f"{m:.2f} +- {sd:.2f}"


def _escribir_parametros(var, res, ruta):
    """Deja en config/parametros_etapas.csv los parametros con los que se entrenaron los modelos v2."""
    fuente = f"elegido en validacion (semilla de particion {SEMILLA_PRINCIPAL}) por scripts/experimentos.py final"
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["etapa", "rho", "L", "max_categorias", "epocas", "politica_nulo", "desempate", "fuente"])
        for etapa in ("morfologia", "calidad"):
            w.writerow([etapa, res[f"rho_{etapa}"], 2.0, E.TOPE_CATEGORIAS, var.epocas, var.politica_nulo,
                        var.desempate, fuente])


def cmd_final(args):
    from evaluacion import entregables as EN, cadena, metricas
    ruta_cfg = os.path.join(args.salida, "config_final.json")
    if not os.path.isfile(ruta_cfg):
        print("ERROR: falta resultados/config_final.json. Ejecute primero: python scripts/experimentos.py ablacion",
              file=sys.stderr)
        return COD_ENTRADA
    var = _variante_final(ruta_cfg)
    dir_ds, dir_mod = args.datasets_v2, args.modelos_v2
    for ruta in (os.path.join(dir_mod, "art1_morfologia.json"), os.path.join(dir_mod, "art1_calidad.json")):
        if os.path.exists(ruta) and not args.sobrescribir:
            print(f"ERROR: '{ruta}' ya existe. Use --sobrescribir para regenerarlo (el original no se toca).",
                  file=sys.stderr)
            return 2
    fotos = banco.inventariar(args.imagenes)
    EN.adjuntar_trazabilidad(fotos, os.path.join(dir_ds, "trazabilidad_ids_originales.csv"))
    med = {s: E.leer_mediciones(os.path.join(args.mediciones, f"mediciones_{s}.csv")) for s in ("v1", "v2")}
    fs = E.particion_con_semilla(fotos, SEMILLA_PRINCIPAL)
    por = {p: [f for f in fs if f["particion"] == p] for p in banco.PARTICIONES}
    res = E.correr_split(var, fotos, med[var.segmentador], SEMILLA_PRINCIPAL, fotos_particionadas=fs)
    if "fallo_morfologia" in res or "fallo_calidad" in res:
        print("ERROR: corrida infructuosa en la configuracion final", file=sys.stderr)
        return 1
    entr = E.ordenar_entrenamiento(por["entrenamiento"], 10_000 + SEMILLA_PRINCIPAL)
    rutas = EN.escribir_datasets(dir_ds, fs, por, entr, res["_pre"], med[var.segmentador])
    procedencia = {"semilla_particion": SEMILLA_PRINCIPAL, "manifiesto": "datasets_v2/manifiesto_imagenes.csv",
                   "umbrales_calculados_con": "solo la particion de entrenamiento",
                   "rho_elegido_con": "particion de validacion", "script": "scripts/experimentos.py final",
                   "variante": asdict(var), "n_particion": res["n"]}
    modelos = {}
    for etapa in ("morfologia", "calidad"):
        modelos[etapa] = EN.entrenar_y_guardar(
            var, etapa, rutas[(etapa, "entrenamiento")], res[f"rho_{etapa}"], res["_pre"][etapa],
            os.path.join(dir_mod, f"art1_{etapa}.json"), procedencia, args.sobrescribir)
        mem = res["_modelos"][etapa]
        assert modelos[etapa].red.top_down == mem.red.top_down, "el modelo entrenado desde el CSV difiere del de la evaluacion"
        assert modelos[etapa].meta["mapeo_clusters"] == json.loads(json.dumps(mem.meta["mapeo_clusters"]))
    _escribir_parametros(var, res, os.path.join(RAIZ, "config", "parametros_etapas.csv"))

    # --- evaluacion con los modelos RECARGADOS desde JSON ---------------------------------------
    pm, pc = modelos["morfologia"].preprocesamiento, modelos["calidad"].preprocesamiento
    medv = med[var.segmentador]
    may_tipo = metricas.clase_mayoritaria([f["tipo_real"] for f in por["entrenamiento"]])
    may_cal = metricas.clase_mayoritaria([f["calidad_real"] for f in por["entrenamiento"] if f["tipo_real"] == "carozo"])

    def evaluar(fl):
        regs = cadena.evaluar_cadena(fl, EN.vectores(fl, medv, pm), EN.vectores(fl, medv, pc),
                                     modelos["morfologia"], modelos["calidad"])
        return regs, cadena.resumir_cadena(regs, may_tipo, may_cal)

    mejorado = {p: evaluar(por[p]) for p in banco.PARTICIONES}
    mejorado["todas"] = evaluar(fs)
    assert [r["morf_etiqueta"] for r in mejorado["prueba"][0]] == [r["morf_etiqueta"] for r in res["prueba"]["_registros"]], \
        "guardar/cargar altero las predicciones"
    ruta_orig = os.path.join(RAIZ, "modelos")
    orig_ent = {}
    for p in list(banco.PARTICIONES) + ["todas"]:
        fl = fs if p == "todas" else por[p]
        r = EN.evaluar_original_entregado(fl, med["v1"], ruta_orig)
        orig_ent[p] = (r, cadena.resumir_cadena(r, may_tipo, may_cal))
    var0 = E.ORIGINAL
    res0 = E.correr_split(var0, fotos, med["v1"], SEMILLA_PRINCIPAL, fotos_particionadas=fs)
    o0 = {p: (res0[p]["_registros"], res0[p]) for p in banco.PARTICIONES}

    with ProcessPoolExecutor(args.procesos, initializer=_init, initargs=(args.mediciones, args.imagenes)) as ex:
        sem_mej = _ejecutar(ex, var, SEMILLAS)
        sem_orig = _ejecutar(ex, var0, SEMILLAS)

    # --- informe ----------------------------------------------------------------------------------
    n_man = sum(f["tipo_real"] == "carozo" for f in fs)
    L = ["# Comparacion de la version original y la mejorada\n",
         "Generado por `python scripts/experimentos.py final` (ver docs/informe_cambios.md).\n",
         "## Datos y condiciones\n",
         f"- Banco: {len(fs)} fotos ({n_man} manzanas, "
         f"{sum(f['tipo_real'] == 'citrico' for f in fs)} naranjas, {sum(f['tipo_real'] == 'tropical' for f in fs)} bananas). "
         "Son las que quedaron tras la curacion del primer avance: **ya estan filtradas**, el resultado es optimista respecto del uso real.",
         f"- Particion principal (semilla {SEMILLA_PRINCIPAL}), por grupo de fotos tomadas con <= {banco.GAP_RAFAGA_S} s de diferencia "
         f"(proxy de fruta/rafaga; NO hay id de fruta fisica ni de sesion): entrenamiento {res['n']['entrenamiento']}, "
         f"validacion {res['n']['validacion']}, prueba {res['n']['prueba']}. No se demuestra independencia entre fotos de la misma sesion.",
         f"- Parametros de la version mejorada: rho morfologia {res['rho_morfologia']}, rho calidad {res['rho_calidad']} "
         f"(elegidos en validacion), pasadas hasta {var.epocas}, desempate '{var.desempate}', "
         f"umbrales de morfologia '{var.umbral_morf}' y de calidad '{var.umbral_cal}' calculados con el entrenamiento, "
         f"segmentador {var.segmentador}, tiene_tallo '{var.tallo}', tono '{var.tono}', "
         f"extras morfologia {list(var.extras_morf)}, extras calidad {list(var.extras_cal)}.",
         "- Original entregado: modelos de `modelos/` (rho 0.85 / 0.70), constantes del primer avance; entrenados con fotos que hoy "
         "estan en validacion y prueba, por lo que **sobre esas particiones sus resultados son optimistas** (reinferencia).",
         "- Original reentrenado: la receta del primer avance (segmentacion v1, constantes, asignacion online, etapa 1 solo con manzanas good, "
         "rho 0.85 / 0.70, una pasada) entrenada solo con la particion de entrenamiento.\n"]
    L.append("## A. Diagnostico sobre las 339 fotos (reinferencia: incluye fotos de entrenamiento)\n")
    L.append(EN.tabla_resumen([("original entregado", orig_ent["todas"][1], None),
                               ("mejorado (entrenado con la particion principal)", mejorado["todas"][1], None)],
                              EN.COLUMNAS_PRINCIPALES))
    L.append("\nNo es una validacion externa: en el mejorado, las fotos de entrenamiento (%d) estan incluidas.\n" % res["n"]["entrenamiento"])
    L.append("## B. Particion principal: tres evaluaciones por separado\n")
    for p, titulo in (("entrenamiento", "Reinferencia sobre ENTRENAMIENTO (no es evaluacion)"),
                      ("validacion", "VALIDACION (se uso para elegir rho)"),
                      ("prueba", "PRUEBA (no intervino en ninguna decision)")):
        L.append(f"### {titulo}\n")
        filas = [("original entregado", orig_ent[p][1], None),
                 ("original reentrenado", o0[p][1], None),
                 ("mejorado", mejorado[p][1], EN.ic(mejorado[p][0], 1) if p != "entrenamiento" else None)]
        L.append(EN.tabla_resumen(filas, EN.COLUMNAS_PRINCIPALES))
        L.append("")
    L.append("IC95: remuestreo de grupos de fotos (bootstrap), 2000 repeticiones; con pocos grupos es ancho.\n")
    t = mejorado["prueba"][1]
    L.append("### Matrices de confusion en PRUEBA\n")
    L.append(EN.matriz_md(t["e1_confusion"], cadena.TIPOS, "Mejorado - etapa 1 (tipo)"))
    L.append("")
    L.append(EN.matriz_md(t["e2_aislada_confusion"], cadena.CALIDADES, "Mejorado - etapa 2 aislada (todas las manzanas reales de prueba)"))
    L.append("")
    t0 = o0["prueba"][1]
    L.append(EN.matriz_md(t0["e1_confusion"], cadena.TIPOS, "Original reentrenado - etapa 1 (tipo)"))
    L.append("")
    L.append(EN.matriz_md(t0["e2_aislada_confusion"], cadena.CALIDADES, "Original reentrenado - etapa 2 aislada"))
    L.append("\n## C. Variabilidad entre 30 particiones aleatorias por grupo (media +- desvio)\n")
    L.append("| sistema | particion | cadena total | cadena de manzanas | tipo exactitud | tipo F1 macro | calidad aislada F1 macro | manzanas derivadas (n) | naranjas/bananas derivadas (n) | categorias (morf/cal) | rho (morf/cal) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for nombre, sem in (("original reentrenado", sem_orig), ("mejorado", sem_mej)):
        for p in ("entrenamiento", "validacion", "prueba"):
            a = E.agregar(sem, p)
            L.append(f"| {nombre} | {p} | {_msd(a['cadena_total_exactitud'])} | {_msd(a['cadena_manzanas_exactitud'])} | {_msd(a['e1_exactitud'])} | {a['e1_f1_macro'][0]:.3f} +- {a['e1_f1_macro'][1]:.3f} | "
                     f"{a['e2_aislada_f1_macro'][0]:.3f} +- {a['e2_aislada_f1_macro'][1]:.3f} | "
                     f"{_msd(a['manzanas_derivadas'], False)} | {_msd(a['no_manzanas_derivadas'], False)} | "
                     f"{a['k_morfologia'][0]:.1f} / {a['k_calidad'][0]:.1f} | {a['rho_morfologia'][0]:.2f} / {a['rho_calidad'][0]:.2f} |")
    pr = E.agregar(sem_mej, "prueba")
    L.append(f"\nReferencia 'predecir siempre la mayoritaria' (prueba): cadena de manzanas {_msd(pr['linea_base_cadena_manzanas'])}; "
             f"tipo {_msd(pr['linea_base_tipo'])}. La desviacion refleja la variacion entre particiones; las particiones comparten fotos, "
             "por lo que NO es un intervalo de confianza independiente.\n")
    L.append("## D. Categorias de los modelos mejorados (etiquetado con prototipos finales)\n")
    for etapa in ("morfologia", "calidad"):
        m = modelos[etapa]
        L.append(f"**{etapa}** (rho {m.red.rho}, {len(m.red.top_down)} categorias, variables {m.cols_features})\n")
        L.append("| categoria | prototipo | muestras (reinferencia) | distribucion | etiqueta | estado |")
        L.append("|---|---|---|---|---|---|")
        for cid, info in m.meta["mapeo_clusters"].items():
            L.append(f"| {cid} | {''.join(map(str, m.red.top_down[int(cid)]))} | {info['n']} | {info['distribucion']} | "
                     f"{info['etiqueta'] or '-'} | {info['estado']} |")
        L.append(f"\nMuestras de entrenamiento sin resonancia con los prototipos finales: {m.meta['reinferencia_entrenamiento']['n_sin_resonancia']}.\n")
    with open(os.path.join(args.salida, "comparacion.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    cols = ["foto_id", "particion", "estrato", "tipo_real", "calidad_real", "morf_cluster", "morf_etiqueta", "morf_estado",
            "morf_match", "derivada", "cal_cluster", "cal_etiqueta", "cal_estado", "cal_match", "orac_etiqueta"]
    with open(os.path.join(args.salida, "final_fotos.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for p in banco.PARTICIONES:
            for r in mejorado[p][0]:
                w.writerow({**r, "particion": p})
    resumen = {"variante": asdict(var), "semilla_principal": SEMILLA_PRINCIPAL, "n": res["n"],
               "rho": {e: res[f"rho_{e}"] for e in ("morfologia", "calidad")},
               "mejorado": {p: mejorado[p][1] for p in mejorado},
               "original_reentrenado": {p: {k: v for k, v in o0[p][1].items() if not k.startswith("_")} for p in o0},
               "original_entregado": {p: orig_ent[p][1] for p in orig_ent}}
    with open(os.path.join(args.salida, "final_resumen.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=1, default=float)
    print(open(os.path.join(args.salida, "comparacion.md"), encoding="utf-8").read())
    return COD_OK


N_ORDENES = 30


def _estadistica(valores):
    v = [x for x in valores if x == x]
    if not v:
        return float("nan"), float("nan"), float("nan"), float("nan")
    m = sum(v) / len(v)
    sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
    return m, sd, min(v), max(v)


def _barrer(vectores, etiquetas, grilla, epocas, f1_val=None, politica_nulo="categoria_propia"):
    """Para cada rho: N_ORDENES ordenes aleatorios + el orden original. f1_val(red) -> F1 de validacion (opcional)."""
    import random
    from CarGross import RedART1, ProliferacionError, entrenar, pureza
    filas = []
    for rho in grilla:
        k, p_on, p_fin, f1, fallos = [], [], [], [], 0
        for i in range(N_ORDENES + 1):
            idx = list(range(len(vectores)))
            if i:
                random.Random(7_000 + i).shuffle(idx)          # i=0: orden original
            X_ = [vectores[j] for j in idx]
            y_ = [etiquetas[j] for j in idx]
            red = RedART1(len(X_[0]), rho, 2.0, E.TOPE_CATEGORIAS, politica_nulo)
            try:
                info = entrenar(red, X_, epocas=epocas)
            except ProliferacionError:                           # corrida infructuosa
                fallos += 1
                continue
            k.append(len(red.top_down))
            p_on.append(pureza(info["asignaciones_online"], y_)[0])
            fin = [r["ganador"] if r["resuena"] else None for r in map(red.evaluar, X_)]
            p_fin.append(pureza(fin, y_)[0])
            if f1_val:
                f1.append(f1_val(red, X_, y_))
        fila = {"rho": rho, "ordenes": N_ORDENES + 1, "infructuosas": fallos}
        for nombre, vals in (("k", k), ("pureza_online", p_on), ("pureza_final", p_fin), ("f1_validacion", f1)):
            m, sd, lo, hi = _estadistica(vals)
            fila.update({f"{nombre}_media": round(m, 4), f"{nombre}_sd": round(sd, 4),
                         f"{nombre}_min": round(lo, 4), f"{nombre}_max": round(hi, 4)})
        filas.append(fila)
    return filas


def _techo(vectores, etiquetas):
    por = {}
    for v, e in zip(vectores, etiquetas):
        por.setdefault(tuple(v), Counter()).update([e])
    mezclados = sum(1 for c in por.values() if len(c) > 1)
    return {"filas": len(vectores), "vectores_distintos": len(por), "vectores_con_etiquetas_mezcladas": mezclados,
            "techo_pureza": sum(max(c.values()) for c in por.values()) / len(vectores)}


def cmd_barrido(args):
    import extraccion as X
    from CarGross import Modelo
    from evaluacion import cadena, metricas
    import validacion as val
    os.makedirs(args.salida, exist_ok=True)
    ruta_cfg = os.path.join(args.salida, "config_final.json")
    var = _variante_final(ruta_cfg)
    L = ["# Barrido de rho y sensibilidad al orden de presentacion\n",
         f"{N_ORDENES} permutaciones aleatorias del entrenamiento + el orden original, para cada rho. "
         "Mezclar con una semilla fija permite repetir un orden, pero no elimina la dependencia del orden: "
         "ART1 es un algoritmo en linea y su resultado depende de la secuencia.\n"]
    todas = []
    # --- A. datasets entregados (180 filas), una pasada ---------------------------------------------
    for etapa, nombre in (("morfologia", "dataset_morfologia.csv"), ("calidad", "dataset_calidad_real.csv")):
        ds = val.leer_dataset(os.path.join(RAIZ, "baseline_original", "datasets", nombre))
        t = _techo(ds.entradas, ds.etiquetas)
        filas = _barrer(ds.entradas, ds.etiquetas, E.GRILLA_RHO, 1)
        for f in filas:
            f["dataset"] = f"entregado_{etapa}"
        todas += filas
        L.append(f"## Dataset entregado de {etapa} (180 filas, una pasada)\n")
        L.append(f"Vectores distintos: {t['vectores_distintos']}; con etiquetas mezcladas: {t['vectores_con_etiquetas_mezcladas']}; "
                 f"**techo de pureza por vectores identicos: {100 * t['techo_pureza']:.2f}%** (ningun rho lo supera: "
                 "vectores iguales con etiquetas distintas no se separan).\n")
        L.append("| rho | categorias media (min-max) | pureza online media (min-max) | pureza final media | corridas infructuosas |")
        L.append("|---|---|---|---|---|")
        for f in filas:
            L.append(f"| {f['rho']} | {f['k_media']:.1f} ({f['k_min']:.0f}-{f['k_max']:.0f}) | "
                     f"{100 * f['pureza_online_media']:.1f}% ({100 * f['pureza_online_min']:.1f}-{100 * f['pureza_online_max']:.1f}) | "
                     f"{100 * f['pureza_final_media']:.1f}% | {f['infructuosas']}/{f['ordenes']} |")
        L.append("")
    # --- B. version mejorada: entrenamiento de la particion principal ---------------------------------------
    fotos = banco.inventariar(args.imagenes)
    med = {s: E.leer_mediciones(os.path.join(args.mediciones, f"mediciones_{s}.csv")) for s in ("v1", "v2")}
    fs = E.particion_con_semilla(fotos, SEMILLA_PRINCIPAL)
    por = {p: [f for f in fs if f["particion"] == p] for p in banco.PARTICIONES}
    m = med[var.segmentador]
    for etapa, campo in (("morfologia", "tipo_real"), ("calidad", "calidad_real")):
        grupo = [f for f in por["entrenamiento"] if etapa == "morfologia" or f["tipo_real"] == "carozo"]
        pre = E._pre(var, etapa, [m[f["foto_id"]] for f in grupo])
        vec = [X.vectorizar(m[f["foto_id"]], pre) for f in grupo]
        et = [f[campo] for f in grupo]
        vfotos = por["validacion"]
        vvec = {f["foto_id"]: X.vectorizar(m[f["foto_id"]], pre) for f in vfotos}

        def f1_val(red, X_, y_, _etapa=etapa, _vv=vvec, _vf=vfotos, _pre=pre):
            from CarGross import construir_mapeo
            mapeo, _ = construir_mapeo(red, X_, y_, var.desempate)
            mod = Modelo(red, {"cols_features": _pre["cols_features"], "mapeo_clusters": mapeo}, _pre, 2, [], None)
            return E._f1_etapa(mod, _vv, _vf, _etapa)

        t = _techo(vec, et)
        filas = _barrer(vec, et, E.GRILLA_RHO, var.epocas, f1_val, var.politica_nulo)
        for f in filas:
            f["dataset"] = f"mejorado_{etapa}"
        todas += filas
        L.append(f"## Version mejorada - {etapa} (entrenamiento de la particion principal, {len(vec)} fotos, hasta {var.epocas} pasadas)\n")
        L.append(f"Vectores distintos: {t['vectores_distintos']}; con etiquetas mezcladas: {t['vectores_con_etiquetas_mezcladas']}; "
                 f"techo de pureza por vectores identicos: {100 * t['techo_pureza']:.2f}%.\n")
        L.append("| rho | categorias media (min-max) | pureza online media | pureza final media | F1 macro validacion media (min-max) | infructuosas |")
        L.append("|---|---|---|---|---|---|")
        for f in filas:
            L.append(f"| {f['rho']} | {f['k_media']:.1f} ({f['k_min']:.0f}-{f['k_max']:.0f}) | "
                     f"{100 * f['pureza_online_media']:.1f}% | {100 * f['pureza_final_media']:.1f}% | "
                     f"{f['f1_validacion_media']:.3f} ({f['f1_validacion_min']:.3f}-{f['f1_validacion_max']:.3f}) | "
                     f"{f['infructuosas']}/{f['ordenes']} |")
        L.append("")
    cols = sorted({k for f in todas for k in f}, key=lambda k: (k not in ("dataset", "rho"), k))
    with open(os.path.join(args.salida, "barrido_rho.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(todas)
    # --- C. sensibilidad de la evaluacion de punta a punta al orden (rho y particion fijos) ---------------------------------------
    L.append("## Sensibilidad al orden de las metricas de punta a punta (misma particion, rho fijo)\n")
    L.append("| sistema | particion | cadena total media +- sd (min-max) | tipo F1 macro media +- sd | categorias (morf/cal) media (min-max) |")
    L.append("|---|---|---|---|---|")
    filas_o = []
    for nombre, v0 in (("original reentrenado", E.ORIGINAL), ("mejorado", None)):
        if v0 is None:
            res_ref = E.correr_split(var, fotos, m, SEMILLA_PRINCIPAL, fotos_particionadas=fs)
            v0 = replace(var, rho_morf=res_ref["rho_morfologia"], rho_cal=res_ref["rho_calidad"])
        medv = med[v0.segmentador]
        corridas = [E.correr_split(v0, fotos, medv, SEMILLA_PRINCIPAL, fotos_particionadas=fs, orden_semilla=20_000 + i)
                    for i in range(N_ORDENES)]
        for p in ("validacion", "prueba"):
            c = [r[p]["cadena_total_exactitud"] for r in corridas if p in r]
            f1 = [r[p]["e1_f1_macro"] for r in corridas if p in r]
            k = [(r["entrenamiento_morfologia"]["n_categorias"], r["entrenamiento_calidad"]["n_categorias"]) for r in corridas if p in r]
            mc, sc, lc, hc = _estadistica(c)
            mf, sf, _, _ = _estadistica(f1)
            km = _estadistica([a for a, _ in k]); kc = _estadistica([b for _, b in k])
            L.append(f"| {nombre} | {p} | {100 * mc:.1f}% +- {100 * sc:.1f} ({100 * lc:.1f}-{100 * hc:.1f}) | {mf:.3f} +- {sf:.3f} | "
                     f"{km[0]:.1f} ({km[2]:.0f}-{km[3]:.0f}) / {kc[0]:.1f} ({kc[2]:.0f}-{kc[3]:.0f}) |")
            filas_o.append({"sistema": nombre, "particion": p, "cadena_media": mc, "cadena_sd": sc, "cadena_min": lc,
                            "cadena_max": hc, "f1_tipo_media": mf, "f1_tipo_sd": sf})
    with open(os.path.join(args.salida, "sensibilidad_orden.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_o[0]))
        w.writeheader()
        w.writerows(filas_o)
    texto = "\n".join(L).replace("nan (nan-nan)", "- (corrida infructuosa)").replace("nan%", "-").replace("nan", "-")
    with open(os.path.join(args.salida, "barrido_orden.md"), "w", encoding="utf-8") as f:
        f.write(texto + "\n")
    print(texto)
    return COD_OK


def cmd_fondo(args):
    """Diagnostico de confusion fondo-clase: ¿cuanto se puede 'clasificar' mirando SOLO el entorno de la foto?"""
    from evaluacion import cadena, metricas
    ruta = os.path.join(args.mediciones, "fondo.csv")
    if not os.path.isfile(ruta):
        print("ERROR: falta resultados/mediciones/fondo.csv (python scripts/extraer_mediciones.py)", file=sys.stderr)
        return COD_ENTRADA
    with open(ruta, newline="", encoding="utf-8") as f:
        fondo = {r["foto_id"]: {k: float(v) for k, v in r.items() if k != "foto_id"} for r in csv.DictReader(f)}
    cols = sorted(next(iter(fondo.values())))
    fotos = banco.inventariar(args.imagenes)
    L = ["# Diagnostico de confusion fondo-clase\n",
         "Pregunta: si NO se mira la fruta sino solo un marco del 5% en el borde de la foto, cuanto se acierta? "
         "Si el entorno por si solo predice la clase, las etiquetas estan mezcladas con el escenario de la foto "
         "y la exactitud de cualquier sistema sobre este banco no prueba que mida la fruta.\n",
         "## Entorno por origen (media del marco exterior)\n",
         "| origen | fotos | fraccion de pixeles verdes del marco | saturacion media del marco | luminosidad L media |",
         "|---|---|---|---|---|"]
    for est in ("manzana_good", "manzana_mixed", "manzana_bad", "naranja", "banana"):
        fl = [f for f in fotos if f["estrato"] == est]
        mv = sum(fondo[f["foto_id"]]["fondo_verde"] for f in fl) / len(fl)
        ms = sum(fondo[f["foto_id"]]["fondo_saturacion"] for f in fl) / len(fl)
        ml = sum(fondo[f["foto_id"]]["fondo_L"] for f in fl) / len(fl)
        L.append(f"| {est} | {len(fl)} | {100 * mv:.0f}% | {ms:.2f} | {ml:.0f} |")
    # lookup-table sobre bits de fondo (umbrales = mediana del entrenamiento), 30 particiones
    res = {"tipo": [], "calidad": []}
    for s in SEMILLAS:
        fs = E.particion_con_semilla(fotos, s)
        por = {p: [f for f in fs if f["particion"] == p] for p in banco.PARTICIONES}
        for etapa, campo, clases, filtro in (("tipo", "tipo_real", cadena.TIPOS, lambda f: True),
                                              ("calidad", "calidad_real", cadena.CALIDADES, lambda f: f["tipo_real"] == "carozo")):
            tr = [f for f in por["entrenamiento"] if filtro(f)]
            te = [f for f in por["prueba"] if filtro(f)]
            med = {c: sorted(fondo[f["foto_id"]][c] for f in tr)[len(tr) // 2] for c in cols}
            vec = lambda f: tuple(int(fondo[f["foto_id"]][c] >= med[c]) for c in cols)
            tabla = {}
            for f in tr:
                tabla.setdefault(vec(f), Counter()).update([f[campo]])
            pred = []
            for f in te:
                c = tabla.get(vec(f))
                pred.append(c.most_common(1)[0][0] if c else None)
            verdad = [f[campo] for f in te]
            may = metricas.clase_mayoritaria([f[campo] for f in tr])
            res[etapa].append((metricas.exactitud(verdad, pred), metricas.f1_macro(verdad, pred, clases),
                               metricas.linea_base_mayoritaria(verdad, may, clases)["exactitud"]))
    with ProcessPoolExecutor(args.procesos, initializer=_init, initargs=(args.mediciones, args.imagenes)) as ex:
        sem = _ejecutar(ex, _variante_final(os.path.join(args.salida, "config_final.json")), SEMILLAS)
    ag = E.agregar(sem, "prueba")
    L.append("\n## Exactitud en PRUEBA (media +- desvio entre 30 particiones por grupo)\n")
    L.append("| tarea | solo el entorno (5 bits, mediana de entrenamiento) | referencia mayoritaria | sistema mejorado (fruta) |")
    L.append("|---|---|---|---|")
    for etapa, titulo, clave_a, clave_f in (("tipo", "tipo de fruta (etapa 1)", "e1_exactitud", "e1_f1_macro"),
                                            ("calidad", "calidad de manzana (etapa 2 aislada)", "e2_aislada_exactitud", "e2_aislada_f1_macro")):
        a = metricas.media_y_desvio([x[0] for x in res[etapa]])
        f1 = metricas.media_y_desvio([x[1] for x in res[etapa]])
        b = metricas.media_y_desvio([x[2] for x in res[etapa]])
        L.append(f"| {titulo} | {_msd(a)} (F1 macro {f1[0]:.3f}) | {_msd(b)} | {_msd(ag[clave_a])} (F1 macro {ag[clave_f][0]:.3f}) |")
    L.append("\nLectura: si la columna 'solo el entorno' se acerca a la del sistema, parte de lo que el sistema 'aprende' puede ser "
             "el escenario (pasto, guantes, pisos) y no la fruta. Es un diagnostico, no una correccion: no hay forma de separar "
             "ambos efectos con este banco (no hay fotos de la misma fruta en escenarios distintos).\n")
    with open(os.path.join(args.salida, "diagnostico_fondo.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return COD_OK


def main():
    ap = argparse.ArgumentParser(description="Experimentos reproducibles (ablacion, final, barrido)")
    ap.add_argument("comando", choices=("ablacion", "final", "barrido", "fondo", "todo"))
    ap.add_argument("--mediciones", default=os.path.join(cfg.DIR_RESULTADOS, "mediciones"))
    ap.add_argument("--imagenes", default=cfg.DIR_IMAGENES, help="solo se usa para listar nombres de archivo")
    ap.add_argument("--salida", default=cfg.DIR_RESULTADOS)
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--datasets-v2", default=cfg.DIR_DATASETS_V2, help="carpeta de los datasets generados")
    ap.add_argument("--modelos-v2", default=cfg.DIR_MODELOS_V2, help="carpeta de los modelos generados")
    ap.add_argument("--sobrescribir", action="store_true", help="permite regenerar modelos v2 existentes")
    args = ap.parse_args()
    if not os.path.isfile(os.path.join(args.mediciones, "mediciones_v1.csv")):
        print("ERROR: faltan las mediciones. Genere con: python scripts/extraer_mediciones.py", file=sys.stderr)
        return COD_ENTRADA
    if args.comando in ("ablacion", "todo"):
        rc = cmd_ablacion(args)
        if rc != COD_OK or args.comando == "ablacion":
            return rc
    if args.comando in ("final", "todo"):
        rc = cmd_final(args)
        if rc != COD_OK:
            return rc
    if args.comando in ("barrido", "todo"):
        rc = cmd_barrido(args)
        if rc != COD_OK:
            return rc
    if args.comando in ("fondo", "todo"):
        return cmd_fondo(args)
    return COD_OK


if __name__ == "__main__":
    sys.exit(main())
