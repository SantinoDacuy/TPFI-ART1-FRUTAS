"""
scripts/comparar_ronda2.py
Comparacion de variantes bajo las MISMAS particiones (ronda 2, particion de informe reservada).

Para cada variante:
  A. 30 particiones internas entrenamiento/validacion (sobre 'desarrollo') + el mismo bloque de informe:
     media +- desvio (y minimo-maximo) en validacion e informe. Es exploratorio: las particiones comparten fotos.
  B. AJUSTE FINAL: entrena con TODO 'desarrollo' (rho = mediana de los rho elegidos en A, sin superar el tope de
     categorias de la variante) y evalua UNA vez en el bloque de informe, con IC95 por remuestreo de grupos.
  C. Sensibilidad al orden: 30 ordenes de presentacion distintos con el ajuste final.
Ademas: matrices de confusion, diagnostico basado solo en el fondo y comparacion pareada de la principal
contra la receta original reentrenada.

Definicion de denominadores (N = fotos del informe; M = manzanas del informe):
  tipo exactitud = tipo correcto / N; cadena total = (manzanas con tipo y calidad correctos + otras frutas con
  tipo correcto) / N; cadena de manzanas = manzanas con tipo y calidad correctos / M;
  calidad aislada = calidad correcta entre las M manzanas, evaluadas todas en la etapa 2;
  calidad acoplada = calidad correcta entre las manzanas que la etapa 1 deriva (denominador = derivadas);
  una novedad o una categoria sin etiqueta cuentan como error y se informan aparte.
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "scripts"))

import busqueda as B
import configuracion as cfg
from evaluacion import banco, cadena, experimentos as E, metricas
from evaluacion import entregables as EN
from validacion import ALIAS_MANZANA, COD_ENTRADA, COD_OK


def cargar_principal(ruta):
    with open(ruta, encoding="utf-8") as f:
        d = json.load(f)["variante"]
    return E.Variante(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})


def variantes(principal):
    """{nombre: (Variante, rol)} de todas las que se comparan."""
    ronda1 = None
    ruta1 = os.path.join(cfg.DIR_RESULTADOS, "config_final.json")
    if os.path.isfile(ruta1):
        with open(ruta1, encoding="utf-8") as f:
            d = json.load(f)["variante"]
        ronda1 = E.Variante(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()})
        ronda1 = replace(ronda1, nombre="medianas_ronda1")
    out = {
        "original_reentrenado": (E.ORIGINAL, "comparador: receta del primer avance"),
        "compatible_minima": (replace(B.COMPATIBLE, nombre="compatible_minima"),
                              "solo correcciones y compromisos (sin descriptores adicionales)"),
        "principal": (replace(principal, nombre="principal"), "PRINCIPAL compatible con el primer avance"),
        "exp_sin_tope": (replace(principal, nombre="exp_sin_tope", tope_k=None),
                         "experimental: sin el tope de 14 categorias del primer avance"),
        "exp_medianas": (replace(principal, nombre="exp_medianas", umbral_morf="mediana"),
                         "experimental: umbrales morfologicos por mediana"),
        "exp_protuberancia": (replace(principal, nombre="exp_protuberancia", tallo="protuberancia"),
                              "experimental: tiene_tallo por protuberancia (sin anotaciones)"),
        "alt_orden_canonico": (replace(principal, nombre="alt_orden_canonico", orden="canonico_asc"),
                               "alternativa compatible: orden de presentacion canonico (determinista)"),
        "exp_comite": (replace(principal, nombre="exp_comite", comite=11, orden="aleatorio"),
                       "experimental: comite de 11 redes ART1 (cambia la arquitectura; costo 11x)"),
        "exp_segmentacion_v2": (replace(principal, nombre="exp_segmentacion_v2", segmentador="v2"),
                                "experimental: segmentacion v2"),
        "exp_normalizacion": (replace(principal, nombre="exp_normalizacion", normalizacion="estirar"),
                              "experimental: estiramiento de contraste antes de medir (mas estable, menos exacto)"),
    }
    if ronda1 is not None:
        out["exp_medianas_ronda1"] = (ronda1, "experimental: receta de medianas de la ronda 1 (sin tope)")
    return out


def pct(x):
    return "-" if x is None or x != x else f"{100 * x:.1f}%"


def fila_informe(nombre, res, regs):
    """Metricas de una evaluacion unica sobre el informe (res = resumen de cadena)."""
    n, nm = res["n"], res["n_manzanas"]
    deriv = [r for r in regs if r["derivada"] and r["tipo_real"] == ALIAS_MANZANA]
    acopl_ok = sum(1 for r in deriv if r.get("cal_etiqueta") == r["calidad_real"])
    e1_ok, ca_ok = EN.aciertos(regs)
    grupos = [r["grupo"] for r in regs]
    return {
        "n": n, "n_manzanas": nm,
        "cadena_total": (res["cadena_total_correcta"], n), "cadena_manzanas": (res["cadena_manzanas_correctas"], nm),
        "tipo_exactitud": (res["e1_tipo_correcto"], n), "tipo_f1": res["e1_f1_macro"],
        "calidad_aislada": res.get("e2_aislada_exactitud"), "calidad_aislada_f1": res.get("e2_aislada_f1_macro"),
        "calidad_acoplada": (acopl_ok, len(deriv)),
        "derivadas_ok": (res["manzanas_derivadas"], nm), "derivadas_mal": (res["no_manzanas_derivadas"], res["n_no_manzanas"]),
        "novedades_e1": res["e1_novedades"], "sin_etiqueta_e1": res["e1_sin_etiqueta"],
        "novedades_e2_aislada": res.get("e2_aislada_novedades"), "sin_etiqueta_e2_aislada": res.get("e2_aislada_sin_etiqueta"),
        "ic_cadena_total": metricas.ic_bootstrap_por_grupos(ca_ok, grupos),
        "ic_tipo": metricas.ic_bootstrap_por_grupos(e1_ok, grupos),
        "base_tipo": res["linea_base_tipo"]["exactitud"], "base_tipo_f1": res["linea_base_tipo"]["f1_macro"],
        "base_cadena_manzanas": res.get("linea_base_cadena_manzanas"),
        "_e1_ok": e1_ok, "_ca_ok": ca_ok, "_grupos": grupos,
    }


def rho_final(resultados, etapa, var):
    """Mediana (a la grilla) de los rho elegidos en las particiones internas; si la variante tiene tope de
    categorias se baja rho de a un paso hasta cumplirlo con el ajuste final."""
    vals = sorted(r[f"rho_{etapa}"] for r in resultados if f"rho_{etapa}" in r)
    med = vals[len(vals) // 2]
    return min(E.GRILLA_RHO, key=lambda g: abs(g - med))


def _correr_orden(args):
    var, rm, rc, i = args
    v = replace(var, rho_morf=rm, rho_cal=rc)
    fs = [dict(f) for f in B._FOTOS]
    banco.reservar_informe(fs, B.SEMILLA_INFORME)
    for f in fs:
        f["particion"] = "entrenamiento" if f["bloque"] == "desarrollo" else "prueba"
    r = E.correr_split(v, B._FOTOS, B._MED[E.clave_mediciones(v)], 0, fotos_particionadas=fs, orden_semilla=50_000 + i)
    if "prueba" not in r:
        return None
    s = r["prueba"]
    return {"cadena_total": s["cadena_total_exactitud"], "cadena_manzanas": s["cadena_manzanas_exactitud"],
            "tipo_exactitud": s["e1_exactitud"], "k_morf": r["entrenamiento_morfologia"]["n_categorias"],
            "k_cal": r["entrenamiento_calidad"]["n_categorias"]}


def fusionar(a, b, clave):
    return [x[clave] for x in (a, b) if x]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Comparacion de variantes (ronda 2)")
    ap.add_argument("--principal", default=os.path.join(B.DIR_SALIDA, "config_principal.json"))
    ap.add_argument("--mediciones", default=os.path.join(cfg.DIR_RESULTADOS, "mediciones"))
    ap.add_argument("--imagenes", default=cfg.DIR_IMAGENES)
    ap.add_argument("--salida", default=B.DIR_SALIDA)
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    if not os.path.isfile(a.principal):
        print("ERROR: falta config_principal.json. Ejecute: python scripts/busqueda.py buscar", file=sys.stderr)
        return COD_ENTRADA
    t0 = time.time()
    principal = cargar_principal(a.principal)
    vs = variantes(principal)
    fotos = banco.inventariar(a.imagenes)
    med = E.cargar_todas_las_mediciones(a.mediciones)
    fs = [dict(f) for f in fotos]
    banco.reservar_informe(fs, B.SEMILLA_INFORME)
    informe = [f for f in fs if f["bloque"] == "informe"]
    desarrollo = [f for f in fs if f["bloque"] == "desarrollo"]
    salida = {}
    with ProcessPoolExecutor(a.procesos, initializer=B._init, initargs=(a.mediciones, a.imagenes)) as ex:
        for nombre, (var, rol) in vs.items():
            inner = B._ejecutar(ex, var) if var.rho_morf is None or var.rho_cal is None else B._ejecutar(ex, var)
            rm = var.rho_morf if var.rho_morf is not None else rho_final(inner, "morfologia", var)
            rc = var.rho_cal if var.rho_cal is not None else rho_final(inner, "calidad", var)
            fin = E.ajustar_desarrollo(var, fotos, med[E.clave_mediciones(var)], B.SEMILLA_INFORME, rm, rc)
            # tope de categorias en el ajuste final
            if var.tope_k is not None:
                while fin.get("entrenamiento_morfologia", {}).get("n_categorias", 0) > var.tope_k and rm > E.GRILLA_RHO[0]:
                    rm = round(rm - 0.05, 2)
                    fin = E.ajustar_desarrollo(var, fotos, med[E.clave_mediciones(var)], B.SEMILLA_INFORME, rm, rc)
                while fin.get("entrenamiento_calidad", {}).get("n_categorias", 0) > var.tope_k and rc > E.GRILLA_RHO[0]:
                    rc = round(rc - 0.05, 2)
                    fin = E.ajustar_desarrollo(var, fotos, med[E.clave_mediciones(var)], B.SEMILLA_INFORME, rm, rc)
            if "prueba" not in fin:
                salida[nombre] = {"rol": rol, "fallo": fin.get("fallo_morfologia") or fin.get("fallo_calidad")}
                continue
            regs = fin["prueba"]["_registros"]
            fi = fila_informe(nombre, fin["prueba"], regs)
            ordenes = list(ex.map(_correr_orden, [(var, rm, rc, i) for i in range(30)], chunksize=2))
            salida[nombre] = {
                "rol": rol, "variante": var, "rho": (rm, rc), "informe": fi, "regs": regs, "fin": fin,
                "inner": inner, "ordenes": [o for o in ordenes if o],
                "k": (fin["entrenamiento_morfologia"], fin["entrenamiento_calidad"]),
            }
            print(f"{nombre}: rho {rm}/{rc} informe cadena total {pct(fi['cadena_total'][0] / fi['cadena_total'][1])} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    # ---- original entregado (diagnostico historico, contaminado) ----
    med1 = med[("v1", "ninguna")]
    regs_ent = EN.evaluar_original_entregado(informe, med1, cfg.DIR_MODELOS_ORIGINAL)
    may_t = metricas.clase_mayoritaria([f["tipo_real"] for f in desarrollo])
    may_c = metricas.clase_mayoritaria([f["calidad_real"] for f in desarrollo if f["tipo_real"] == ALIAS_MANZANA])
    res_ent = cadena.resumir_cadena(regs_ent, may_t, may_c)
    salida["original_entregado"] = {"rol": "diagnostico historico (entrenado con fotos del informe: optimista)",
                                    "informe": fila_informe("original_entregado", res_ent, regs_ent), "regs": regs_ent}
    escribir_informe(salida, a.salida, informe, desarrollo, fs, a)
    print(f"Comparacion terminada en {time.time() - t0:.0f}s")
    return COD_OK


def seccion_fondo(a, fs, salida):
    """Diagnostico: acierto mirando SOLO un marco del 5 % del borde de la foto (5 bits por mediana de entrenamiento)."""
    ruta = os.path.join(a.mediciones, "fondo.csv")
    if not os.path.isfile(ruta):
        return ["## F. Diagnostico basado unicamente en el fondo\n", "(falta resultados/mediciones/fondo.csv)\n"]
    with open(ruta, newline="", encoding="utf-8") as f:
        fondo = {r["foto_id"]: {k: float(v) for k, v in r.items() if k != "foto_id"} for r in csv.DictReader(f)}
    cols = sorted(next(iter(fondo.values())))

    def evaluar(tr, te, campo, clases, filtro):
        tr, te = [f for f in tr if filtro(f)], [f for f in te if filtro(f)]
        med = {c: sorted(fondo[f["foto_id"]][c] for f in tr)[len(tr) // 2] for c in cols}
        vec = lambda f: tuple(int(fondo[f["foto_id"]][c] >= med[c]) for c in cols)
        tabla = defaultdict(Counter)
        for f in tr:
            tabla[vec(f)][f[campo]] += 1
        pred = [tabla[vec(f)].most_common(1)[0][0] if vec(f) in tabla else None for f in te]
        verdad = [f[campo] for f in te]
        return metricas.exactitud(verdad, pred), metricas.f1_macro(verdad, pred, clases), len(te)

    L = ["## F. Diagnostico basado unicamente en el fondo (marco del 5 % del borde, 5 bits)\n",
         "Mirando SOLO el entorno de la foto (sin la fruta) se predice la clase con una tabla de consulta entrenada con "
         "'desarrollo'. Si se acerca al sistema, parte de lo que este acierta puede ser el escenario.\n",
         "| tarea | solo fondo: informe (ajuste final) | solo fondo: informe, 30 particiones internas | referencia mayoritaria | principal (ajuste final) |",
         "|---|---|---|---|---|"]
    dev = [f for f in fs if f["bloque"] == "desarrollo"]
    inf = [f for f in fs if f["bloque"] == "informe"]
    for etapa, campo, clases, filtro, clave in (("tipo", "tipo_real", cadena.TIPOS, lambda f: True, "tipo_exactitud"),
                                                ("calidad (manzanas)", "calidad_real", cadena.CALIDADES,
                                                 lambda f: f["tipo_real"] == ALIAS_MANZANA, "calidad_aislada")):
        ac, f1, n = evaluar(dev, inf, campo, clases, filtro)
        internas = []
        for s in B.SEMILLAS:
            fp = E.particion_informe(fs, s, B.SEMILLA_INFORME)
            internas.append(evaluar([f for f in fp if f["particion"] == "entrenamiento"],
                                    [f for f in fp if f["particion"] == "prueba"], campo, clases, filtro)[0])
        m, sd = metricas.media_y_desvio(internas)
        may = metricas.clase_mayoritaria([f[campo] for f in dev if filtro(f)])
        base = metricas.linea_base_mayoritaria([f[campo] for f in inf if filtro(f)], may, clases)["exactitud"]
        p = salida.get("principal", {}).get("informe", {})
        v = p.get(clave) if p else None
        val_p = (v[0] / v[1] if isinstance(v, tuple) else v) if v is not None else None
        L.append(f"| {etapa} | {pct(ac)} (F1 macro {f1:.3f}, n={n}) | {100 * m:.1f}% +- {100 * sd:.1f} | {pct(base)} | {pct(val_p)} |")
    L.append("\nEs un diagnostico, no una correccion: no hay fotos de la misma fruta en escenarios distintos, asi que no se puede separar el "
             "efecto del escenario del de la fruta. Ver tambien `scripts/robustez.py` (sustitucion de fondo e iluminacion).\n")
    return L


def _fmt(par):
    return f"{par[0]}/{par[1]} ({pct(par[0] / par[1]) if par[1] else '-'})"


def escribir_informe(salida, carpeta, informe, desarrollo, fs, a):
    os.makedirs(carpeta, exist_ok=True)
    nombres = [n for n in salida if "fallo" not in salida[n]]
    L = ["# Comparacion de variantes - segunda ronda\n",
         "Generado por `python scripts/comparar_ronda2.py` (ver `docs/informe_cambios.md`, seccion de la segunda ronda).\n",
         "## Condiciones\n",
         f"- Banco: 339 fotos, **ya inspeccionadas y reutilizadas** en la primera ronda; sin identificador de fruta fisica ni de sesion.",
         f"- Bloque de informe reservado (semilla {B.SEMILLA_INFORME}, por grupos de fotos con <= {banco.GAP_RAFAGA_S} s de diferencia, "
         f"estratificado por origen): {len(informe)} fotos ({sum(f['tipo_real'] == ALIAS_MANZANA for f in informe)} manzanas). "
         f"Desarrollo: {len(desarrollo)} fotos. **No es un conjunto externo intacto**: solo se evito usarlo para elegir en esta ronda.",
         "- Todas las variantes se entrenan y se evaluan sobre las mismas particiones. Los umbrales dependientes de datos usan solo el "
         "entrenamiento; rho se elige en validacion interna; el informe no interviene en ninguna decision.",
         "- Denominadores: ver el docstring de `scripts/comparar_ronda2.py` (N = fotos del informe; M = manzanas del informe).\n",
         "## A. Ajuste final (todo 'desarrollo') evaluado UNA vez en el informe\n"]
    cab = ("| variante | rol | rho (morf/cal) | cadena total | cadena de manzanas | tipo exactitud (IC95) | tipo F1 macro | "
           "calidad aislada (exact. / F1) | calidad acoplada | manzanas derivadas | otras frutas derivadas | "
           "novedades / sin etiqueta (etapa 1) | categorias morf / cal (unitarias) |")
    L += [cab, "|" + "---|" * 13]
    for n in nombres:
        s = salida[n]
        i = s["informe"]
        k = s.get("k")
        kk = f"{k[0]['n_categorias']} ({k[0]['unitarias']}) / {k[1]['n_categorias']} ({k[1]['unitarias']})" if k else "-"
        rho = f"{s['rho'][0]} / {s['rho'][1]}" if "rho" in s else "(entregado: 0.85 / 0.70)"
        L.append(f"| {n} | {s['rol']} | {rho} | {_fmt(i['cadena_total'])} IC [{pct(i['ic_cadena_total'][0])}, {pct(i['ic_cadena_total'][1])}] | "
                 f"{_fmt(i['cadena_manzanas'])} | {_fmt(i['tipo_exactitud'])} [{pct(i['ic_tipo'][0])}, {pct(i['ic_tipo'][1])}] | "
                 f"{i['tipo_f1']:.3f} | {pct(i['calidad_aislada'])} / {i['calidad_aislada_f1']:.3f} | {_fmt(i['calidad_acoplada'])} | "
                 f"{_fmt(i['derivadas_ok'])} | {_fmt(i['derivadas_mal'])} | {i['novedades_e1']} / {i['sin_etiqueta_e1']} | {kk} |")
    i0 = salida[nombres[0]]["informe"]
    L.append(f"\nReferencias mayoritarias en el informe (clase del entrenamiento): tipo {pct(i0['base_tipo'])} "
             f"(F1 macro {i0['base_tipo_f1']:.3f}); cadena de manzanas {pct(i0.get('base_cadena_manzanas'))}. "
             "IC95: remuestreo de grupos (2000 repeticiones); con pocos grupos es ancho y no implica significacion.\n")
    # tamanos de categorias
    L.append("### Tamanos de las categorias del ajuste final (muestras de entrenamiento con prototipos finales)\n")
    L += ["| variante | etapa | categorias | unitarias | sin etiqueta | mediana | maximo | tamanos (descendente) |", "|---|---|---|---|---|---|---|---|"]
    for n in nombres:
        for et, idx in (("morfologia", 0), ("calidad", 1)):
            if "k" not in salida[n]:
                continue
            kinfo = salida[n]["k"][idx]
            t = kinfo["tamanos"]
            L.append(f"| {n} | {et} | {kinfo['n_categorias']} | {kinfo['unitarias']} | {kinfo['sin_etiqueta']} | "
                     f"{statistics.median(t):.1f} | {max(t)} | {t} |")
    # B. variabilidad
    L.append("\n## B. Variabilidad entre 30 particiones internas (media +- desvio [minimo-maximo])\n")
    L.append("Entrenamiento/validacion salen de 'desarrollo'; el informe es el mismo bloque en todas. Cada particion entrena con ~70 % de desarrollo.\n")
    L += ["| variante | validacion: cadena total | validacion: cadena manzanas | informe: cadena total | informe: cadena manzanas | informe: tipo exactitud | informe: tipo F1 macro | informe: otras frutas derivadas (n) | categorias (morf/cal) | rho elegidos (morf/cal) |",
          "|---|---|---|---|---|---|---|---|---|---|"]

    def ms(valores):
        m, sd = metricas.media_y_desvio(valores)
        return f"{100 * m:.1f}% +- {100 * sd:.1f} [{100 * min(valores):.0f}-{100 * max(valores):.0f}]"

    for n in nombres:
        if "inner" not in salida[n]:
            continue
        inn = [r for r in salida[n]["inner"] if "prueba" in r]
        agv = E.agregar(inn, "validacion")
        agi = E.agregar(inn, "prueba")
        L.append(f"| {n} | {ms([r['validacion']['cadena_total_exactitud'] for r in inn])} | "
                 f"{ms([r['validacion']['cadena_manzanas_exactitud'] for r in inn])} | "
                 f"{ms([r['prueba']['cadena_total_exactitud'] for r in inn])} | "
                 f"{ms([r['prueba']['cadena_manzanas_exactitud'] for r in inn])} | "
                 f"{ms([r['prueba']['e1_exactitud'] for r in inn])} | "
                 f"{agi['e1_f1_macro'][0]:.3f} +- {agi['e1_f1_macro'][1]:.3f} | "
                 f"{agi['no_manzanas_derivadas'][0]:.1f} +- {agi['no_manzanas_derivadas'][1]:.1f} | "
                 f"{agi['k_morfologia'][0]:.1f} / {agi['k_calidad'][0]:.1f} | "
                 f"{agi['rho_morfologia'][0]:.2f} / {agi['rho_calidad'][0]:.2f} |")
    L.append(f"\nFallos (corridas infructuosas) por variante: " + ", ".join(
        f"{n}: {sum(1 for r in salida[n]['inner'] if 'prueba' not in r)}" for n in nombres if "inner" in salida[n]) + ".\n")
    # C. orden
    L.append("## C. Sensibilidad al orden de presentacion (ajuste final, 30 ordenes aleatorios, informe)\n")
    L += ["| variante | cadena total: media +- desvio [min-max] | cadena de manzanas | tipo exactitud | categorias morf (min-max) | categorias cal (min-max) |",
          "|---|---|---|---|---|---|"]
    for n in nombres:
        o = salida[n].get("ordenes")
        if not o:
            continue
        kmo = [x["k_morf"] for x in o]
        kca = [x["k_cal"] for x in o]
        L.append(f"| {n} | {ms([x['cadena_total'] for x in o])} | {ms([x['cadena_manzanas'] for x in o])} | "
                 f"{ms([x['tipo_exactitud'] for x in o])} | {min(kmo)}-{max(kmo)} | {min(kca)}-{max(kca)} |")
    # D. matrices
    L.append("\n## D. Matrices de confusion en el informe (ajuste final)\n")
    for n in ("principal", "original_reentrenado", "exp_medianas"):
        if n not in salida or "fin" not in salida[n]:
            continue
        r = salida[n]["fin"]["prueba"]
        L.append(EN.matriz_md(r["e1_confusion"], cadena.TIPOS, f"{n}: etapa 1 (tipo)"))
        L.append("")
        L.append(EN.matriz_md(r["e2_aislada_confusion"], cadena.CALIDADES, f"{n}: etapa 2 aislada (todas las manzanas del informe)"))
        L.append("")
    # E. comparacion pareada
    L += seccion_fondo(a, fs, salida)
    L.append("## E. Comparacion pareada con la receta original reentrenada (informe, ajuste final)\n")
    if "principal" in salida and "original_reentrenado" in salida:
        p, o = salida["principal"]["informe"], salida["original_reentrenado"]["informe"]
        for etiqueta, kp, ko in (("cadena total", "_ca_ok", "_ca_ok"), ("tipo", "_e1_ok", "_e1_ok")):
            d = [int(x) - int(y) for x, y in zip(p[kp], o[ko])]
            lo, hi = metricas.ic_bootstrap_por_grupos(d, p["_grupos"])
            L.append(f"- {etiqueta}: diferencia media (principal - original reentrenado) = {100 * sum(d) / len(d):+.1f} puntos "
                     f"sobre {len(d)} fotos; intervalo percentil 95 por remuestreo de grupos [{100 * lo:+.1f}, {100 * hi:+.1f}].")
        L.append("\nEl intervalo describe la incertidumbre por la composicion del informe (pocos grupos); no se afirma significacion estadistica "
                 "ni independencia: las fotos son de un mismo banco reutilizado y las rafagas pueden compartir fruta.\n")
    with open(os.path.join(carpeta, "comparacion.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    resumen = {n: {"rol": s["rol"], "rho": list(s.get("rho", [])),
                   "informe": {k: v for k, v in s["informe"].items() if not k.startswith("_")}} for n, s in salida.items() if "informe" in s}
    with open(os.path.join(carpeta, "comparacion.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=1, default=float)
    import csv
    with open(os.path.join(carpeta, "informe_fotos_principal.csv"), "w", newline="", encoding="utf-8") as f:
        if "principal" in salida:
            cols = ["foto_id", "estrato", "grupo", "tipo_real", "calidad_real", "morf_cluster", "morf_etiqueta", "morf_estado",
                    "morf_match", "derivada", "cal_cluster", "cal_etiqueta", "cal_estado", "cal_match", "orac_etiqueta"]
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(salida["principal"]["regs"])
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
