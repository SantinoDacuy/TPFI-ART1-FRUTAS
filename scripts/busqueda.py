"""
scripts/busqueda.py
Segunda ronda: busqueda de mejoras DENTRO del alcance del primer avance, con una PARTICION DE INFORME
reservada, y comparacion de variantes bajo las mismas particiones.

Protocolo (declarado antes de mirar resultados)
  * Se reserva el 30 % de los GRUPOS de fotos (estratificado por origen, semilla SEMILLA_INFORME elegida solo por
    el equilibrio de tamanos) como bloque de 'informe'. Ninguna decision (variantes, umbrales, rho) lo usa.
  * Con el 70 % restante ("desarrollo") se repiten 30 particiones agrupadas entrenamiento/validacion
    (70/30 de los grupos). Los umbrales dependientes de datos se calculan solo con el entrenamiento de cada
    particion; rho se elige en validacion.
  * Cada paso prueba opciones sobre la configuracion vigente. Reglas de adopcion (metrica: exactitud de la
    cadena completa en VALIDACION, media entre particiones, diferencias pareadas):
      'obligatoria' correccion de metodo o compromiso del primer avance: se adopta siempre; se informa el efecto;
      'correccion'  arregla un defecto demostrado: se adopta si no es peor que -MARGEN (no inferior);
      'mejora'      se adopta si mejora >= MARGEN y mas de 1 error estandar de la diferencia pareada.
    Los pasos 'experimental' NO son adoptables en la variante principal (alteran una definicion comprometida):
    se miden aparte.
  * ADVERTENCIA: las 339 fotos ya fueron inspeccionadas y reutilizadas en la primera ronda. La particion de
    informe no es un conjunto externo intacto; solo se evita que esta ronda la use para elegir. Los errores
    estandar describen la variacion entre particiones que comparten fotos; no son intervalos independientes.

Subcomandos: buscar | comparar | todo
Salidas en resultados/ronda2/.
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import configuracion as cfg
from evaluacion import banco, experimentos as E
from validacion import COD_ENTRADA, COD_OK

SEMILLA_INFORME = 8        # elegida solo por el equilibrio de tamanos por origen (ver banco.reservar_informe)
SEMILLAS = tuple(range(30))
MARGEN = 0.01
CRITERIO = "cadena_total_exactitud"
TOPE_K_COMPROMISO = 14     # criterio de exito del primer avance: 15 o mas clusters para 3 clases = resultado fallido
DIR_SALIDA = os.path.join(cfg.DIR_RESULTADOS, "ronda2")

_FOTOS = None
_MED = None


def _init(dir_med, dir_img):
    global _FOTOS, _MED
    _FOTOS = banco.inventariar(dir_img)
    _MED = E.cargar_todas_las_mediciones(dir_med)


def _correr(args):
    var, semilla = args
    r = E.correr_split(var, _FOTOS, _MED[E.clave_mediciones(var)], semilla, semilla_informe=SEMILLA_INFORME)
    for k in ("_modelos", "_pre"):
        r.pop(k, None)
    for p in banco.PARTICIONES:
        if p in r:
            r[p].pop("_registros", None)
    return r


def _ejecutar(ex, var, semillas=SEMILLAS):
    return list(ex.map(_correr, [(var, s) for s in semillas], chunksize=2))


# ---------------------------------------------------------------------------
# Variantes
# ---------------------------------------------------------------------------
# Compatible con el primer avance: segmentacion v1; cortes morfologicos ORIGINALES; tiene_tallo por defectos de
# convexidad (el criterio original, corregido para OpenCV 5); tono circular; calidad con medianas del
# entrenamiento (la logica de lote del primer avance); etapa 1 con manzanas de todas las calidades; etiquetado
# con prototipos finales; rho en validacion.
COMPATIBLE = E.Variante("compatible", segmentador="v1", tallo="convexidad", tono="circular",
                        umbral_morf="compromiso", umbral_cal="compromiso", cobertura="todas",
                        mapeo="prototipos_finales", desempate="indeterminada", epocas=1,
                        tope_k=TOPE_K_COMPROMISO)
EXTRAS_MORF = ("tono_calido", "saturacion_alta", "croma_alto", "color_con_gradiente", "tono_verdoso",
               "tono_rojizo", "tono_amarillento", "rojo_dominante", "naranja_dominante", "amarillo_dominante",
               "verde_dominante")
EXTRAS_CAL = ("sin_oscurecimiento", "color_sin_gradiente", "croma_alto")


def _pasos_iniciales():
    """Correcciones y compromisos aplicados sobre la receta original reentrenada (se informa cada efecto)."""
    return [
        ("P1", "etiquetado de categorias con prototipos finales", "obligatoria",
         {"asignaciones online (original)": lambda v: v,
          "prototipos finales": lambda v: replace(v, mapeo="prototipos_finales")}),
        ("P2", "cobertura de la etapa 1 (manzanas de todas las calidades)", "obligatoria",
         {"solo manzanas good (original)": lambda v: v,
          "manzanas de todas las calidades": lambda v: replace(v, cobertura="todas")}),
        ("P3", "eleccion de rho en validacion", "obligatoria",
         {"rho fijo del entregado": lambda v: v,
          "rho elegido en validacion": lambda v: replace(v, rho_morf=None, rho_cal=None)}),
        ("P4", "umbrales de calidad solo con entrenamiento", "obligatoria",
         {"constantes del lote original": lambda v: replace(v, legado=False, umbral_cal="fijo", tallo="v1",
                                                            tono="lineal", umbral_morf="fijo"),
          "medianas del entrenamiento": lambda v: replace(v, legado=False, umbral_cal="mediana", tallo="v1",
                                                          tono="lineal", umbral_morf="fijo")}),
        ("P5", "tiene_tallo: mismo criterio (convexidad) compatible con OpenCV 5", "correccion",
         {"v1 (vale 0 con OpenCV 5)": lambda v: v,
          "convexidad corregida": lambda v: replace(v, tallo="convexidad")}),
        ("P6", "desvio de tono circular (color_uniforme / color_homogeneo)", "correccion",
         {"lineal (v1)": lambda v: v, "circular ponderado": lambda v: replace(v, tono="circular")}),
        ("P7", "cortes morfologicos = los originales (compromiso con el primer avance)", "obligatoria",
         {"constantes fijas (equivalente)": lambda v: v,
          "cortes originales ('compromiso')": lambda v: replace(v, umbral_morf="compromiso", umbral_cal="compromiso")}),
        ("P8", f"criterio de exito del primer avance: rho admisible solo si forma <= {TOPE_K_COMPROMISO} categorias",
         "obligatoria",
         {"sin tope": lambda v: v, f"tope de {TOPE_K_COMPROMISO} categorias": lambda v: replace(v, tope_k=TOPE_K_COMPROMISO)}),
    ]


def _fila(paso, opcion, res, base_res, adoptada, nota="", tipo=""):
    f = {"paso": paso, "opcion": opcion, "tipo": tipo, "adoptada": "si" if adoptada else "no", "nota": nota}
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
    f["unitarias_morfologia"] = round(ag["unitarias_morfologia"][0], 2)
    f["fallos"] = ag["fallos"]
    for part in ("validacion", "prueba"):
        d, se = E.diferencia_pareada(base_res, res, part, CRITERIO)
        f[f"delta_{part}"], f[f"se_delta_{part}"] = round(d, 4), round(se, 4)
    return f


def _decidir(tipo, nombres, deltas, adoptable=True):
    mejor = max(nombres[1:], key=lambda n: deltas[n][0])
    d, se = deltas[mejor]
    if not adoptable:
        return None
    if tipo == "obligatoria":
        return nombres[1]
    if tipo == "correccion":
        return mejor if d >= -MARGEN else None
    return mejor if (d >= MARGEN and d >= se) else None


def cmd_buscar(args):
    t0 = time.time()
    os.makedirs(DIR_SALIDA, exist_ok=True)
    filas, log = [], []
    with ProcessPoolExecutor(args.procesos, initializer=_init, initargs=(args.mediciones, args.imagenes)) as ex:
        config = E.ORIGINAL
        base = _ejecutar(ex, config)
        filas.append(_fila("S0", "original reentrenado (referencia)", base, base, True, "receta del primer avance", "ref"))

        def paso(pid, desc, tipo, opciones, adoptable=True):
            nonlocal config, base
            nombres = list(opciones)
            cands = [(n, opciones[n](config)) for n in nombres]
            resultados = {n: (base if c == config else _ejecutar(ex, c)) for n, c in cands}
            deltas = {n: E.diferencia_pareada(base, resultados[n], "validacion", CRITERIO) for n in nombres}
            elegido = _decidir(tipo, nombres, deltas, adoptable)
            for n, c in cands:
                if elegido is not None:
                    adoptada = n == elegido
                    nota = {"obligatoria": "obligatoria (metodo o compromiso del primer avance)",
                            "correccion": "correccion de defecto, no inferior (>= -margen)",
                            "mejora": "mejora >= margen y > 1 SE"}[tipo] if adoptada else ""
                else:
                    adoptada = c == config
                    nota = "" if adoptada else ("experimental: no adoptable en la variante principal" if not adoptable
                                                else "sin evidencia suficiente")
                filas.append(_fila(f"{pid} {desc}", n, resultados[n], base, adoptada, nota, tipo))
            if elegido is not None:
                config, base = dict(cands)[elegido], resultados[elegido]
            mejor = max(nombres[1:], key=lambda n: deltas[n][0])
            log.append({"paso": pid, "descripcion": desc, "tipo": tipo, "elegida": elegido or nombres[0],
                        "mejor_alternativa": mejor, "delta_validacion": round(deltas[mejor][0], 4),
                        "se": round(deltas[mejor][1], 4)})
            print(f"{pid} {desc}: -> {elegido or nombres[0]} (mejor alternativa {mejor}: delta val "
                  f"{deltas[mejor][0]:+.4f} +- {deltas[mejor][1]:.4f})", flush=True)

        for pid, desc, tipo, op in _pasos_iniciales():
            paso(pid, desc, tipo, op)
        paso("N1", "normalizacion de la iluminacion antes de segmentar y medir", "mejora",
             {"ninguna (primer avance)": lambda v: v,
              "estirar contraste (percentiles 1-99)": lambda v: replace(v, normalizacion="estirar"),
              "balance de blancos mundo gris": lambda v: replace(v, normalizacion="gris_mundo"),
              "gamma a luminosidad media 0.5": lambda v: replace(v, normalizacion="gamma_media")})
        # --- descriptores cromaticos adicionales: seleccion hacia adelante, uno por vez ---
        for etapa, catalogo, maximo in (("morfologia", EXTRAS_MORF, 3), ("calidad", EXTRAS_CAL, 2)):
            for ronda in range(maximo):
                actuales = config.extras_morf if etapa == "morfologia" else config.extras_cal
                libres = [e for e in catalogo if e not in actuales]
                if not libres:
                    break
                campo = "extras_morf" if etapa == "morfologia" else "extras_cal"
                op = {"(sin agregar)": lambda v: v}
                for e in libres:
                    op[f"+ {e}"] = (lambda e: lambda v: replace(v, **{campo: tuple(getattr(v, campo)) + (e,)}))(e)
                paso(f"X{etapa[:3]}{ronda + 1}", f"descriptor cromatico adicional en {etapa} (ya: {list(actuales)})",
                     "mejora", op)
                if (config.extras_morf if etapa == "morfologia" else config.extras_cal) == actuales:
                    break
        paso("M1", "desempate de etiquetas", "mejora",
             {"indeterminada": lambda v: v, "mayoritaria_global": lambda v: replace(v, desempate="mayoritaria_global")})
        if config.extras_morf or config.extras_cal:
            paso("M3", "medicion del color de los descriptores adicionales", "mejora",
                 {"mascara completa": lambda v: v, "nucleo (sin borde fruta/fondo)": lambda v: replace(v, color="nucleo"),
                  "balance de blancos (mundo gris)": lambda v: replace(v, color="balanceado"),
                  "nucleo + balance de blancos": lambda v: replace(v, color="nucleo_balanceado")})
        paso("M4", "orden de presentacion durante el entrenamiento", "mejora",
             {"aleatorio fijo": lambda v: v, "mas activos primero": lambda v: replace(v, orden="densidad_desc"),
              "menos activos primero": lambda v: replace(v, orden="densidad_asc"),
              "canonico (menos activos primero y lexicografico; determinista)": lambda v: replace(v, orden="canonico_asc")})
        paso("M2", "pasadas de entrenamiento", "mejora",
             {"una pasada": lambda v: v, "hasta estabilizar (max 10)": lambda v: replace(v, epocas=10)})
        paso("M5", "criterio para elegir rho de la etapa 1", "mejora",
             {"F1 macro de la etapa": lambda v: v, "cadena en validacion": lambda v: replace(v, criterio_rho="cadena")})
        paso("M6", "soporte minimo para etiquetar una categoria", "mejora",
             {"1 muestra": lambda v: v, "2 muestras": lambda v: replace(v, soporte_min=2),
              "3 muestras": lambda v: replace(v, soporte_min=3)})
        paso("M7", "coincidencia minima de la etapa 1 para derivar a calidad", "mejora",
             {"basta resonar": lambda v: v, "coincidencia >= 0.75": lambda v: replace(v, derivacion_min=0.75),
              "coincidencia = 1.0": lambda v: replace(v, derivacion_min=1.0)})
        paso("M8", "segmentacion", "mejora",
             {"v1 (saturacion + Otsu)": lambda v: v, "v2 (candidatos + puntaje)": lambda v: replace(v, segmentador="v2")})
        principal = config
        # --- experimentales: no adoptables, se miden sobre la principal ---
        paso("E1", "tiene_tallo por protuberancia (otro detector, SIN anotaciones de referencia)", "mejora",
             {"convexidad (principal)": lambda v: v, "protuberancia": lambda v: replace(v, tallo="protuberancia")},
             adoptable=False)
        paso("E2", "umbrales morfologicos por mediana (cambia el significado de las variables)", "mejora",
             {"cortes originales (principal)": lambda v: v, "mediana del entrenamiento": lambda v: replace(v, umbral_morf="mediana")},
             adoptable=False)
        paso("E3", "sin el tope de categorias del primer avance", "mejora",
             {f"tope {TOPE_K_COMPROMISO} (principal)": lambda v: v, "sin tope": lambda v: replace(v, tope_k=None)},
             adoptable=False)
        paso("E4", "comite de 11 redes ART1 con ordenes distintos y voto de etiqueta (cambia la arquitectura: costo 11x)",
             "mejora",
             {"una red (principal)": lambda v: v,
              "comite de 11 redes": lambda v: replace(v, comite=11, orden="aleatorio")}, adoptable=False)
        assert config == principal
    ruta = os.path.join(DIR_SALIDA, "busqueda.csv")
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    with open(os.path.join(DIR_SALIDA, "config_principal.json"), "w", encoding="utf-8") as f:
        json.dump({"variante": asdict(config), "decisiones": log, "semillas_internas": list(SEMILLAS),
                   "semilla_informe": SEMILLA_INFORME, "criterio": CRITERIO, "margen": MARGEN}, f, indent=1)
    print(f"\nBusqueda terminada en {time.time() - t0:.0f}s -> {ruta}")
    return COD_OK


def main():
    ap = argparse.ArgumentParser(description="Segunda ronda: busqueda de mejoras con particion de informe reservada")
    ap.add_argument("comando", choices=("buscar",))
    ap.add_argument("--mediciones", default=os.path.join(cfg.DIR_RESULTADOS, "mediciones"))
    ap.add_argument("--imagenes", default=cfg.DIR_IMAGENES, help="solo se usa para listar nombres de archivo")
    ap.add_argument("--procesos", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    if not os.path.isfile(os.path.join(args.mediciones, "mediciones_v1.csv")):
        print("ERROR: faltan las mediciones. Genere con: python scripts/extraer_mediciones.py", file=sys.stderr)
        return COD_ENTRADA
    return cmd_buscar(args)


if __name__ == "__main__":
    sys.exit(main())
