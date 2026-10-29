"""Etiquetado posterior de categorias con los prototipos FINALES: coherencia con la
inferencia, empates, categorias sin muestras y entradas sin resonancia."""
from CarGross import RedART1, construir_mapeo, entrenar, mapeo_online, pureza


def entrenada(X, rho):
    red = RedART1(len(X[0]), rho)
    info = entrenar(red, X)
    return red, info


# Tres patrones encadenados: el primero deja de resonar con el prototipo final
X_CADENA = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]]


def test_los_prototipos_cambian_despues_de_asignar():
    red, info = entrenada(X_CADENA, 0.7)
    assert info["asignaciones_online"] == [0, 0, 0]          # online: todos en la categoria 0
    assert red.top_down == [[1, 1, 0, 0, 0, 0]]              # prototipo final = AND de los tres
    # con el prototipo final, los dos primeros patrones ya NO superan la vigilancia (2/6 y 2/4 < 0.7)
    assert not red.evaluar(X_CADENA[0])["resuena"]
    assert not red.evaluar(X_CADENA[1])["resuena"]
    assert red.evaluar(X_CADENA[2])["resuena"]


def test_muestra_sin_resonancia_no_vota_y_se_informa():
    red, _ = entrenada(X_CADENA, 0.7)
    mapeo, reinf = construir_mapeo(red, X_CADENA, ["a", "b", "b"])
    assert reinf["n_sin_resonancia"] == 2 and reinf["distribucion_sin_resonancia"] == {"a": 1, "b": 1}
    assert reinf["n_resuenan"] == 1
    assert mapeo["0"]["distribucion"] == {"b": 1} and mapeo["0"]["etiqueta"] == "b"
    assert mapeo["0"]["pureza"] == 1.0 and mapeo["0"]["estado"] == "etiquetada"


def test_el_mapeo_online_no_describe_lo_mismo_que_el_final():
    red, info = entrenada(X_CADENA, 0.7)
    et = ["a", "a", "b"]
    online = mapeo_online(info["asignaciones_online"], et)
    final, _ = construir_mapeo(red, X_CADENA, et)
    assert online["0"]["etiqueta"] == "a" and online["0"]["n"] == 3     # mayoria de asignaciones online
    assert final["0"]["n"] == 1 and final["0"]["distribucion"] == {"b": 1}
    assert final["0"]["etiqueta"] == "b"            # la etiqueta cambia al usar los prototipos finales


def test_empate_deja_la_categoria_sin_etiqueta_por_defecto():
    red = RedART1(4, 0.5)
    red.top_down = [[1, 1, 0, 0]]
    mapeo, _ = construir_mapeo(red, [[1, 1, 0, 0], [1, 1, 0, 0]], ["a", "b"])
    assert mapeo["0"]["etiqueta"] is None and mapeo["0"]["estado"] == "empate"
    assert mapeo["0"]["empatadas"] == ["a", "b"] and mapeo["0"]["pureza"] == 0.5


def test_desempate_por_frecuencia_global_es_determinista():
    red = RedART1(4, 0.5)
    red.top_down = [[1, 1, 0, 0], [0, 0, 1, 1]]
    X = [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]
    y = ["a", "b", "b", "b", "a"]
    m1, _ = construir_mapeo(red, X, y, desempate="mayoritaria_global")
    m2, _ = construir_mapeo(red, list(reversed(X)), list(reversed(y)), desempate="mayoritaria_global")
    assert m1["0"]["etiqueta"] == "b" and m1["0"]["estado"] == "etiquetada_por_desempate"
    assert m1 == m2


def test_categoria_sin_muestras_queda_sin_etiqueta():
    red = RedART1(4, 0.9)
    red.top_down = [[1, 1, 0, 0], [1, 1, 1, 1]]
    # ambas muestras resuenan con la categoria 0 (mayor neto): la 1 queda sin muestras
    mapeo, _ = construir_mapeo(red, [[1, 1, 0, 0], [1, 1, 0, 0]], ["a", "a"])
    assert mapeo["1"] == {"n": 0, "etiqueta": None, "estado": "sin_muestras", "pureza": None, "distribucion": {}}
    assert mapeo["0"]["etiqueta"] == "a"


def test_el_mapeo_usa_la_misma_inferencia_que_el_programa():
    """Para toda muestra, la categoria que cuenta al etiquetar es la que devuelve evaluar()."""
    X = [[1, 0, 1, 0, 1], [1, 1, 1, 0, 1], [0, 1, 0, 1, 0], [0, 0, 0, 1, 1], [1, 0, 1, 1, 1], [0, 1, 0, 1, 1]]
    y = ["a", "a", "b", "b", "a", "b"]
    red, _ = entrenada(X, 0.6)
    mapeo, reinf = construir_mapeo(red, X, y)
    cuenta = {}
    for x, e in zip(X, y):
        r = red.evaluar(x)
        if r["resuena"]:
            cuenta.setdefault(str(r["ganador"]), {}).setdefault(e, 0)
            cuenta[str(r["ganador"])][e] += 1
    for cid, info in mapeo.items():
        assert info["distribucion"] == cuenta.get(cid, {})
    assert sum(i["n"] for i in mapeo.values()) + reinf["n_sin_resonancia"] == len(X)


def test_vector_nulo_en_el_etiquetado_usa_la_categoria_nula():
    red = RedART1(3, 0.7)
    entrenar(red, [[1, 1, 0], [0, 0, 0], [0, 0, 0]])
    mapeo, _ = construir_mapeo(red, [[1, 1, 0], [0, 0, 0], [0, 0, 0]], ["a", "n", "n"])
    nula = [i for i, t in enumerate(red.top_down) if not any(t)][0]
    assert mapeo[str(nula)]["etiqueta"] == "n" and mapeo[str(nula)]["n"] == 2


def test_pureza_excluye_sin_resonancia_y_informa_cobertura():
    p, cob = pureza([0, 0, None, 1], ["a", "a", "b", "b"])
    assert p == 1.0 and cob == 0.75
    assert pureza([None, None], ["a", "b"]) == (0.0, 0.0)
