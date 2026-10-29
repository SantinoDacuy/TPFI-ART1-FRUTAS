"""Pruebas pequenas del ciclo ART1: competencia, vigilancia, reset, resonancia,
aprendizaje, inferencia sin cambio de pesos, vector nulo y asimetria del match."""
import itertools

import pytest

from CarGross import ProliferacionError, RedART1, VectorNuloError


def red(n=6, rho=0.7, **kw):
    return RedART1(n, rho, **kw)


# --- ciclo completo ---------------------------------------------------------
def test_primer_patron_compromete_categoria_nueva():
    r = red()
    assert r.clasificar([1, 1, 1, 0, 0, 0]) == 0
    assert r.top_down == [[1, 1, 1, 0, 0, 0]]


def test_resonancia_actualiza_prototipo_con_and():
    r = red(rho=0.6)
    r.clasificar([1, 1, 1, 1, 0, 0])
    # |I^T|/|I| = 3/4 >= 0.6 -> resuena y el prototipo pasa a la interseccion
    assert r.clasificar([1, 1, 1, 0, 0, 0]) == 0
    assert r.top_down == [[1, 1, 1, 0, 0, 0]]


def test_vigilancia_rechaza_y_se_crea_categoria():
    r = red(rho=0.8)
    r.clasificar([1, 1, 1, 1, 0, 0])
    # 2/4 = 0.5 < 0.8 -> reset y categoria nueva
    assert r.clasificar([1, 1, 0, 0, 1, 1]) == 1
    assert len(r.top_down) == 2
    assert r.top_down[0] == [1, 1, 1, 1, 0, 0]  # la categoria reseteada no cambia


def test_competencia_gana_mayor_entrada_neta_y_reset_pasa_al_siguiente():
    r = red(rho=0.7)
    r.top_down = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0]]
    # I = 111000: neto_0 = 2*3/(1+6)=0.857 ; neto_1 = 2*3/(1+3)=1.5 -> compite 1 y resuena
    res = r.evaluar([1, 1, 1, 0, 0, 0])
    assert res["ganador"] == 1 and res["resuena"]
    # I = 110000 con prototipo A=110000 y B=111111 (B gana en orden de indice si empatan)
    r.top_down = [[1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]]
    res = r.evaluar([0, 0, 0, 0, 1, 1])
    assert res["ganador"] == 1
    # el primer candidato (mayor neto) se resetea y se prueba el siguiente
    r.top_down = [[1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0]]
    res = r.evaluar([1, 1, 1, 1, 0, 0])
    assert res["ganador"] == 1
    assert [c["cluster"] for c in res["candidatos"]] == [1]


def test_reset_registra_candidatos_rechazados():
    r = red(rho=0.9)
    r.top_down = [[1, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]
    res = r.evaluar([1, 1, 1, 1, 0, 0])
    assert not res["resuena"] and res["ganador"] is None
    assert res["motivo"] == "vigilancia"
    assert [c["cluster"] for c in res["candidatos"]] == [0, 1]
    assert all(not c["resuena"] for c in res["candidatos"])


def test_empate_exacto_gana_menor_indice():
    r = red(rho=0.5)
    r.top_down = [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0]]
    res = r.evaluar([1, 1, 1, 1, 0, 0])
    assert res["candidatos"][0]["cluster"] == 0


def test_equivalencia_con_uncommitted_nunca_gana_la_competencia():
    # Una categoria nueva solo se crea si ninguna comprometida resuena, aunque
    # una categoria "no comprometida" hubiera tenido mayor entrada neta.
    r = red(rho=0.3)
    r.top_down = [[1, 0, 0, 1, 1, 1]]
    # I=111000: solape 1, ratio 1/3 >= 0.3 -> resuena con la existente
    assert r.clasificar([1, 1, 1, 0, 0, 0]) == 0
    assert len(r.top_down) == 1


# --- inferencia sin modificar pesos ------------------------------------------
def test_inferir_no_modifica_pesos():
    r = red(rho=0.7)
    for x in ([1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 1, 0]):
        r.clasificar(x)
    antes, huella = [list(t) for t in r.top_down], r.huella()
    for x in itertools.product([0, 1], repeat=6):
        r.evaluar(list(x))
        r.clasificar(list(x), aprender=False)
    assert r.top_down == antes and r.huella() == huella


def test_inferencia_sin_resonancia_devuelve_novedad_y_no_crea_categoria():
    r = red(rho=0.9)
    r.clasificar([1, 1, 1, 1, 0, 0])
    assert r.clasificar([0, 0, 0, 0, 1, 1], aprender=False) is None
    assert len(r.top_down) == 1


# --- vector nulo ---------------------------------------------------------------
def test_vector_nulo_politica_categoria_propia():
    r = red()
    r.clasificar([1, 1, 0, 0, 0, 0])
    assert r.clasificar([0] * 6, aprender=False) is None            # aun no aprendido
    assert r.evaluar([0] * 6)["motivo"] == "vector_nulo_no_aprendido"
    k = r.clasificar([0] * 6)                                         # se compromete
    assert r.top_down[k] == [0] * 6
    res = r.evaluar([0] * 6)
    assert res["resuena"] and res["ganador"] == k and res["ratio"] == 1.0
    # un patron no nulo nunca resuena con el prototipo nulo (|I^0| = 0 < rho*|I|)
    assert r.evaluar([1, 0, 0, 0, 0, 0])["ganador"] != k
    assert r.clasificar([1, 1, 0, 0, 0, 0]) != k
    assert r.top_down[k] == [0] * 6                                   # y no lo cambia


def test_vector_nulo_politica_rechazar():
    r = red(politica_nulo="rechazar")
    r.clasificar([1, 1, 0, 0, 0, 0])
    with pytest.raises(VectorNuloError):
        r.clasificar([0] * 6)
    res = r.evaluar([0] * 6)
    assert not res["resuena"] and res["motivo"] == "vector_nulo_rechazado"
    assert r.clasificar([0] * 6, aprender=False) is None
    assert len(r.top_down) == 1


def test_politica_nulo_invalida():
    with pytest.raises(ValueError):
        red(politica_nulo="inventada")


# --- asimetria del match -------------------------------------------------------
def test_match_es_asimetrico_y_no_es_igualdad():
    a, b = [1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]
    r1 = red(rho=0.1); r1.top_down = [b]
    r2 = red(rho=0.1); r2.top_down = [a]
    # |I^T|/|I|: I=a contra T=b -> 2/4 ; I=b contra T=a -> 2/2
    assert r1.evaluar(a)["ratio"] == 0.5
    assert r2.evaluar(b)["ratio"] == 1.0
    # match del 100% con vectores distintos: no es igualdad
    r3 = red(rho=1.0); r3.top_down = [a]
    res = r3.evaluar(b)
    assert res["resuena"] and res["ratio"] == 1.0 and a != b


def test_match_100_por_ciento_no_implica_misma_categoria_de_referencia():
    # dos patrones distintos con match 1.0 contra el mismo prototipo comparten
    # categoria aunque tengan etiquetas de referencia distintas
    r = red(rho=1.0)
    r.clasificar([1, 1, 0, 0, 0, 0])
    assert r.clasificar([1, 1, 0, 0, 0, 0]) == 0


# --- entradas invalidas ---------------------------------------------------------
@pytest.mark.parametrize("malo", [[1, 0, 1], [1, 0, 2, 0, 0, 0], [0.5] * 6, [None] * 6])
def test_entrada_invalida_levanta_valueerror(malo):
    with pytest.raises(ValueError):
        red().evaluar(malo)
    with pytest.raises(ValueError):
        red().clasificar(malo)


@pytest.mark.parametrize("rho", [0, -0.1, 1.5])
def test_rho_invalido(rho):
    with pytest.raises(ValueError):
        RedART1(4, rho)


def test_proliferacion_levanta_error():
    r = RedART1(4, 1.0, max_categorias=2)
    r.clasificar([1, 0, 0, 0])
    r.clasificar([0, 1, 0, 0])
    with pytest.raises(ProliferacionError):
        r.clasificar([0, 0, 1, 0])
