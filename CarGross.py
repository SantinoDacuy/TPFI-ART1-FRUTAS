"""
CarGross.py
Red de Carpenter-Grossberg (ART1) para clasificacion no supervisada de
frutas. TFI Redes Neuronales - Grupo 2 - UADER FCyT.

ESTADO: pendiente de implementar (proximo paso del roadmap).
Este archivo documenta la arquitectura acordada para que cualquiera del
grupo pueda continuar desde aca.

Entrada: CSV con columna fruta_id + N variables binarias (0/1).
         Las columnas de metadata (tipo_real, calidad_real) se leen pero
         NO se pasan a la red - son solo para validar resultados despues.

Pendiente de implementar:
  - inicializar_pesos(n_features, n_max_categorias)
  - entrada -> capa F1 -> competencia en F2 (elegir nodo ganador)
  - test de vigilancia: |I ^ w_j| / |I| >= rho ?
      si pasa -> actualizar w_j = I ^ w_j (AND logico), aprendizaje rapido
      si no pasa -> resetear ese nodo, probar el siguiente mejor candidato
      si ninguno pasa -> comprometer un nodo nuevo (categoria nueva)
  - manejo de corridas infructuosas: si con el rho dado se generan mas
    categorias que registros/2 (proliferacion descontrolada), cortar y
    avisar en vez de colgarse.
  - CLI: python CarGross.py --csv datasets/dataset_morfologia.csv --rho 0.8
"""

# TODO: implementar segun la notacion de Lau.pp5.a.11 / Lau.pp12.a.14
# (subir esos PDFs al chat cuando programemos esta parte para que la
# notacion coincida con la bibliografia de la materia)

raise NotImplementedError("CarGross.py todavia no está implementado - ver docstring")
