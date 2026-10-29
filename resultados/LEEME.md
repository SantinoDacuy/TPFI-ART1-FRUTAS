# resultados/

Todo se regenera con los scripts (ver README, sección "Reproducir los resultados"). No se edita a mano.

| archivo | qué es | cómo se genera |
|---|---|---|
| `baseline_original.md/.json`, `baseline_fotos.csv` | línea de base: el primer avance tal como fue entregado y los hallazgos comprobados | `python scripts/baseline_original.py` |
| `mediciones/mediciones_v1.csv`, `mediciones_v2.csv` | medidas continuas de las 339 fotos con cada segmentador | `python scripts/extraer_mediciones.py` (requiere las fotos) |
| `mediciones/fondo.csv` | descriptores del entorno de la foto (solo para el diagnóstico de fondo) | ídem |
| `ablacion.csv`, `config_final.json` | cadena de decisiones sobre validación y configuración elegida | `python scripts/experimentos.py ablacion` |
| `comparacion.md`, `final_resumen.json`, `final_fotos.csv` | original vs. mejorada: tablas, matrices de confusión, categorías, predicciones por foto | `python scripts/experimentos.py final` |
| `barrido_rho.csv`, `barrido_orden.md`, `sensibilidad_orden.csv` | barrido de ρ y sensibilidad al orden | `python scripts/experimentos.py barrido` |
| `diagnostico_fondo.md` | ¿cuánto se acierta mirando solo el entorno de la foto? | `python scripts/experimentos.py fondo` |

Las mediciones son la única parte que necesita las fotos (carpeta fuera del repositorio por tamaño); con ellas
versionadas, los experimentos se reproducen sin las fotos.

## Segunda ronda: `ronda2/` (protocolo con bloque de informe reservado)

| archivo | qué es | cómo se genera |
|---|---|---|
| `ronda2/busqueda.csv`, `config_principal.json` | cadena de decisiones sobre validación y configuración principal | `python scripts/busqueda.py buscar` |
| `ronda2/comparacion.md/.json`, `informe_fotos_principal.csv` | comparación de variantes bajo las mismas particiones, matrices, orden, fondo | `python scripts/comparar_ronda2.py` |
| `ronda2/robustez_*.md/.csv` | estabilidad ante sustitución de fondo e iluminación | `python scripts/robustez.py ...` |
| `ronda2/historial_busquedas/`, `ronda2/borrador_sin_fracciones/` | búsquedas anteriores (antes de agregar candidatos): se conservan por transparencia | |
| `mediciones/mediciones_v1[_<normalizacion>].csv`, `mediciones_v2.csv`, `fondo.csv` | medidas continuas (incluye descriptores nuevos) | `python scripts/extraer_mediciones.py` (requiere las fotos) |
| `verificacion/` | registro de la instalación en un entorno nuevo | `python scripts/verificar_instalacion.py` |

Los archivos del nivel superior (`ablacion.csv`, `comparacion.md`, `barrido_*`, `final_*`, `diagnostico_fondo.md`, `config_final.json`) son de la **primera ronda**
(protocolo exploratorio); `ronda1/` guarda las mediciones originales de esa ronda.
