"""
scripts/verificar_instalacion.py
Comprueba la instalacion en un ENTORNO NUEVO: crea un entorno virtual temporal FUERA del repositorio, instala las
versiones fijadas de requirements-dev.txt, y ejecuta (1) la lista de paquetes, (2) las pruebas, (3) la demo y
(4) los ejemplos del README. Guarda un registro con versiones y resultados en resultados/verificacion/.

Requiere conexion a internet (pip). No modifica el entorno del proyecto ni el repositorio, salvo el registro.

Uso:
    python scripts/verificar_instalacion.py [--registro resultados/verificacion] [--conservar-entorno]
"""
import argparse
import datetime
import os
import platform
import subprocess
import sys
import tempfile

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def correr(cmd, log, titulo, cwd=RAIZ, timeout=1800):
    log.write(f"\n$ {' '.join(cmd)}\n")
    log.flush()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    log.write(p.stdout[-6000:] + ("\n[stderr]\n" + p.stderr[-3000:] if p.stderr.strip() else ""))
    log.write(f"\n[{titulo}: codigo de salida {p.returncode}]\n")
    log.flush()
    return p.returncode


def main(argv=None):
    ap = argparse.ArgumentParser(description="Verifica la instalacion en un entorno virtual nuevo")
    ap.add_argument("--registro", default=os.path.join(RAIZ, "resultados", "verificacion"))
    ap.add_argument("--conservar-entorno", action="store_true", help="no borra el entorno temporal al terminar")
    a = ap.parse_args(argv)
    os.makedirs(a.registro, exist_ok=True)
    resumen = []
    tmp = tempfile.mkdtemp(prefix="art1_venv_")
    entorno = os.path.join(tmp, "venv")
    ruta_log = os.path.join(a.registro, "verificacion_instalacion.log")
    with open(ruta_log, "w", encoding="utf-8") as log:
        log.write(f"Verificacion de instalacion - {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        log.write(f"Sistema: {platform.platform()} | Python base: {sys.version.split()[0]} | entorno temporal: {entorno}\n")
        resumen.append(("crear entorno virtual", correr([sys.executable, "-m", "venv", entorno], log, "venv")))
        py = os.path.join(entorno, "Scripts" if os.name == "nt" else "bin", "python" + (".exe" if os.name == "nt" else ""))
        resumen.append(("instalar requirements-dev.txt", correr(
            [py, "-m", "pip", "install", "--disable-pip-version-check", "-r", "requirements-dev.txt"], log, "pip install")))
        resumen.append(("versiones instaladas (pip freeze)", correr([py, "-m", "pip", "freeze"], log, "pip freeze")))
        resumen.append(("importar dependencias", correr(
            [py, "-c", "import sys, cv2, numpy, pytest; print(sys.version.split()[0], cv2.__version__, numpy.__version__, pytest.__version__)"],
            log, "import")))
        resumen.append(("pruebas (pytest)", correr([py, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"], log, "pytest")))
        resumen.append(("demo.py", correr([py, "demo.py"], log, "demo")))
        resumen.append(("probar_imagen.py --legado (ejemplo)", correr(
            [py, "probar_imagen.py", "--imagen", "ejemplos/naranja.jpg", "--legado", "--sin-grafico"], log, "probar_imagen legado")))
        resumen.append(("probar_imagen.py (variante principal)", correr(
            [py, "probar_imagen.py", "--imagen", "ejemplos/manzana_premium.jpg", "--sin-grafico"], log, "probar_imagen principal")))
        log.write("\n=== RESUMEN ===\n")
        for nombre, rc in resumen:
            log.write(f"{'OK   ' if rc == 0 else 'FALLO'} {nombre} (codigo {rc})\n")
    if not a.conservar_entorno:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    for nombre, rc in resumen:
        print(f"{'OK   ' if rc == 0 else 'FALLO'} {nombre}")
    print(f"Registro: {ruta_log}")
    return 0 if all(rc == 0 for _, rc in resumen) else 1


if __name__ == "__main__":
    sys.exit(main())
