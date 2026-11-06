"""Run a project module in the complete Windows environment: python run.py -m ..."""

import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    local_appdata = os.environ.get("LOCALAPPDATA")
    if not local_appdata:
        raise SystemExit("Windows LOCALAPPDATA is unavailable; use your Python interpreter directly.")
    prefix = Path(local_appdata) / "BookRecommender" / "env311"
    python = prefix / "python.exe"
    if not python.is_file():
        raise SystemExit("Complete environment is missing. See README: install using environment.yml.")
    env = os.environ.copy()
    # DLL của các gói Conda nằm trong Library/bin; chỉ thay PATH của tiến trình con.
    env["PATH"] = os.pathsep.join([str(prefix), str(prefix / "Library" / "bin"),
                                  str(prefix / "Scripts"), env.get("PATH", "")])
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    arguments = sys.argv[1:] or ["-m", "experiments.run_all", "--mode", "quick"]
    return subprocess.call([str(python), *arguments], cwd=root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
