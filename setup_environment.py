"""Install the complete Windows environment: python setup_environment.py."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import urllib.request


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    if os.name != "nt":
        raise SystemExit("Use conda env create -f environment.yml on non-Windows systems.")
    root = Path(__file__).resolve().parent
    local = Path(os.environ["LOCALAPPDATA"]) / "BookRecommender"
    if (local / "env311" / "python.exe").is_file():
        print("Environment already installed. Run: python run.py -m experiments.run_all --mode quick")
        return
    local.mkdir(parents=True, exist_ok=True)
    # libmamba gặp lỗi khi đường dẫn spec/cache chứa tiếng Việt.
    spec = local / "environment.yml"
    shutil.copyfile(root / "environment.yml", spec)
    executable = root / ".cache" / "micromamba" / "Library" / "bin" / "micromamba.exe"
    if not executable.is_file():
        executable.parent.mkdir(parents=True, exist_ok=True)
        archive = root / ".cache" / "micromamba" / "package.tar.bz2"
        print("Downloading micromamba from micro.mamba.pm...", flush=True)
        with urllib.request.urlopen("https://micro.mamba.pm/api/micromamba/win-64/latest", timeout=120) as source:
            with archive.open("wb") as target:
                shutil.copyfileobj(source, target)
        # Chỉ lấy executable đã chỉ định, không giải nén đường dẫn tùy ý từ tar.
        with tarfile.open(archive, "r:bz2") as package:
            with package.extractfile("Library/bin/micromamba.exe") as source:
                with executable.open("wb") as target:
                    shutil.copyfileobj(source, target)
    subprocess.run([str(executable), "create", "--root-prefix", str(local / "mamba"),
                    "--prefix", str(local / "env311"), "--file", str(spec), "--yes"], check=True)
    print("Ready. Run: python run.py -m experiments.run_all --mode quick", flush=True)


if __name__ == "__main__":
    main()
