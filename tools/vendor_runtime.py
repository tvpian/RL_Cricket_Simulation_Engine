"""Bundle verified Pyodide packages and pure-Python wheels for same-origin use."""

import hashlib
import json
import subprocess
import sys
from urllib.request import urlopen

BASE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/"


def bundle(output):
    runtime = output / "runtime"
    vendor = output / "vendor"
    runtime.mkdir(exist_ok=True)
    vendor.mkdir(exist_ok=True)

    def download(name, expected=None):
        path = runtime / name
        if not path.exists() or (
            expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            with urlopen(BASE + name, timeout=120) as response:
                content = response.read()
            if expected and hashlib.sha256(content).hexdigest() != expected:
                raise RuntimeError(f"Checksum mismatch: {name}")
            path.write_bytes(content)
        return path

    download("pyodide-lock.json")
    lock = json.loads((runtime / "pyodide-lock.json").read_text())
    for name in [
        "pyodide.js",
        "pyodide.asm.js",
        "pyodide.asm.wasm",
        "python_stdlib.zip",
    ]:
        download(name)
    visited = set()

    def package(name):
        if name in visited:
            return
        visited.add(name)
        meta = lock["packages"][name]
        for dep in meta["depends"]:
            package(dep)
        download(meta["file_name"], meta["sha256"])

    package("numpy")
    package("micropip")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--only-binary=:all:",
            "--no-deps",
            "--dest",
            str(vendor),
            "gymnasium==1.4.0",
            "cloudpickle==3.1.2",
            "typing_extensions==4.16.0",
            "Farama-Notifications==0.0.6",
        ],
        check=True,
    )
    wheels = ["vendor/" + p.name for p in sorted(vendor.glob("*.whl"))]
    (output / "dependency-wheels.json").write_text(json.dumps(wheels))
    files = [p for folder in [runtime, vendor] for p in folder.iterdir() if p.is_file()]
    (output / "runtime-integrity.json").write_text(
        json.dumps(
            {
                str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files
            },
            indent=2,
        )
    )
