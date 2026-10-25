"""Build the GitHub Pages demo from the production UI and Python wheel."""

import argparse
import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("site"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Build into an empty directory so an older cached wheel is never selected.
    with tempfile.TemporaryDirectory(prefix="cricket-wheel-") as temporary:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-build-isolation",
                str(root),
                "--wheel-dir",
                temporary,
            ],
            check=True,
        )
        wheel = next(Path(temporary).glob("cricket_strategy_rl-*.whl"))
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        package_dir = output / "packages" / digest[:16]
        package_dir.mkdir(parents=True, exist_ok=True)
        wheel_name = wheel.name
        shutil.copyfile(wheel, package_dir / wheel_name)
    (output / "app-wheel.json").write_text(
        json.dumps(
            {
                "wheel": (package_dir / wheel_name).relative_to(output).as_posix(),
                "sha256": digest,
            }
        )
    )
    html = (root / "cricket_rl/dashboard.html").read_text()
    html = html.replace('href="/dashboard.css"', 'href="./dashboard.css"').replace(
        'src="/dashboard.js"', 'src="./dashboard.js"'
    )
    html = html.replace(
        '<script src="./dashboard.js" defer>',
        '<script src="./bootstrap.js" defer></script><script src="./dashboard.js" defer>',
    )
    html = (
        html.replace('href="/learn', 'href="./learn.html')
        .replace('href="/api/schema"', 'href="./schema.json"')
        .replace('href="/"', 'href="./"')
    )
    html = html.replace(
        "Environment connected", "Browser environment connected"
    ).replace('href="/favicon.svg"', 'href="./favicon.svg"')
    html = html.replace(
        "<main>",
        """<main><p class="help">Browser-only playground · first load downloads about 18 MB. Tabular trained policies are included. Training and neural checkpoints use the Python version. <a href="./replay.html">Open an instant replay instead</a> · <a href="./models/model-card.json">Demo model card</a>.</p>""",
        1,
    )
    public_url = "https://tvpian.github.io/RL_Cricket_Simulation_Engine/"
    html = html.replace(
        "</head>",
        f'<meta property="og:url" content="{public_url}"><meta property="og:image" content="{public_url}preview.png"><meta name="twitter:card" content="summary_large_image"></head>',
    )
    shutil.copyfile(root / "web" / "preview.png", output / "preview.png")
    shutil.copyfile(root / "cricket_rl" / "favicon.svg", output / "favicon.svg")
    (output / "index.html").write_text(html)
    for name in ["dashboard.css", "dashboard.js"]:
        shutil.copyfile(root / "cricket_rl" / name, output / name)
    guide = (
        (root / "cricket_rl/learn.html")
        .read_text()
        .replace('href="/dashboard.css"', 'href="./dashboard.css"')
        .replace('href="/api/schema"', 'href="./schema.json"')
        .replace('href="/"', 'href="./"')
        .replace('href="/#lessonPanel"', 'href="./#lessonPanel"')
    )
    (output / "learn.html").write_text(guide)
    for name in ["bootstrap.js", "python-worker.js"]:
        shutil.copyfile(root / "web" / name, output / name)
    shutil.copyfile(root / "web" / "replay.html", output / "replay.html")
    shutil.copytree(root / "web" / "models", output / "models", dirs_exist_ok=True)
    from vendor_runtime import bundle

    bundle(output)
    from cricket_rl.api import EnvironmentAPI

    (output / "schema.json").write_text(
        json.dumps(
            {
                "environments": [
                    EnvironmentAPI.schema(role) for role in ["batting", "bowling"]
                ]
            },
            indent=2,
        )
    )
    (output / ".nojekyll").touch()
    print(f"Browser demo built at {output}")


if __name__ == "__main__":
    main()
