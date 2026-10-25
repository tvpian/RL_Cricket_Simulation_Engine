# Browser-only demo

The static site reuses the production UI and the **same Python simulation** via
Pyodide in a Web Worker. It supports batting, bowling, scripted/random policies,
guided lessons, field editing, journals and snapshots without a Python server.
A narrow browser protocol adapter avoids unavailable multiprocessing vector
environments in WebAssembly. Desktop training and registered environments still
use real Gymnasium. This adapter does not duplicate the physics or scoring.
PyTorch checkpoint inference and training remain features of the Python version;
those options are not offered in the browser-only demo.

The build bundles Python, NumPy and Gymnasium with the site (about 18 MB).
The builder fetches pinned artifacts using HTTPS and checks Pyodide package hashes
against its release manifest. Browser requests stay on the site origin; it does
not fetch Python packages from third-party hosts at runtime. No match or journal
data is uploaded to a hosted backend. First load may take a minute. Files still
need HTTP hosting; a service-worker offline cache is not implemented.

Build and preview from the repository root:

```bash
pip install -e .
pip install 'setuptools>=68' wheel
python tools/build_site.py
python -m http.server 8010 --directory site
```

Open the local preview on your own computer at port 8010. Serve the directory over
HTTP; browsers generally prevent Web Workers loading from `file://`. For a replay
that opens directly as a downloaded file, use the separate offline replay exporter.

## GitHub Pages

1. Push to `master`.
2. In **Settings → Pages → Build and deployment**, choose **GitHub Actions**.
3. Open **Actions → Browser demo**. Run or re-run the workflow.
4. Open the deployment URL shown in its `github-pages` environment.

The expected project URL is
`https://tvpian.github.io/RL_Cricket_Simulation_Engine/`, but it is only live after
the Pages deployment succeeds. A Git push alone does not establish publication.
If Pages is not enabled, the build artifact can still be downloaded from the
workflow's build job. It must be served over HTTP as described above.

`bootstrap.js` bridges UI requests to the worker. `python-worker.js` loads the
packaged environment and manages one local session. `tools/build_site.py` copies
assets and builds the Python wheel; generated `site/` is ignored by Git.
