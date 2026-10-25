/* NumPy and the repository wheel execute inside Pyodide, off the UI thread. */
let initialized = false;
const ready = (async () => {
  postMessage({ progress: "Loading Python · first visit may take a minute…" });
  importScripts("./runtime/pyodide.js");
  const py = await loadPyodide({
    indexURL: new URL("./runtime/", self.location.href).href,
  });
  await py.loadPackage(["numpy", "micropip"]);
  postMessage({ progress: "Loading cricket environment…" });
  const base = new URL("./", self.location.href).href;
  const wheels = await (await fetch(base + "dependency-wheels.json")).json();
  const manifestResponse = await fetch(base + "app-wheel.json", {
    cache: "no-store",
  });
  if (!manifestResponse.ok)
    throw new Error(
      "Application manifest is unavailable. Please reload the page.",
    );
  const manifest = await manifestResponse.json();
  py.globals.set(
    "dependency_wheels",
    JSON.stringify(wheels.map((name) => base + name)),
  );
  py.globals.set(
    "wheel_url",
    base + manifest.wheel + "#sha256=" + manifest.sha256,
  );
  await py.runPythonAsync(`
import micropip
import json
await micropip.install(json.loads(dependency_wheels), deps=False)
await micropip.install(wheel_url, deps=False)
import json
from cricket_rl.session import Session
session = Session()
def browser_request(raw):
    request = json.loads(raw)
    path, body = request['path'], request.get('body') or {}
    if path == 'state': result = session.state()
    elif path == 'reset': result = session.reset(body)
    elif path == 'bowl': result = session.bowl(body)
    elif path == 'field':
        session.env.set_field(body.get('fielders'))
        result = session.state()
    else: raise ValueError('Unsupported browser operation')
    return json.dumps(result)
`);
  try {
    const cardResponse = await fetch(base + "models/model-card.json");
    if (!cardResponse.ok) throw new Error("Missing model card");
    const card = await cardResponse.json();
    py.FS.mkdirTree("/demo_models");
    for (const role of ["batting", "bowling"]) {
      const model = card.models[role];
      const response = await fetch(base + "models/" + model.file);
      if (!response.ok) throw new Error("Missing demo weights");
      const buffer = await response.arrayBuffer();
      const hash = Array.from(
        new Uint8Array(await crypto.subtle.digest("SHA-256", buffer)),
      )
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("");
      if (hash !== model.sha256) throw new Error("Demo checksum mismatch");
      py.FS.writeFile("/demo_models/" + role + ".npz", new Uint8Array(buffer));
    }
    py.runPython(`
from cricket_rl.course_experiment import load_policy
session.batting_model = load_policy('/demo_models/batting.npz','batting')
session.model = load_policy('/demo_models/bowling.npz','bowling')
`);
  } catch (error) {
    py.runPython(
      "session.model_notice = 'Demo weights could not load. Manual, random and rule-based policies remain available.'",
    );
  }
  initialized = true;
  return py;
})();
ready.catch(() => {}); // Requests report startup failures through the UI.
let queue = Promise.resolve();
self.onmessage = ({ data }) => {
  queue = queue.then(async () => {
    try {
      const py = await ready;
      py.globals.set("request_json", JSON.stringify(data));
      const result = JSON.parse(py.runPython("browser_request(request_json)"));
      postMessage({ id: data.id, result });
    } catch (error) {
      const lines = String(error.message).split("\n").filter(Boolean);
      const detail = lines[lines.length - 1] || "Unknown error";
      postMessage({
        id: data.id,
        fatal: !initialized,
        error: initialized
          ? detail
          : "Could not load the browser environment. " +
            detail +
            " Use Try connecting again.",
      });
    }
  });
};
