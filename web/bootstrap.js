/* Recoverable, bounded worker transport. No training runs on the UI thread. */
let worker = null;
let failure = null;
let sequence = 0;
const pending = new Map();
function failAll(message) {
  failure = new Error(message);
  failure.runtimeFailure = true;
  for (const task of pending.values()) {
    clearTimeout(task.timer);
    task.reject(failure);
  }
  pending.clear();
  if (worker) worker.terminate();
  worker = null;
}
function startWorker() {
  worker = new Worker("./python-worker.js");
  worker.onmessage = ({ data }) => {
    if (data.progress) {
      const status = document.getElementById("connection");
      if (status) status.textContent = data.progress;
      return;
    }
    const task = pending.get(data.id);
    if (!task) return;
    pending.delete(data.id);
    clearTimeout(task.timer);
    if (data.error) {
      const error = new Error(data.error);
      error.runtimeFailure = Boolean(data.fatal);
      task.reject(error);
    } else task.resolve(data.result);
    if (data.fatal) failAll(data.error);
  };
  worker.onerror = () =>
    failAll(
      "The browser environment could not start. Use “Try connecting again” to reload it.",
    );
}
window.cricketApi = (path, body) =>
  new Promise((resolve, reject) => {
    if (failure) {
      reject(failure);
      return;
    }
    if (!worker) startWorker();
    const id = ++sequence;
    const timer = setTimeout(
      () =>
        failAll(
          "Loading took too long. Check your connection and try connecting again.",
        ),
      90000,
    );
    pending.set(id, { resolve, reject, timer });
    worker.postMessage({ id, path, body });
  });
window.cricketApi.restart = () => {
  if (worker || pending.size) failAll("Environment restarted.");
  failure = null;
};

window.cricketApi.failed = () => Boolean(failure);
