"use strict";
const $ = (id) => document.getElementById(id);
const canvas = $("field"),
  ctx = canvas.getContext("2d");
const reducedMotion = window.matchMedia(
  "(prefers-reduced-motion: reduce)",
).matches;
let state = null,
  delivery = null,
  frame = 0,
  playing = !reducedMotion,
  editing = false;
let dragging = null,
  busy = false,
  last = 0,
  dirty = false,
  lastResponse = null;
let sceneRevision = 0,
  paintedScene = "",
  runtimeUnavailable = false;
let sideBounds = { minY: -4, maxY: 22, maxZ: 3 };
let journal = [],
  archivedRuns = [],
  activeLesson = null,
  lessonSteps = 0,
  runConfig = null;
const configIds = [
  "role",
  "seed",
  "overs",
  "style",
  "pitch",
  "target",
  "adaptive",
];
const actionIds = ["line", "length", "stance", "direction", "power"];
const lines = ["leg side", "at the stumps", "off side"];
const lengths = ["yorker", "good length", "short ball"];
const directions = ["wide left", "left", "straight", "right", "wide right"];
const lessons = {
  first: {
    title: "Meet the RL loop",
    tag: "BEGINNER · MANUAL PLAY",
    role: "bowling",
    steps: 3,
    description:
      "Make three decisions. Predict each outcome, then inspect the reward. Your goal is to understand the loop, not get a high score.",
    question:
      "Could the same action lead to a different outcome? What part is random?",
  },
  patterns: {
    title: "Can a batter predict you?",
    tag: "EXPERIMENT · BOWLING",
    role: "bowling",
    steps: 6,
    description:
      "Repeat a straight yorker, then change its length. Watch the current batter’s anticipation. Each batter has a separate memory.",
    question:
      "Did anticipation change? Why would one innings be too little evidence that a strategy is better?",
  },
  gaps: {
    title: "Give the shot somewhere to go",
    tag: "EXPERIMENT · BATTING",
    role: "batting",
    steps: 6,
    description:
      "Inspect the released ball and the field. Choose a stance and a shot direction. Move a fielder and predict how that changes your next decision.",
    question:
      "Did the direction of the shot matter? Did more power always lead to a better reward?",
  },
};
function stored(key, fallback) {
  try {
    return JSON.parse(localStorage.getItem(key)) ?? fallback;
  } catch {
    return fallback;
  }
}
function persist(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* Storage is optional. */
  }
}
let snapshots = stored("cricket-lab-snapshots-v1", []);
if (!Array.isArray(snapshots)) snapshots = [];
snapshots = snapshots
  .filter(
    (s) =>
      s && ["batting", "bowling"].includes(s.role) && Number.isFinite(s.runs),
  )
  .slice(-4);
let completedLessons = stored("cricket-lab-lessons-v1", []);
if (!Array.isArray(completedLessons)) completedLessons = [];
function element(tag, text, className) {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}
function signed(value) {
  return value > 0 ? `+${value}` : String(value).replace("-", "−");
}
function config() {
  return {
    role: $("role").value,
    seed: Number($("seed").value),
    overs: Number($("overs").value),
    style: $("style").value,
    pitch: $("pitch").value,
    target: $("target").value ? Number($("target").value) : null,
    adaptive_batting: $("adaptive").value === "true",
  };
}
function manualAction() {
  return state?.role === "batting"
    ? Number($("stance").value) * 15 +
        Number($("direction").value) * 3 +
        Number($("power").value)
    : Number($("line").value) * 3 + Number($("length").value);
}
function describeAction(action, role) {
  if (role === "batting") {
    const stance = Math.floor(action / 15),
      rest = action % 15;
    return `${["Leg", "Centre", "Off"][stance]} stance · ${directions[Math.floor(rest / 3)]} · ${["defend", "balanced", "attack"][rest % 3]}`;
  }
  return `${lines[Math.floor(action / 3)]} · ${lengths[action % 3]}`;
}
function controls() {
  const batting = state?.role === "batting",
    automated = $("policy").value !== "manual";
  $("batControls").hidden = !batting;
  $("bowlingControls").hidden = batting;
  for (const id of actionIds)
    $(id).disabled =
      busy ||
      automated ||
      (batting
        ? ["line", "length"].includes(id)
        : ["stance", "direction", "power"].includes(id));
  for (const id of configIds)
    $(id).disabled = busy || (id === "adaptive" && batting);
  $("policy").disabled = busy;
  $("reset").disabled = busy || runtimeUnavailable;
  $("startLesson").disabled = busy || runtimeUnavailable;
  $("edit").disabled = busy || !state || runtimeUnavailable;
  $("bowl").disabled =
    busy || !state || state.done || dirty || runtimeUnavailable;
  $("bowl").textContent = busy
    ? "Running delivery…"
    : batting
      ? "Play shot →"
      : "Bowl delivery →";
  $("actionCode").textContent = automated
    ? "Policy chooses action"
    : `ID ${manualAction()} · ${batting ? 45 : 9} choices`;
  $("actionHelp").textContent = automated
    ? "The policy chooses the action. You can inspect its actual choice after the delivery."
    : batting
      ? "Stance affects contact; direction and power change the outgoing shot. Directions are relative to the batter’s handedness."
      : "A yorker lands near the batter; a short delivery bounces farther away.";
  $("roleHelp").textContent =
    $("role").value === "batting"
      ? "Score runs while protecting your wickets."
      : "Take wickets while conceding fewer runs.";
  $("adaptiveHelp").textContent = batting
    ? "This switch affects scripted batters in bowling mode. Your shot replaces their automatic choices here."
    : "The scripted batter remembers deliveries. This setting does not train a neural agent.";
  $("policyHelp").textContent = {
    manual: "Make a prediction, then play a delivery.",
    random: "Actions are sampled at random. No policy is learned.",
    heuristic: batting
      ? "A fixed rule reacts to the delivery and looks for a field gap. It does not learn."
      : "A fixed rule selects a good-length delivery using the field. It does not learn.",
    ppo: "A saved checkpoint makes decisions. It is not being trained or updated in this view.",
  }[$("policy").value];
  $("settingsNotice").textContent = dirty
    ? "Settings changed. Start a new innings to apply them."
    : "10 wickets · legal balls count toward the innings limit.";
  $("settingsNotice").style.color = dirty ? "var(--orange)" : "";
  for (const id of ["play", "restart", "timeline"]) $(id).disabled = !delivery;
  $("export").disabled = !delivery;
  $("pinRun").disabled = !state || !state.history.length || busy;
  $("exportSession").disabled = !(
    journal.length ||
    archivedRuns.length ||
    $("reflection").value.trim() ||
    state?.history.length
  );
  $("play").textContent =
    playing && delivery && frame < delivery.trajectory.length - 1
      ? "Pause"
      : "Play";
  $("play").setAttribute("aria-label", $("play").textContent + " replay");
  $("edit").textContent = editing ? "Edit field: on" : "Edit field: off";
  $("edit").setAttribute("aria-pressed", String(editing));
  $("fieldEditor").hidden = !editing;
}
async function api(path, body) {
  if (window.cricketApi) return window.cricketApi(path, body);
  const response = await fetch(
    "/api/" + path,
    body === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
  );
  let data;
  try {
    data = await response.json();
  } catch {
    throw Error(
      "The simulator returned an unreadable response. Try connecting again.",
    );
  }
  if (!response.ok) {
    const error = Error(data.error || "The request could not be completed.");
    error.reconnect = response.status >= 500;
    throw error;
  }
  return data;
}
function update(s) {
  sceneRevision++;
  state = s;
  const batting = s.role === "batting",
    q = s.score;
  $("connection").textContent = "Environment connected";
  $("connection").className = "connection connected";
  $("reconnect").hidden = true;
  $("inningsLabel").textContent = batting
    ? "YOUR BATTING EXPERIMENT"
    : "YOUR BOWLING EXPERIMENT";
  $("score").textContent = `${q.runs} / ${q.wickets}`;
  $("oversLabel").textContent = `${q.overs} overs · ${q.extras} extras`;
  $("status").textContent = s.done
    ? q.complete
      ? "INNINGS COMPLETE"
      : "STEP LIMIT REACHED"
    : q.free_hit
      ? "FREE HIT"
      : "READY FOR A DECISION";
  $("chase").textContent = s.target
    ? `Target ${s.target} · ${Math.max(0, s.target - q.runs)} needed`
    : "";
  $("batter").textContent =
    `Batter ${q.striker + 1} · ${s.batter.handedness === 1 ? "Right" : "Left"} handed · ${s.batter.style} · anticipation ${Math.round(q.anticipation * 100)}%`;
  const expected = q.expected_action;
  $("anticipation").textContent =
    batting && s.pending_delivery !== null
      ? `Released ball: ${lines[Math.floor(s.pending_delivery / 3)]} / ${lengths[s.pending_delivery % 3]}`
      : !q.adaptive_batting
        ? "Static batting control"
        : expected === null
          ? "No bowling history yet"
          : `Preparing for ${lines[Math.floor(expected / 3)]} / ${lengths[expected % 3]}`;
  $("cueHelp").textContent = batting
    ? "The released line and length are visible before you choose. Contact noise and future outcomes remain hidden."
    : "The scripted batter uses its own past deliveries to prepare. It does not see your next action.";
  $("observationTitle").textContent = `${batting ? 66 : 57} observed values`;
  $("observationText").textContent = batting
    ? "Field, batter, pitch, score and history, plus the released delivery cue."
    : "Batter, field, pitch, score and bowling history. Future outcomes stay hidden.";
  const modelOption = $("policy").querySelector("[value=ppo]");
  modelOption.disabled = !s.model_available;
  modelOption.textContent = s.model_available
    ? `Trained policy · ${s.model_label ?? "checkpoint loaded"}`
    : "Trained policy · not loaded";
  $("modelAvailability").hidden = !s.model_notice;
  $("modelAvailability").textContent = s.model_notice ?? "";
  if ($("policy").value === "ppo" && !s.model_available)
    $("policy").value = "manual";
  $("history").replaceChildren();
  for (const h of [...s.history].reverse()) {
    const row = element("tr", "");
    for (const text of [
      h.overs,
      h.action,
      h.extra_type || h.event,
      signed(h.reward),
      `${h.runs}/${h.wickets}`,
    ])
      row.append(element("td", String(text)));
    $("history").append(row);
  }
  $("logCount").textContent = `${s.history.length} deliveries`;
  $("balls").replaceChildren();
  for (const h of s.history.slice(-12)) {
    const text = h.wicket
      ? "W"
      : h.extra_type === "wide"
        ? "Wd"
        : h.extra_type === "no_ball"
          ? "Nb"
          : String(h.delivery_runs);
    const ball = element(
      "span",
      text,
      "ball " + (h.wicket ? "wicket" : h.delivery_runs >= 4 ? "boundary" : ""),
    );
    ball.setAttribute("role", "listitem");
    ball.title = `${h.overs} overs: ${h.extra_type || h.event}, reward ${h.reward}`;
    $("balls").append(ball);
  }
  updateFielder();
  controls();
}
function showDelivery(s, prediction, policy) {
  lastResponse = s;
  delivery = s.delivery;
  frame = 0;
  playing = !reducedMotion;
  sideBounds = { minY: -4, maxY: 22, maxZ: 3 };
  for (const p of delivery.trajectory) {
    sideBounds.minY = Math.min(sideBounds.minY, p[1]);
    sideBounds.maxY = Math.max(sideBounds.maxY, p[1]);
    sideBounds.maxZ = Math.max(sideBounds.maxZ, p[2]);
  }
  $("timeline").max = delivery.trajectory.length - 1;
  const label = (
    delivery.dismissal ||
    delivery.extra_type ||
    delivery.event
  ).replaceAll("_", " ");
  $("event").textContent = label.charAt(0).toUpperCase() + label.slice(1);
  $("detail").textContent =
    `${delivery.runs} bat runs + ${delivery.extras} extras. ${delivery.legal ? "One legal ball completed." : "This ball does not count toward the over."} Batter readiness: ${Math.round(delivery.readiness * 100)}%.`;
  const weight = s.wicket_value ?? 8,
    totalRuns = delivery.runs + delivery.extras,
    wicket = Number(delivery.wicket);
  $("rewardValue").textContent = signed(s.reward);
  $("rewardValue").className = s.reward < 0 ? "negative" : "";
  $("rewardFormula").textContent =
    s.role === "batting"
      ? `${totalRuns} runs − ${wicket} wickets × ${weight} = ${signed(s.reward)}`
      : `${wicket} wickets × ${weight} − ${totalRuns} runs = ${signed(s.reward)}`;
  $("lastAction").textContent = `Action ${s.action}`;
  $("lastActionText").textContent = describeAction(s.action, s.role);
  $("nextState").textContent = s.done
    ? "Episode finished"
    : `${s.score.overs} overs · next decision`;
  const outcome = delivery.wicket ? "wicket" : totalRuns > 0 ? "runs" : "dot";
  $("predictionFeedback").textContent =
    prediction === "unsure"
      ? "What would you try next? One outcome does not establish a strategy."
      : prediction === outcome
        ? "Your prediction matched. Would it hold over many deliveries?"
        : "A different outcome than you predicted. Which uncertainty could explain it?";
  journal.push({
    attempt: s.score.attempts,
    policy,
    action: s.action,
    action_description: describeAction(s.action, s.role),
    prediction,
    outcome,
    reward: s.reward,
    score: s.score,
    fielders: s.fielders,
  });
  if (
    activeLesson &&
    policy === "manual" &&
    s.role === lessons[activeLesson].role
  ) {
    lessonSteps++;
    if (lessonSteps >= lessons[activeLesson].steps) {
      if (!completedLessons.includes(activeLesson))
        completedLessons.push(activeLesson);
      persist("cricket-lab-lessons-v1", completedLessons);
    }
    lessonProgress();
  }
  controls();
}
async function guard(fn) {
  if (busy) return;
  busy = true;
  $("error").textContent = "";
  controls();
  try {
    await fn();
  } catch (error) {
    $("error").textContent =
      error.message ||
      "Could not reach the simulator. Check that the server is running, then try again.";
    if (
      error.runtimeFailure ||
      error.reconnect ||
      error instanceof TypeError ||
      window.cricketApi?.failed?.()
    ) {
      runtimeUnavailable = true;
      $("reconnect").hidden = false;
      $("connection").textContent = "Environment unavailable";
      $("connection").className = "connection";
    }
  } finally {
    busy = false;
    controls();
  }
}
async function resetInnings() {
  const form = config();
  if ($("seed").value.trim() === "")
    throw Error(
      "Enter a random seed, such as 42, so you can repeat this experiment.",
    );
  if (!Number.isInteger(form.seed) || form.seed < 0 || form.seed >= 2 ** 32)
    throw Error("Choose a whole-number seed between 0 and 4294967295.");
  if (
    form.target !== null &&
    (!Number.isInteger(form.target) || form.target < 1 || form.target > 500)
  )
    throw Error(
      "Choose a whole-number target from 1 to 500, or leave it empty.",
    );
  const s = await api("reset", form);
  if (journal.length || $("reflection").value.trim())
    archivedRuns.push({
      settings: runConfig,
      deliveries: journal,
      reflection: $("reflection").value,
    });
  journal = [];
  runConfig = form;
  dirty = false;
  delivery = null;
  frame = 0;
  lastResponse = null;
  editing = false;
  $("reflection").value = "";
  $("timeline").value = 0;
  $("event").textContent = "A fresh experiment";
  $("detail").textContent =
    "Choose a decision, predict its outcome, then inspect the transition.";
  $("rewardValue").textContent = "—";
  $("rewardValue").className = "";
  $("rewardFormula").textContent =
    s.role === "batting"
      ? "Score a run: +1. Lose a wicket: −8."
      : "Take a wicket: +8. Concede a run: −1.";
  $("lastAction").textContent = "Your choice";
  $("lastActionText").textContent = "No action taken yet.";
  $("nextState").textContent = "Ready to begin";
  $("predictionFeedback").textContent =
    "Make a prediction, then see the result.";
  $("telemetry").textContent =
    "Your first delivery will appear here. Choose an action below to get started.";
  update(s);
}
function project(p) {
  const view = $("view").value;
  if (view === "top") return [500 + p[0] * 4.2, 325 - p[1] * 4.2];
  if (view === "side") {
    const { minY, maxY, maxZ } = sideBounds;
    return [
      75 + ((p[1] - minY) / (maxY - minY)) * 850,
      560 - (p[2] / maxZ) * 430,
    ];
  }
  return [500 + p[0] * 5.6, 338 - p[1] * 3.3 - p[2] * 22];
}
function unproject(x, y) {
  return [(x - 500) / 4.2, (325 - y) / 4.2];
}
function path(points, fill, stroke, width = 1) {
  ctx.beginPath();
  points.forEach((p, i) =>
    i ? ctx.lineTo(...project(p)) : ctx.moveTo(...project(p)),
  );
  if (fill) {
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
  }
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    ctx.stroke();
  }
}
function circle(point, radius, color) {
  const [x, y] = project(point);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
}
function player(point, color, number, selected) {
  const [x, y] = project(point);
  ctx.fillStyle = "#031b183d";
  ctx.beginPath();
  ctx.ellipse(x, y + 6, 11, 5, 0, 0, Math.PI * 2);
  ctx.fill();
  if (selected) {
    ctx.strokeStyle = "#e3edb5";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, 16, 0, Math.PI * 2);
    ctx.stroke();
  }
  circle(point, 8, color);
  ctx.fillStyle = "#e4f2dc";
  ctx.font = "11px system-ui";
  if (number) ctx.fillText(number, x + 12, y - 9);
}
function draw(now) {
  requestAnimationFrame(draw);
  if (!state) return;
  const elapsed = last ? Math.min((now - last) / 1000, 0.1) : 0;
  last = now;
  if (playing && delivery && !editing) {
    frame = Math.min(
      delivery.trajectory.length - 1,
      frame + elapsed * 50 * Number($("speed").value),
    );
    if (frame >= delivery.trajectory.length - 1) {
      playing = false;
      controls();
    }
  }
  const index = Math.floor(frame),
    view = $("view").value;
  const key = `${sceneRevision}:${index}:${view}:${editing}:${$("fielderSelect").value}`;
  if (key === paintedScene) return;
  paintedScene = key;
  $("timeline").value = index;
  ctx.fillStyle = "#133e2f";
  ctx.fillRect(0, 0, 1000, 650);
  for (let stripe = 0; stripe < 13; stripe++) {
    ctx.fillStyle = stripe % 2 ? "#ffffff02" : "#00000002";
    ctx.fillRect(0, stripe * 50, 1000, 50);
  }
  const glow = ctx.createRadialGradient(500, 330, 30, 500, 330, 500);
  glow.addColorStop(0, "#74924b18");
  glow.addColorStop(1, "#052e2510");
  ctx.fillStyle = glow;
  ctx.fillRect(0, 0, 1000, 650);
  for (const radius of [30, state.pitch.boundary]) {
    const points = [];
    for (let i = 0; i <= 120; i++) {
      const a = (i / 120) * Math.PI * 2;
      points.push([Math.sin(a) * radius, Math.cos(a) * radius, 0]);
    }
    path(
      points,
      null,
      radius === 30 ? "#c3d5ae37" : "#e4edd486",
      radius === 30 ? 1 : 2,
    );
  }
  if (view !== "side") {
    path(
      [
        [-1.65, -1.22, 0],
        [1.65, -1.22, 0],
        [1.65, 20.12, 0],
        [-1.65, 20.12, 0],
      ],
      "#bcaa7b",
    );
    path(
      [
        [-1.2, -1.22, 0],
        [1.2, -1.22, 0],
        [1.2, 20.12, 0],
        [-1.2, 20.12, 0],
      ],
      "#c9b88d",
    );
    for (const y of [0, 1.22, 18.9, 20.12])
      path(
        [
          [-1.7, y, 0],
          [1.7, y, 0],
        ],
        null,
        "#f2ead4bb",
        1.5,
      );
  } else {
    path(
      [
        [0, -4, 0],
        [0, 70, 0],
      ],
      null,
      "#b7cc9e6e",
    );
  }
  for (const x of [-0.12, 0, 0.12])
    path(
      [
        [x, 0, 0],
        [x, 0, 0.711],
      ],
      null,
      "#eee5c9",
      2,
    );
  const fielders =
    delivery && !editing
      ? delivery.fielder_trajectory[index] || state.fielders
      : state.fielders;
  if (view !== "side")
    fielders.forEach((f, i) =>
      player(
        [...f, 0],
        "#8ccbf2",
        String(i + 1),
        editing && i === Number($("fielderSelect").value),
      ),
    );
  player([0.7, 0, 0], "#efe2b1");
  player([0, 18, 0], "#d8e9a1");
  if (delivery && !editing) {
    for (let segment = 0; segment < 2; segment++) {
      const cut = delivery.contact_frame ?? delivery.trajectory.length,
        start = segment === 0 ? 0 : cut,
        end = segment === 0 ? Math.min(index, cut) : index;
      if (end < start) continue;
      path(
        delivery.trajectory.slice(start, end + 1),
        null,
        segment === 0 ? "#d3e6c4bd" : "#f2d885d9",
        2.2,
      );
    }
    const ball = delivery.trajectory[index];
    circle([ball[0], ball[1], 0], 5, "#001c1c6e");
    circle(ball, 6, "#ffe7a1");
    const [x, y] = project(ball);
    ctx.strokeStyle = "#bb6348";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(x, y, 4, -1, 1);
    ctx.stroke();
    $("telemetry").textContent =
      `${(index * 0.02).toFixed(2)} s · height ${ball[2].toFixed(2)} m · ${index + 1}/${delivery.trajectory.length} frames${delivery.contact_frame !== null && index >= delivery.contact_frame ? " · after contact" : ""}`;
  }
  $("sceneTitle").textContent = editing
    ? "Field editor · drag a player"
    : $("view").selectedOptions[0].textContent;
}
function updateFielder() {
  if (!state) return;
  const [x, y] = state.fielders[Number($("fielderSelect").value)];
  $("fielderPosition").textContent =
    `x ${x.toFixed(1)} m · y ${y.toFixed(1)} m`;
}
function limitField(point) {
  const r = Math.hypot(...point),
    max = state.pitch.boundary - 1;
  return r > max ? point.map((v) => (v * max) / r) : point;
}
async function saveField() {
  try {
    update(await api("field", { fielders: state.fielders }));
  } catch (error) {
    try {
      update(await api("state"));
    } catch {
      /* Preserve original failure. */
    }
    throw error;
  }
}
function pointer(event) {
  const r = canvas.getBoundingClientRect();
  return [
    ((event.clientX - r.left) * canvas.width) / r.width,
    ((event.clientY - r.top) * canvas.height) / r.height,
  ];
}
canvas.onpointerdown = (event) => {
  if (!editing || busy) return;
  const [x, y] = pointer(event);
  let distance = 24;
  dragging = null;
  state.fielders.forEach((f, i) => {
    const p = project([...f, 0]),
      d = Math.hypot(p[0] - x, p[1] - y);
    if (d < distance) {
      dragging = i;
      distance = d;
    }
  });
  if (dragging !== null) {
    $("fielderSelect").value = dragging;
    updateFielder();
    canvas.setPointerCapture(event.pointerId);
  }
};
canvas.onpointermove = (event) => {
  if (dragging === null) return;
  state.fielders[dragging] = limitField(unproject(...pointer(event)));
  sceneRevision++;
  updateFielder();
};
async function finishDrag() {
  if (dragging === null) return;
  dragging = null;
  await guard(saveField);
}
canvas.onpointerup = finishDrag;
canvas.onpointercancel = finishDrag;
canvas.onkeydown = (event) => {
  if (!editing || busy) return;
  const shifts = {
    ArrowLeft: [-3, 0],
    ArrowRight: [3, 0],
    ArrowUp: [0, 3],
    ArrowDown: [0, -3],
  };
  if (shifts[event.key]) {
    event.preventDefault();
    nudge(...shifts[event.key]);
  }
};
async function nudge(dx, dy) {
  if (!editing || busy) return;
  await guard(async () => {
    const i = Number($("fielderSelect").value);
    state.fielders[i] = limitField([
      state.fielders[i][0] + dx,
      state.fielders[i][1] + dy,
    ]);
    await saveField();
  });
}
function lessonProgress() {
  const id = $("lessonSelect").value,
    lesson = lessons[id],
    steps = activeLesson === id ? Math.min(lessonSteps, lesson.steps) : 0;
  $("lessonProgressBar").style.width = `${(steps / lesson.steps) * 100}%`;
  $("lessonProgress").textContent =
    activeLesson === id
      ? steps >= lesson.steps
        ? "Experiment complete. Write a reflection, then try one changed condition."
        : `${steps} of ${lesson.steps} manual decisions · predict, play, inspect.`
      : completedLessons.includes(id)
        ? "Previously completed. Start again to explore a different outcome."
        : "Starts a fresh one-over innings with seed 42.";
}
function selectLesson() {
  const lesson = lessons[$("lessonSelect").value];
  $("lessonTag").textContent = lesson.tag;
  $("lessonTitle").textContent = lesson.title;
  $("lessonDescription").textContent = lesson.description;
  $("lessonQuestion").textContent = lesson.question;
  lessonProgress();
}
function renderSnapshots() {
  $("savedRuns").replaceChildren();
  if (!snapshots.length)
    $("savedRuns").append(
      element(
        "div",
        "No snapshots yet. Play a few deliveries, then save a run to compare.",
        "empty-card",
      ),
    );
  snapshots.forEach((s, i) => {
    const card = element("article", "", "snapshot");
    card.append(
      element(
        "strong",
        `${String.fromCharCode(65 + i)} · ${s.role === "batting" ? "Batting" : "Bowling"} · ${s.policy}`,
      ),
    );
    card.append(element("p", `${s.runs}/${s.wickets}`, "snapshot-score"));
    card.append(
      element(
        "p",
        `${s.overs} overs · ${s.attempts} attempts · total reward ${signed(s.reward)}`,
      ),
    );
    card.append(
      element(
        "p",
        `Seed ${s.config?.seed ?? "?"} · ${s.config?.pitch ?? "?"} pitch · ${s.complete ? "Complete innings" : "Partial innings"}`,
      ),
    );
    $("savedRuns").append(card);
  });
  $("clearRuns").disabled = !snapshots.length;
}
function download(value, name) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("reset").onclick = () =>
  guard(async () => {
    await resetInnings();
    activeLesson = null;
    lessonSteps = 0;
    lessonProgress();
  });
$("bowl").onclick = () =>
  guard(async () => {
    editing = false;
    const prediction = $("prediction").value,
      policy = $("policy").value;
    const s = await api("bowl", { policy, action: manualAction() });
    update(s);
    showDelivery(s, prediction, policy);
  });
$("play").onclick = () => {
  if (!delivery) return;
  if (frame >= delivery.trajectory.length - 1) frame = 0;
  playing = !playing;
  controls();
};
$("restart").onclick = () => {
  frame = 0;
  playing = true;
  controls();
};
$("timeline").oninput = () => {
  frame = Number($("timeline").value);
  playing = false;
  controls();
};
$("edit").onclick = () => {
  editing = !editing;
  if (editing) $("view").value = "top";
  controls();
  updateFielder();
};
$("view").onchange = () => {
  if (editing && $("view").value !== "top") {
    editing = false;
    controls();
  }
};
$("fielderSelect").onchange = updateFielder;
document
  .querySelectorAll("[data-dx]")
  .forEach(
    (button) =>
      (button.onclick = () =>
        nudge(Number(button.dataset.dx), Number(button.dataset.dy))),
  );
for (const id of configIds)
  $(id).onchange = () => {
    dirty = true;
    controls();
  };
for (const id of [...actionIds, "policy"]) $(id).onchange = controls;
$("lessonSelect").onchange = selectLesson;
$("startLesson").onclick = () =>
  guard(async () => {
    const id = $("lessonSelect").value,
      lesson = lessons[id];
    $("role").value = lesson.role;
    $("seed").value = "42";
    $("overs").value = "1";
    $("style").value = "balanced";
    $("pitch").value = "standard";
    $("target").value = "";
    $("adaptive").value = "true";
    $("policy").value = "manual";
    $("line").value = "1";
    $("length").value = id === "patterns" ? "0" : "1";
    await resetInnings();
    activeLesson = id;
    lessonSteps = 0;
    lessonProgress();
    if (id === "gaps") $("view").value = "top";
    $("playground").scrollIntoView({
      behavior: reducedMotion ? "instant" : "smooth",
    });
  });
$("pinRun").onclick = () => {
  if (!state?.history.length) return;
  const q = state.score;
  const policies = [
    ...new Set(state.history.map((h) => h.policy ?? "unrecorded")),
  ];
  snapshots.push({
    role: state.role,
    policy: policies.length === 1 ? policies[0] : "mixed policies",
    config: runConfig,
    runs: q.runs,
    wickets: q.wickets,
    overs: q.overs,
    attempts: q.attempts,
    complete: q.complete,
    reward: state.history.reduce((a, h) => a + h.reward, 0),
  });
  snapshots = snapshots.slice(-4);
  persist("cricket-lab-snapshots-v1", snapshots);
  renderSnapshots();
};
$("clearRuns").onclick = () => {
  snapshots = [];
  persist("cricket-lab-snapshots-v1", snapshots);
  renderSnapshots();
};
$("export").onclick = () => {
  if (delivery) download({ state, delivery }, "cricket-replay.json");
};
$("exportSession").onclick = () =>
  download(
    {
      format: "cricket-learning-journal",
      version: 1,
      created_at: new Date().toISOString(),
      settings: runConfig,
      lesson: activeLesson,
      deliveries: journal,
      reflection: $("reflection").value,
      earlier_runs: archivedRuns,
      snapshots,
      note: "Exploratory delivery-level simulator. Same seeds do not align noise between different action sequences. Playing does not update policy weights.",
    },
    "cricket-learning-journal.json",
  );
async function connect(restart = false) {
  if (restart && window.cricketApi?.restart) window.cricketApi.restart();
  await guard(async () => {
    const s = await api("state");
    const c = s.settings ?? {
      role: s.role,
      seed: s.seed,
      overs: (s.ball_limit ?? 24) / 6,
      style: s.batter.style,
      pitch: "standard",
      target: s.target,
      adaptive_batting: s.score.adaptive_batting,
    };
    $("role").value = c.role;
    $("seed").value = c.seed;
    $("overs").value = String(c.overs);
    $("style").value = c.style;
    $("pitch").value = c.pitch;
    $("target").value = c.target ?? "";
    $("adaptive").value = String(c.adaptive_batting);
    if (restart) {
      if (journal.length || $("reflection").value.trim())
        archivedRuns.push({
          settings: runConfig,
          deliveries: journal,
          reflection: $("reflection").value,
        });
      journal = [];
      delivery = null;
      lastResponse = null;
      frame = 0;
      activeLesson = null;
      lessonSteps = 0;
      lessonProgress();
      $("reflection").value = "";
      $("rewardValue").textContent = "—";
      $("rewardValue").className = "";
      $("lastAction").textContent = "Your choice";
      $("lastActionText").textContent = "No action in the new environment yet.";
      $("nextState").textContent = "Ready to begin";
      $("event").textContent = "Environment reconnected";
      $("telemetry").textContent = "Choose an action to start a new delivery.";
      $("detail").textContent =
        "Environment reconnected. Earlier notes are retained in the journal archive.";
    }
    runtimeUnavailable = false;
    runConfig = c;
    dirty = false;
    update(s);
    if (s.history.length) {
      $("event").textContent = "Existing innings restored";
      $("detail").textContent =
        "The server has earlier deliveries in this innings. Start fresh for a guided lesson.";
    }
  });
  if (!state) {
    $("connection").textContent = "Environment offline";
    $("connection").className = "connection";
    $("reconnect").hidden = false;
  }
}
$("reconnect").onclick = () => connect(true);
$("reflection").oninput = controls;
selectLesson();
renderSnapshots();
controls();
connect();
requestAnimationFrame(draw);
