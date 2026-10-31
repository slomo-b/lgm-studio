/* LGM Studio -- Oberflaeche.
 *
 * Haelt keinen eigenen Zustand ueber die Pipeline: Rust startet die Prozesse
 * und schickt jede Ausgabezeile als Event. Die Oberflaeche zeigt nur an, was
 * passiert -- so kann sie nicht "luegen", wenn ein Lauf scheitert.
 */

import { SplatViewer } from "./splat-viewer.js";

const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;

const $ = (id) => document.getElementById(id);

const state = {
  image: null,
  ply: null,
  workspace: null,
  steps: 16,
  running: false,
  downloading: false,
  env: null,
  models: null,
  dlDone: 0,
};

/* ---------------------------------------------------------------- Log */

function log(text, cls = "") {
  const el = $("log");
  const line = document.createElement("div");
  if (cls) line.className = cls;
  line.textContent = text;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

function logLine(raw) {
  const t = raw.replace(/\s+$/, "");
  if (!t) return;
  const cls = /fertig|\[out\]|\[done\]|PASS/i.test(t)
    ? "ok"
    : /error|traceback|fehler|FAIL/i.test(t)
    ? "err"
    : /\[info\]|\[warn\]/i.test(t)
    ? "dim"
    : "";
  log(t, cls);
}

/* ---------------------------------------------------------------- Umgebung */

async function refreshEnv() {
  try {
    const env = await invoke("env_info");
    state.env = env;
    const dot = $("env-dot");
    const txt = $("env-text");
    if (env.ok) {
      dot.className = "dot ok";
      txt.textContent = "bereit";
      $("btn-env").title = env.root;
    } else {
      dot.className = "dot err";
      txt.textContent = env.problems.length + " Problem(e)";
      $("btn-env").title = env.problems.join("\n") + "\n\n" + env.root;
      env.problems.forEach((p) => log("Problem: " + p, "err"));
    }
    $("status-right").textContent = env.root;
    updateButtons();
    return env;
  } catch (e) {
    $("env-dot").className = "dot err";
    $("env-text").textContent = "Fehler";
    log("Umgebungsprüfung fehlgeschlagen: " + e, "err");
    return null;
  }
}

/* ---------------------------------------------------------------- Eingabe */

async function pickImage() {
  const p = await invoke("pick_image");
  if (p) setImage(p);
}

function setImage(path) {
  state.image = path;
  const img = $("preview");
  // Nativer Pfad -> ueber den read_file-Befehl laden, damit kein HTTP noetig ist
  invoke("read_file", { path })
    .then((bytes) => {
      const blob = new Blob([new Uint8Array(bytes)]);
      img.src = URL.createObjectURL(blob);
      img.hidden = false;
      $("dz-hint").hidden = true;
      updateButtons();
    })
    .catch((e) => log("Bild konnte nicht geladen werden: " + e, "err"));
  $("hint").textContent = path.split(/[\\/]/).pop();
}

/* ---------------------------------------------------------------- Generieren */

let timer = null;

function startProgress(expectedSeconds) {
  const t0 = performance.now();
  $("stage-loading").hidden = false;
  $("bar-fill").style.width = "2%";
  timer = setInterval(() => {
    const elapsed = (performance.now() - t0) / 1000;
    // Nie ueber 96 %: fertig ist erst, wenn der Prozess fertig ist.
    const frac = Math.min(0.96, elapsed / expectedSeconds);
    $("bar-fill").style.width = (frac * 100).toFixed(1) + "%";
    $("loading-eta").textContent =
      elapsed.toFixed(0) + " s" + (frac < 0.96 ? " · geschätzt " + expectedSeconds + " s" : "");
  }, 250);
}

function stopProgress() {
  if (timer) clearInterval(timer);
  timer = null;
  $("stage-loading").hidden = true;
  $("bar-fill").style.width = "0%";
}

async function generate() {
  if (!state.image || state.running) return;
  state.running = true;
  updateButtons();
  $("log").textContent = "";
  $("status").textContent = "Generiere…";

  const expected = Math.round(state.steps * 4.3 + 16);
  startProgress(expected);

  try {
    const res = await invoke("generate", { image: state.image, steps: state.steps });
    (res.log || []).slice(-40).forEach(logLine);
    state.workspace = res.workspace;

    if (res.ok && res.ply) {
      state.ply = res.ply;
      log(`Fertig in ${res.seconds.toFixed(1)} s`, "ok");
      $("status").textContent = "Fertig in " + res.seconds.toFixed(1) + " s";
      await showSplat(res.ply);
    } else {
      log("Kein .ply erzeugt — siehe Verlauf", "err");
      $("status").textContent = "Fehlgeschlagen";
    }
  } catch (e) {
    log("Fehler: " + e, "err");
    $("status").textContent = "Fehler";
  } finally {
    stopProgress();
    state.running = false;
    updateButtons();
  }
}

/* ---------------------------------------------------------------- Viewer */

let viewer = null;

async function showSplat(path) {
  const bytes = await invoke("read_file", { path });
  const buf = new Uint8Array(bytes).buffer;

  if (!viewer) {
    viewer = new SplatViewer($("canvas"), { background: [0.043, 0.055, 0.071, 1] });
    viewer.start();
  }
  viewer._parse(buf);
  viewer.reset();
  // LGM-Splats sind auf ~1 Einheit normiert; die Standard-Distanz des Viewers
  // rahmt sie zu knapp. Faktor lieber zu weit als abgeschnitten.
  viewer.distance *= 1.7;
  viewer.dirty = true;

  $("stage-empty").hidden = true;
  $("stage-toolbar").hidden = false;
  const n = (buf.byteLength / 62 / 4) | 0; // grobe Schaetzung aus der Dateigroesse
  $("stat").textContent = path.split(/[\\/]/).pop();
}

/* ---------------------------------------------------------------- Export */

async function exportUsd() {
  if (!state.ply || state.running) return;
  const cls = $("obj-class").value;
  $("btn-export").disabled = true;
  $("status").textContent = "Exportiere SimReady…";
  log("SimReady-Export: " + cls, "dim");
  try {
    const res = await invoke("export_simready", {
      ply: state.ply,
      objClass: cls,
      targetSize: null,
    });
    (res.log || []).forEach(logLine);
    if (res.ok) {
      log("USD: " + res.usd, "ok");
      $("status").textContent = "USD geschrieben";
      $("btn-folder").disabled = false;
    } else {
      log("Export ohne Ergebnis", "err");
      $("status").textContent = "Export fehlgeschlagen";
    }
  } catch (e) {
    log("Export-Fehler: " + e, "err");
    $("status").textContent = "Export-Fehler";
  } finally {
    updateButtons();
  }
}

/* ---------------------------------------------------------------- Modelle */

function fmtBytes(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + " GB";
  if (n >= 1e6) return (n / 1e6).toFixed(0) + " MB";
  return (n / 1e3).toFixed(0) + " KB";
}

function setSetupProgress(done, total) {
  const frac = total > 0 ? Math.min(1, done / total) : 0;
  $("setup-fill").style.width = (frac * 100).toFixed(2) + "%";
}

async function checkModels() {
  try {
    const st = await invoke("models_status");
    state.models = st;
    if (st.complete) {
      $("setup").hidden = true;
      updateButtons();
      return;
    }
    $("setup").hidden = false;
    $("setup-size").textContent = fmtBytes(st.total);
    $("setup-hint").textContent = "Ziel: " + st.dir;
    $("setup-file").textContent =
      st.present > 0 ? fmtBytes(st.present) + " bereits vorhanden" : "bereit";
    setSetupProgress(st.present, st.total);
    updateButtons();
    // Testpfad: Download ohne Klick starten
    if (state.env && state.env.autodownload) setTimeout(startDownload, 400);
  } catch (e) {
    log("Modellprüfung fehlgeschlagen: " + e, "err");
  }
}

let dlTimer = null;

async function startDownload() {
  if (state.downloading) return;
  state.downloading = true;
  $("setup-start").disabled = true;
  $("setup-cancel").hidden = false;

  const t0 = performance.now();
  const base = state.models ? state.models.present : 0;
  let lastDone = base;
  let speed = 0;

  if (dlTimer) clearInterval(dlTimer);
  dlTimer = setInterval(() => {
    const done = state.dlDone ?? base;
    const dt = (performance.now() - t0) / 1000;
    if (dt > 1) speed = (done - base) / dt;
    const total = state.models ? state.models.total : 0;
    const rest = total - done;
    $("setup-eta").textContent =
      speed > 1e6
        ? `${fmtBytes(speed)}/s · noch ${Math.max(1, Math.round(rest / speed / 60))} min`
        : "";
    lastDone = done;
  }, 500);

  try {
    await invoke("download_models");
    log("Modelle vollständig.", "ok");
  } catch (e) {
    log("Download: " + e, "err");
  } finally {
    clearInterval(dlTimer);
    dlTimer = null;
    state.downloading = false;
    $("setup-start").disabled = false;
    $("setup-cancel").hidden = true;
    await checkModels();
  }
}

/* ---------------------------------------------------------------- Zustand */

function updateButtons() {
  const envOk = state.env ? state.env.ok : false;
  const modelsOk = state.models ? state.models.complete : false;
  const ready = envOk && modelsOk;
  $("btn-generate").disabled = !state.image || state.running || !ready;
  $("btn-export").disabled = !state.ply || state.running || !envOk;
  $("btn-folder").disabled = !state.workspace;
}

/* ---------------------------------------------------------------- Start */

async function init() {
  await refreshEnv();

  $("dropzone").addEventListener("click", pickImage);
  $("dropzone").addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pickImage(); }
  });
  $("btn-env").addEventListener("click", refreshEnv);
  $("btn-generate").addEventListener("click", generate);
  $("btn-export").addEventListener("click", exportUsd);
  $("setup-start").addEventListener("click", startDownload);
  $("setup-cancel").addEventListener("click", () => invoke("cancel_download"));
  $("btn-folder").addEventListener("click", () => state.workspace && invoke("reveal", { path: state.workspace }));

  // Konsole ein-/ausklappen
  const drawer = $("drawer");
  $("drawer-toggle").addEventListener("click", () => {
    const closed = drawer.classList.toggle("closed");
    $("drawer-toggle").setAttribute("aria-expanded", String(!closed));
  });

  $("btn-reset").addEventListener("click", () => viewer && viewer.reset());
  $("btn-rotate").addEventListener("click", (e) => {
    if (!viewer) return;
    const on = !viewer.autoRotate;
    viewer.setAutoRotate(on);
    e.currentTarget.style.color = on ? "var(--accent)" : "";
  });

  document.querySelectorAll("#steps button").forEach((b) => {
    b.addEventListener("click", () => {
      document.querySelectorAll("#steps button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      state.steps = parseInt(b.dataset.steps, 10);
      $("steps-note").textContent = b.querySelector("span").textContent;
    });
  });

  // Drag & Drop: HTML5-Wege reichen im WebView, Tauri blockiert sie nicht,
  // solange dragDropEnabled nicht Dateipfade liefert -- deshalb beides.
  const dz = $("dropzone");
  ["dragover", "dragenter"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("over"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, () => dz.classList.remove("over"))
  );
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) {
      const p = f.path;
      if (p) setImage(p);
      else log("Dateipfad nicht verfügbar — bitte über den Dialog wählen", "err");
    }
  });

  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); generate(); }
  });

  await listen("pipeline-line", (ev) => logLine(ev.payload.line));

  // Download-Fortschritt
  await listen("dl-progress", (ev) => {
    const p = ev.payload;
    state.dlDone = p.overall_done;
    setSetupProgress(p.overall_done, p.overall_total);
    $("setup-file").textContent = p.path + " · " + fmtBytes(p.file_done);
    if (p.error) log("Download-Fehler: " + p.error, "err");
  });

  await checkModels();
  await listen("setup-line", (ev) => logLine(`[${ev.payload.phase}] ${ev.payload.line}`));
  await listen("runtime-progress", (ev) => {
    const p = ev.payload;
    if (p.total) setSetupProgress(p.done, p.total);
    $("setup-file").textContent = "CPython: " + fmtBytes(p.done) + " / " + fmtBytes(p.total);
  });
  log("Bereit.", "dim");

  // Testpfad: Laufzeit ohne Klick einrichten
  if (state.env && state.env.setupruntime) {
    $("setup").hidden = false;
    $("setup-file").textContent = "Laufzeit wird eingerichtet …";
    setTimeout(() => invoke("install_runtime").catch((e) => log("Laufzeit: " + e, "err")), 400);
  }

  // Startbild (Kommandozeile / "Oeffnen mit")
  const start = await invoke("initial_image");
  if (start) {
    setImage(start);
    log("Startbild: " + start, "dim");
    // Automatischer Lauf nur fuer Tests
    if (state.env && state.env.autorun && state.env.ok) {
      setTimeout(generate, 600);
    }
  }
}

window.addEventListener("DOMContentLoaded", init);
