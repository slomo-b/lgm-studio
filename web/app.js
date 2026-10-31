import { SplatViewer } from '/static/splat-viewer.js';

const $ = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ i18n */

const I18N = {
  de: {
    'brand.sub': 'Bild → 3D-Gaussian',
    'engine.title': 'Render-Engine (aceone)',
    'engine.ready': 'aceone bereit',
    'engine.missing': 'aceone fehlt',
    'source.title': 'Quelle',
    'source.reset': 'zurücksetzen',
    'drop.title': 'Bild hierher ziehen',
    'drop.hint': 'oder klicken · PNG, JPG, WEBP · Strg+V',
    'variants.label': 'Varianten',
    'cand.title': 'Kandidat {i} von {n}',
    'cand.keep': 'Diesen behalten',
    'cand.discard': 'Verwerfen',
    'cand.kept': 'Kandidat übernommen',
    'job.candidate': 'Kandidat {i}/{n} …',
    'quality.label': 'Qualität',
    'quality.fast': 'Schnell',
    'quality.std': 'Standard',
    'quality.fine': 'Fein',
    'samples.label': 'Beispiele',
    'samples.bird': 'Vogel',
    'samples.cat': 'Katze',
    'samples.figure': 'Figur',
    'btn.generate': 'Generieren',
    'btn.running': 'Läuft…',
    'btn.download': 'Herunterladen',
    'btn.delete': 'Löschen',
    'footnote': 'Ein Objekt, einfacher Hintergrund. ~1 Min (erster Lauf länger).',
    'history.title': 'Verlauf',
    'history.empty': 'Noch nichts generiert.',
    'empty.title': 'Noch kein Modell geladen',
    'empty.sub': 'Links ein Bild wählen oder einen Eintrag aus dem Verlauf öffnen.',
    'progress.title': 'Wird generiert…',
    'step.bg': 'Hintergrund entfernen',
    'step.views': 'Ansichten erzeugen',
    'step.gauss': '3D-Gaussians rechnen',
    'step.write': 'Modell schreiben',
    'tool.autorotate': 'Auto-Rotation',
    'tool.reset': 'Ansicht zurücksetzen',
    'tool.bg': 'Hintergrund wechseln',
    'tool.full': 'Vollbild',
    'badge.hint': 'ziehen: drehen · scrollen: zoomen',
    'badge.loading': 'lädt…',
    'badge.engine': 'gerendert mit aceone',
    'badge.still': 'Standbild',
    'splats': 'Splats',
    'job.starting': 'Starte…',
    'job.loading': 'Lade Modelle…',
    'job.bg': 'Entferne Hintergrund…',
    'job.loaded': 'Modelle geladen, starte…',
    'job.views': 'Erzeuge 4 Ansichten ({n}/30)',
    'job.gauss': 'Rechne 3D-Gaussians…',
    'job.write': 'Schreibe Modell…',
    'job.preview': 'Rendere Vorschau…',
    'job.done': 'Fertig',
    'job.subdone': 'in {s} s',
    'toast.done': 'Modell fertig',
    'toast.deleted': 'Modell gelöscht',
    'toast.deleteFail': 'Löschen fehlgeschlagen',
    'toast.notImage': 'Das ist kein Bild.',
    'toast.tooBig': 'Datei zu groß (max. 25 MB).',
    'toast.connection': 'Verbindung zum Server verloren. webui.cmd neu starten.',
    'toast.uploadFail': 'Upload fehlgeschlagen',
    'toast.error': 'Fehler',
    'toast.disconnected': 'Getrennt',
    'toast.aborted': 'Abgebrochen',
    'toast.unknown': 'Unbekannter Fehler',
    'toast.webgl': 'WebGL2 nicht verfügbar — es wird ein Standbild gezeigt.',
    'toast.viewer': 'Viewer',
    'toast.model': 'Modell',
    'toast.png': 'PNG',
    'confirm.delete': '"{name}" löschen?',
    'ago.now': 'gerade eben',
    'ago.min': 'vor {n} min',
    'ago.hour': 'vor {n} h',
    'err.busy': 'Es läuft bereits ein Auftrag.',
    'err.format': 'Format {ext} nicht unterstützt',
    'err.empty': 'Leere Datei',
    'err.too_big': 'Datei zu groß (max. 25 MB)',
    'err.upload': 'Upload unvollständig',
    'err.lgm': 'LGM abgebrochen',
    'err.no_ply': 'Keine .ply erzeugt',
    'err.no_engine': 'aceone splat-render.exe nicht gefunden',
    'err.no_model': 'Modell nicht gefunden',
    'err.render': 'Render fehlgeschlagen',
    'err.not_found': 'Nicht gefunden',
    'err.unknown': 'Unbekannt',
    'err.delete_running': 'Modell wird gerade erzeugt',
  },
  en: {
    'brand.sub': 'Image → 3D Gaussian',
    'engine.title': 'Render engine (aceone)',
    'engine.ready': 'aceone ready',
    'engine.missing': 'aceone missing',
    'source.title': 'Source',
    'source.reset': 'reset',
    'drop.title': 'Drag an image here',
    'drop.hint': 'or click · PNG, JPG, WEBP · Ctrl+V',
    'variants.label': 'Variants',
    'cand.title': 'Candidate {i} of {n}',
    'cand.keep': 'Keep this one',
    'cand.discard': 'Discard',
    'cand.kept': 'Candidate saved',
    'job.candidate': 'Candidate {i}/{n} …',
    'quality.label': 'Quality',
    'quality.fast': 'Fast',
    'quality.std': 'Standard',
    'quality.fine': 'High',
    'samples.label': 'Samples',
    'samples.bird': 'Bird',
    'samples.cat': 'Cat',
    'samples.figure': 'Figure',
    'btn.generate': 'Generate',
    'btn.running': 'Running…',
    'btn.download': 'Download',
    'btn.delete': 'Delete',
    'footnote': 'One object, plain background. ~1 min (first run longer).',
    'history.title': 'History',
    'history.empty': 'Nothing generated yet.',
    'empty.title': 'No model loaded',
    'empty.sub': 'Pick an image on the left or open an entry from history.',
    'progress.title': 'Generating…',
    'step.bg': 'Remove background',
    'step.views': 'Generate views',
    'step.gauss': 'Compute 3D Gaussians',
    'step.write': 'Write model',
    'tool.autorotate': 'Auto-rotate',
    'tool.reset': 'Reset view',
    'tool.bg': 'Toggle background',
    'tool.full': 'Fullscreen',
    'badge.hint': 'drag: rotate · scroll: zoom',
    'badge.loading': 'loading…',
    'badge.engine': 'rendered with aceone',
    'badge.still': 'Still image',
    'splats': 'splats',
    'job.starting': 'Starting…',
    'job.loading': 'Loading models…',
    'job.bg': 'Removing background…',
    'job.loaded': 'Models loaded, starting…',
    'job.views': 'Generating 4 views ({n}/30)',
    'job.gauss': 'Computing 3D Gaussians…',
    'job.write': 'Writing model…',
    'job.preview': 'Rendering preview…',
    'job.done': 'Done',
    'job.subdone': 'in {s} s',
    'toast.done': 'Model ready',
    'toast.deleted': 'Model deleted',
    'toast.deleteFail': 'Delete failed',
    'toast.notImage': 'That is not an image.',
    'toast.tooBig': 'File too large (max. 25 MB).',
    'toast.connection': 'Lost connection to the server. Restart webui.cmd.',
    'toast.uploadFail': 'Upload failed',
    'toast.error': 'Error',
    'toast.disconnected': 'Disconnected',
    'toast.aborted': 'Aborted',
    'toast.unknown': 'Unknown error',
    'toast.webgl': 'WebGL2 unavailable — showing a still image.',
    'toast.viewer': 'Viewer',
    'toast.model': 'Model',
    'toast.png': 'PNG',
    'confirm.delete': 'Delete "{name}"?',
    'ago.now': 'just now',
    'ago.min': '{n} min ago',
    'ago.hour': '{n} h ago',
    'err.busy': 'A job is already running.',
    'err.format': 'Format {ext} not supported',
    'err.empty': 'Empty file',
    'err.too_big': 'File too large (max. 25 MB)',
    'err.upload': 'Upload incomplete',
    'err.lgm': 'LGM aborted',
    'err.no_ply': 'No .ply produced',
    'err.no_engine': 'aceone splat-render.exe not found',
    'err.no_model': 'Model not found',
    'err.render': 'Render failed',
    'err.not_found': 'Not found',
    'err.unknown': 'Unknown',
    'err.delete_running': 'Model is being generated',
  },
};

let lang = new URLSearchParams(location.search).get('lang')
  || localStorage.getItem('splatforge.lang')
  || ((navigator.language || 'en').toLowerCase().startsWith('de') ? 'de' : 'en');
if (lang !== 'de' && lang !== 'en') lang = 'en';

function t(key, params) {
  const dict = I18N[lang] || I18N.en;
  let s = dict[key] ?? (I18N.en[key] ?? key);
  if (params) for (const [k, v] of Object.entries(params)) s = s.replace(`{${k}}`, v);
  return s;
}

function applyI18n() {
  document.documentElement.lang = lang;
  document.querySelectorAll('[data-i18n]').forEach((el) => { el.textContent = t(el.dataset.i18n); });
  document.querySelectorAll('[data-i18n-title]').forEach((el) => { el.title = t(el.dataset.i18nTitle); });
  document.querySelectorAll('#lang button').forEach((b) => b.classList.toggle('on', b.dataset.lang === lang));
  if (state.current) {
    ui.metaSplats.textContent = `${fmtInt(state.current.gaussians)} ${t('splats')}`;
  }
  if (!state.assets.length) {
    ui.assets.innerHTML = `<p class="muted-empty">${t('history.empty')}</p>`;
  } else {
    renderAssets();
  }
  paintEngine(state.engine);
  if (state.badgeHint) ui.stageBadge.textContent = t('badge.hint');
}

function setLang(next) {
  lang = next;
  localStorage.setItem('splatforge.lang', next);
  applyI18n();
}

/* ------------------------------------------------------------------ state */

const ui = {
  drop: $('drop'), file: $('file'), dropEmpty: $('dropEmpty'), dropPreview: $('dropPreview'),
  dropMeta: $('dropMeta'), go: $('go'), resetBtn: $('resetBtn'),
  assets: $('assets'), assetCount: $('assetCount'),
  viewer: $('viewer'), canvas: $('canvas'), stageEmpty: $('stageEmpty'),
  stageProgress: $('stageProgress'), progTitle: $('progTitle'), progSub: $('progSub'),
  progFill: $('progFill'), steps: $('steps'), stageBadge: $('stageBadge'),
  stagebar: $('stagebar'), metaName: $('metaName'), metaSplats: $('metaSplats'),
  metaSize: $('metaSize'), btnPly: $('btnPly'), btnPng: $('btnPng'), btnDelete: $('btnDelete'),
  btnRotate: $('btnRotate'), btnReset: $('btnReset'), btnBg: $('btnBg'), btnFull: $('btnFull'),
  quality: $('quality'), qualityHint: $('qualityHint'), variants: $('variants'),
  cands: $('cands'), candsList: $('candsList'), btnKeep: $('btnKeep'), btnDiscard: $('btnDiscard'),
  engineDot: $('engineDot'), engineText: $('engineText'), gpuChip: $('gpuChip'),
  toasts: $('toasts'),
};

const state = {
  file: null,
  objectUrl: null,
  job: { state: 'idle' },
  assets: [],
  engine: null,
  current: null,
  polling: null,
  viewer: null,
  bgIndex: 0,
  badgeHint: false,
  candidates: null,
  candIndex: 0,
  candJob: null,
};

const BACKGROUNDS = [[0.945, 0.953, 0.965], [0.09, 0.11, 0.15]];

/* ------------------------------------------------------------------ utils */

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
const fmtInt = (n) => (n == null ? '—' : n.toLocaleString(lang === 'de' ? 'de-DE' : 'en-US'));
const fmtBytes = (n) => (n == null ? '—' : (n / 1048576).toFixed(2).replace('.', lang === 'de' ? ',' : '.') + ' MB');

function fmtAgo(ts) {
  if (!ts) return '';
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 90) return t('ago.now');
  if (s < 3600) return t('ago.min', { n: Math.round(s / 60) });
  if (s < 86400) return t('ago.hour', { n: Math.round(s / 3600) });
  return new Date(ts * 1000).toLocaleDateString(lang === 'de' ? 'de-DE' : 'en-US');
}

function toast(msg, kind = '', title = '') {
  const el = document.createElement('div');
  el.className = 'toast ' + kind;
  if (title && msg) el.innerHTML = `<div><strong>${escapeHtml(title)}</strong><span>${escapeHtml(msg)}</span></div>`;
  else if (title) el.innerHTML = `<div><strong>${escapeHtml(title)}</strong></div>`;
  else el.innerHTML = `<div><span>${escapeHtml(msg)}</span></div>`;
  ui.toasts.appendChild(el);
  setTimeout(() => {
    el.style.transition = 'opacity .25s, transform .25s';
    el.style.opacity = '0';
    el.style.transform = 'translateY(6px)';
    setTimeout(() => el.remove(), 260);
  }, kind === 'err' ? 7000 : 3800);
}

function apiError(data, fallbackKey) {
  if (data && data.error_key) return t(data.error_key, data.error_params);
  return (data && data.error) || t(fallbackKey || 'err.unknown');
}

/* ------------------------------------------------------------------ viewer */

async function ensureViewer() {
  if (state.viewer) return state.viewer;
  try {
    state.viewer = new SplatViewer(ui.canvas, { background: BACKGROUNDS[0] });
    state.viewer.start();
  } catch (e) {
    toast(t('toast.webgl'), 'err', t('toast.viewer'));
    state.viewer = null;
  }
  return state.viewer;
}

async function openAsset(asset) {
  state.current = asset;
  renderAssets();

  ui.stageEmpty.hidden = true;
  ui.stagebar.hidden = false;
  ui.metaName.textContent = asset.name;
  ui.metaSplats.textContent = `${fmtInt(asset.gaussians)} ${t('splats')}`;
  ui.metaSize.textContent = fmtBytes(asset.bytes);
  ui.btnPly.href = asset.ply;
  ui.btnPly.download = asset.name + '.ply';
  ui.stageBadge.hidden = false;

  const v = await ensureViewer();
  if (!v) {
    ui.stageBadge.textContent = t('badge.still');
    const img = new Image();
    img.src = asset.thumb;
    img.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;object-fit:contain';
    ui.viewer.appendChild(img);
    return;
  }
  ui.stageBadge.textContent = t('badge.loading');
  state.badgeHint = false;
  try {
    await v.load(asset.ply);
    v.reset();
    v.setAutoRotate(ui.btnRotate.classList.contains('on'));
    state.badgeHint = true;
    ui.stageBadge.textContent = t('badge.hint');
  } catch (e) {
    toast(e.message, 'err', t('toast.model'));
    ui.stageBadge.textContent = '';
  }
}

function clearStage() {
  state.current = null;
  state.badgeHint = false;
  ui.cands.hidden = true;
  ui.stageEmpty.hidden = false;
  ui.stagebar.hidden = true;
  ui.stageBadge.hidden = true;
  renderAssets();
}

/* ------------------------------------------------------------------ assets */

async function loadAssets() {
  try {
    const r = await fetch('/api/assets');
    const data = await r.json();
    state.assets = data.assets || [];
    state.engine = data.engine || null;
    paintEngine(state.engine);
    renderAssets();
  } catch (e) { /* Server weg */ }
}

function paintEngine(engine) {
  if (!engine) return;
  const ok = engine.available;
  ui.engineDot.className = 'dot ' + (ok ? 'on' : 'off');
  ui.engineText.textContent = ok ? t('engine.ready') : t('engine.missing');
  if (engine.adapter) ui.gpuChip.textContent = engine.adapter;
}

function renderAssets() {
  ui.assetCount.textContent = state.assets.length ? String(state.assets.length) : '';
  if (!state.assets.length) {
    ui.assets.innerHTML = `<p class="muted-empty">${t('history.empty')}</p>`;
    return;
  }
  const frag = document.createDocumentFragment();
  for (const a of state.assets) {
    const row = document.createElement('div');
    row.className = 'asset' + (state.current && state.current.id === a.id ? ' active' : '');
    row.innerHTML = `
      <div class="asset-thumb" style="background-image:url('${a.thumb}')"></div>
      <div class="asset-body">
        <div class="asset-name">${escapeHtml(a.name)}</div>
        <div class="asset-sub">${fmtInt(a.gaussians)} ${t('splats')} · ${fmtAgo(a.created)}</div>
      </div>
      <button class="asset-x" title="${t('btn.delete')}">
        <svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6L6 18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
      </button>`;
    row.addEventListener('click', (e) => {
      if (e.target.closest('.asset-x')) return;
      openAsset(a);
    });
    row.querySelector('.asset-x').addEventListener('click', async (e) => {
      e.stopPropagation();
      if (!confirm(t('confirm.delete', { name: a.name }))) return;
      const res = await fetch('/api/assets/' + encodeURIComponent(a.id), { method: 'DELETE' });
      if (!res.ok) return toast(apiError(await safeJson(res), 'toast.deleteFail'), 'err');
      if (state.current && state.current.id === a.id) clearStage();
      toast(t('toast.deleted'), 'ok');
      loadAssets();
    });
    frag.appendChild(row);
  }
  ui.assets.innerHTML = '';
  ui.assets.appendChild(frag);
}

async function safeJson(res) {
  try { return await res.json(); } catch (e) { return null; }
}

/* ------------------------------------------------------------------ upload */

function setFile(file) {
  if (!file) return;
  if (!/^image\//.test(file.type)) return toast(t('toast.notImage'), 'err');
  if (file.size > 25 * 1024 * 1024) return toast(t('toast.tooBig'), 'err');
  state.file = file;
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = URL.createObjectURL(file);
  ui.dropPreview.src = state.objectUrl;
  ui.dropPreview.hidden = false;
  ui.dropEmpty.hidden = true;
  ui.dropMeta.hidden = false;
  ui.dropMeta.textContent = `${file.name} · ${(file.size / 1024).toFixed(0)} KB`;
  ui.resetBtn.hidden = false;
  ui.go.disabled = false;
}

function clearFile() {
  state.file = null;
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = null;
  ui.dropPreview.hidden = true;
  ui.dropEmpty.hidden = false;
  ui.dropMeta.hidden = true;
  ui.resetBtn.hidden = true;
  ui.go.disabled = true;
  ui.file.value = '';
}

ui.drop.addEventListener('click', () => ui.file.click());
ui.drop.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); ui.file.click(); }
});
ui.file.addEventListener('change', () => setFile(ui.file.files[0]));
ui.resetBtn.addEventListener('click', (e) => { e.stopPropagation(); clearFile(); });

['dragenter', 'dragover'].forEach((ev) =>
  ui.drop.addEventListener(ev, (e) => { e.preventDefault(); ui.drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) =>
  ui.drop.addEventListener(ev, (e) => { e.preventDefault(); ui.drop.classList.remove('over'); }));
ui.drop.addEventListener('drop', (e) => {
  const f = e.dataTransfer?.files?.[0];
  if (f) setFile(f);
});
window.addEventListener('paste', (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'));
  if (item) setFile(item.getAsFile());
});

/* ------------------------------------------------------------------ generate */

async function generate() {
  if (!state.file) return;
  ui.go.disabled = true;
  ui.go.querySelector('.btn-label').textContent = t('btn.running');
  ui.steps.querySelectorAll('li').forEach((li) => li.className = '');

  let res, data;
  try {
    const steps = ui.quality ? ui.quality.value : '20';
    const count = ui.variants ? ui.variants.value : '1';
    await discardCandidates();
    res = await fetch('/api/generate?name=' + encodeURIComponent(state.file.name)
      + '&steps=' + steps + '&count=' + count, {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: state.file,
    });
    data = await safeJson(res);
  } catch (e) {
    toast(t('toast.connection'), 'err', t('toast.disconnected'));
    return finishJobUi();
  }
  if (!res.ok) {
    if (res.status === 409 && data && data.running) return startPolling();
    toast(apiError(data, 'toast.uploadFail'), 'err', t('toast.error'));
    return finishJobUi();
  }
  startPolling();
}

ui.go.addEventListener('click', generate);
document.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !ui.go.disabled && document.activeElement === document.body) generate();
});

function setStage(stage, progress) {
  [...ui.steps.querySelectorAll('li')].forEach((li, i) => {
    li.className = i < stage ? 'done' : (i === stage ? 'active' : '');
  });
  ui.progFill.style.width = Math.max(3, Math.min(100, Math.round(progress * 100))) + '%';
}

function startPolling() {
  ui.stageEmpty.hidden = true;
  ui.stageProgress.hidden = false;
  ui.stagebar.hidden = true;
  if (state.polling) clearInterval(state.polling);
  state.polling = setInterval(tick, 800);
  tick();
}

async function tick() {
  let job;
  try {
    job = await (await fetch('/api/job')).json();
  } catch (e) {
    clearInterval(state.polling); state.polling = null;
    ui.stageProgress.hidden = true;
    ui.stageEmpty.hidden = false;
    toast(t('toast.connection'), 'err', t('toast.disconnected'));
    return finishJobUi();
  }
  state.job = job;

  if (job.state === 'running') {
    ui.progTitle.textContent = t('progress.title');
    if (job.count > 1 && job.msg_key === 'job.views') {
      ui.progSub.textContent = t('cand.title', { i: (job.cand || 0) + 1, n: job.count })
        + ' · ' + t('job.views', job.msg_params);
    } else {
      ui.progSub.textContent = job.msg_key ? t(job.msg_key, job.msg_params) : (job.message || '…');
    }
    setStage(job.stage ?? 0, job.progress ?? 0);
  } else if (job.state === 'done') {
    clearInterval(state.polling); state.polling = null;
    setStage(4, 1);
    ui.progTitle.textContent = t('job.done');
    ui.progSub.textContent = t('job.subdone', { s: job.elapsed.toFixed(0) });
    setTimeout(async () => {
      ui.stageProgress.hidden = true;
      finishJobUi();
      if (job.candidates && job.candidates.length > 1) {
        showCandidates(job);
        toast(t('toast.done'), 'ok', state.file ? state.file.name : '');
        return;
      }
      await loadAssets();
      const a = state.assets.find((x) => x.id === job.name) || state.assets[0];
      if (a) openAsset(a); else clearStage();
      toast(t('toast.done'), 'ok', state.file ? state.file.name : '');
    }, 600);
  } else if (job.state === 'error') {
    clearInterval(state.polling); state.polling = null;
    ui.stageProgress.hidden = true;
    ui.stageEmpty.hidden = false;
    const head = job.error_key ? t(job.error_key, job.error_params) : t('toast.aborted');
    toast(job.error_detail || '', 'err', head);
    finishJobUi();
  }
}

function finishJobUi() {
  ui.go.disabled = !state.file;
  ui.go.querySelector('.btn-label').textContent = t('btn.generate');
}

/* ------------------------------------------------------------------ Kandidaten */

async function discardCandidates() {
  if (!state.candJob) return;
  try { await fetch('/api/discard?job=' + state.candJob, { method: 'POST' }); } catch (e) { /* egal */ }
  state.candidates = null;
  state.candJob = null;
  ui.cands.hidden = true;
}

function showCandidates(job) {
  state.candidates = job.candidates;
  state.candJob = job.id;
  state.candIndex = 0;
  ui.stageEmpty.hidden = true;
  ui.stagebar.hidden = true;
  ui.cands.hidden = false;

  const n = state.candidates.length;
  ui.candsList.innerHTML = '';
  state.candidates.forEach((c, i) => {
    const b = document.createElement('button');
    b.className = 'cand' + (i === 0 ? ' active' : '');
    b.style.backgroundImage = `url('${c.thumb}')`;
    b.innerHTML = `<span>${i + 1}</span>`;
    b.title = t('cand.title', { i: i + 1, n });
    b.addEventListener('click', () => pickCandidate(i));
    ui.candsList.appendChild(b);
  });

  const title = document.createElement('div');
  title.className = 'cands-title';
  title.textContent = t('cand.title', { i: 1, n });
  ui.candsList.prepend(title);

  pickCandidate(0);
}

async function pickCandidate(i) {
  const c = state.candidates[i];
  if (!c) return;
  state.candIndex = i;
  const n = state.candidates.length;
  [...ui.candsList.querySelectorAll('.cand')].forEach((el, k) => el.classList.toggle('active', k === i));
  const title = ui.candsList.querySelector('.cands-title');
  if (title) title.textContent = t('cand.title', { i: i + 1, n });

  ui.stageBadge.hidden = false;
  ui.stageBadge.textContent = t('badge.loading');
  const v = await ensureViewer();
  if (v) {
    try {
      await v.load(c.ply);
      v.reset();
      v.setAutoRotate(ui.btnRotate.classList.contains('on'));
      ui.stageBadge.textContent = t('badge.hint');
    } catch (e) { toast(e.message, 'err', t('toast.model')); }
  }
}

ui.btnKeep.addEventListener('click', async () => {
  if (!state.candidates) return;
  ui.btnKeep.disabled = true;
  try {
    const r = await fetch(`/api/select?job=${state.candJob}&index=${state.candIndex}`, { method: 'POST' });
    const data = await safeJson(r);
    if (!r.ok) throw new Error(apiError(data, 'err.unknown'));
    state.candidates = null;
    state.candJob = null;
    ui.cands.hidden = true;
    await loadAssets();
    const a = state.assets.find((x) => x.id === data.id);
    if (a) openAsset(a);
    toast(t('cand.kept'), 'ok');
  } catch (e) {
    toast(e.message, 'err', t('err.unknown'));
  } finally {
    ui.btnKeep.disabled = false;
  }
});

ui.btnDiscard.addEventListener('click', async () => {
  await discardCandidates();
  clearStage();
});

/* ------------------------------------------------------------------ toolbar */

ui.btnRotate.addEventListener('click', () => {
  const on = !ui.btnRotate.classList.contains('on');
  ui.btnRotate.classList.toggle('on', on);
  if (state.viewer) state.viewer.setAutoRotate(on);
});
ui.btnReset.addEventListener('click', () => { if (state.viewer) state.viewer.reset(); });
ui.btnBg.addEventListener('click', () => {
  state.bgIndex = (state.bgIndex + 1) % BACKGROUNDS.length;
  if (state.viewer) { state.viewer.bg = BACKGROUNDS[state.bgIndex]; state.viewer.dirty = true; }
});
ui.btnFull.addEventListener('click', () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else ui.viewer.requestFullscreen?.();
});

document.getElementById('lang').addEventListener('click', (e) => {
  const b = e.target.closest('button[data-lang]');
  if (b) setLang(b.dataset.lang);
});

ui.btnDelete.addEventListener('click', async () => {
  if (!state.current) return;
  if (!confirm(t('confirm.delete', { name: state.current.name }))) return;
  const res = await fetch('/api/assets/' + encodeURIComponent(state.current.id), { method: 'DELETE' });
  if (!res.ok) return toast(apiError(await safeJson(res), 'toast.deleteFail'), 'err');
  clearStage();
  await loadAssets();
  toast(t('toast.deleted'), 'ok');
});

ui.btnPng.addEventListener('click', async () => {
  if (!state.current) return;
  const v = state.viewer;
  const q = new URLSearchParams({ id: state.current.id, width: '1400', height: '1050' });
  if (v) {
    // aceone: yaw 180 = Kamera auf +Z; unser azimuth 0 ebenfalls +Z
    // -> gleiche Ansicht bei yaw = azimuth + 180 (empirisch verifiziert)
    q.set('yaw', String((v.azimuth * 180) / Math.PI + 180));
    q.set('pitch', String((v.elevation * 180) / Math.PI));
    q.set('distance', String(v.distance));
  }
  ui.btnPng.disabled = true;
  const old = ui.btnPng.textContent;
  ui.btnPng.textContent = '…';
  try {
    const r = await fetch('/api/render?' + q.toString(), { method: 'POST' });
    const data = await safeJson(r);
    if (!r.ok) throw new Error(apiError(data, 'err.render'));
    const a = document.createElement('a');
    a.href = data.url;
    a.download = (state.current.name || 'splat') + '.png';
    a.click();
  } catch (e) {
    toast(e.message, 'err', t('toast.png'));
  } finally {
    ui.btnPng.disabled = false;
    ui.btnPng.textContent = old;
  }
});

ui.quality?.addEventListener('change', paintQualityHint);

function paintQualityHint() {
  const map = { '14': '~0:55', '20': '~1:10', '30': '~1:35' };
  ui.qualityHint.textContent = map[ui.quality.value] || '';
}

/* ------------------------------------------------------------------ start */

(async function init() {
  applyI18n();
  paintQualityHint();
  await ensureViewer();
  await loadAssets();
  try {
    const st = await (await fetch('/api/job')).json();
    if (st.state === 'running') { startPolling(); return; }
    if (st.state === 'done' && st.candidates && st.candidates.length > 1) {
      showCandidates(st);   // nach Reload: offene Auswahl wieder anzeigen
      return;
    }
  } catch (e) { /* egal */ }
  if (state.assets.length) openAsset(state.assets[0]);
})();
