// LGM Studio -- natives Windows-Fenster (Tauri 2) um die bestehende Pipeline.
//
// Arbeitsteilung:
//   Rust     Fenster, Dateidialoge, Prozesssteuerung, Dateisystem, Export
//   WebView  Oberflaeche (HTML/CSS/JS, kein Bundler noetig)
//   Python   nur die Inferenz -- Torch/XPU laesst sich nicht sinnvoll portieren
//
// Der Python-Teil laeuft als Kindprozess; seine Ausgabe wird zeilenweise als
// Event ans Fenster geschickt, damit die Oberflaeche echten Fortschritt zeigt
// und nicht einen Spinner.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Serialize;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tauri::{Emitter, State};

const STEPS_MIN: u32 = 4;
const STEPS_MAX: u32 = 60;

struct App {
    root: PathBuf,
    initial_image: Option<String>,
    autorun: bool,
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

/// Das Manifest liegt beim Bauen im Binary -- kein Pfad-Problem in der
/// installierten App, und es kann nicht versehentlich fehlen.
const MANIFEST: &str = include_str!("../manifest.json");

#[derive(serde::Deserialize, Clone)]
struct MFile {
    id: String,
    path: String,
    url: String,
    size: u64,
}

#[derive(serde::Deserialize)]
struct Manifest {
    files: Vec<MFile>,
}

fn manifest() -> Result<Vec<MFile>, String> {
    let m: Manifest = serde_json::from_str(MANIFEST).map_err(|e| format!("Manifest: {e}"))?;
    Ok(m.files)
}

#[derive(Serialize)]
struct FileStatus {
    path: String,
    size: u64,
    on_disk: u64,
    complete: bool,
}

#[derive(Serialize)]
struct ModelsStatus {
    dir: String,
    files: Vec<FileStatus>,
    total: u64,
    present: u64,
    complete: bool,
}

// ---------------------------------------------------------------- Hilfen

/// Projektwurzel finden: Umgebungsvariable, dann vom Executable aufwaerts.
fn find_root() -> PathBuf {
    if let Ok(p) = std::env::var("LGM_STUDIO_ROOT") {
        let p = PathBuf::from(p);
        if p.join("LGM").join("infer.py").exists() {
            return p;
        }
    }
    let mut cur = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()));
    while let Some(dir) = cur {
        if dir.join("LGM").join("infer.py").exists() {
            return dir;
        }
        cur = dir.parent().map(|p| p.to_path_buf());
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

/// Python finden: erst explizit gesetzt, dann neben den Modellen, dann die
/// Projekt-Umgebung, zuletzt eine vom Installer angelegte Laufzeit.
///
/// Die Reihenfolge ist Absicht: wer eine eigene Umgebung hat, soll sie
/// vorgeben koennen, ohne dass die App etwas ueberschreibt.
fn python(root: &Path) -> PathBuf {
    if let Ok(p) = std::env::var("LGM_STUDIO_PYTHON") {
        let p = PathBuf::from(p);
        if p.exists() {
            return p;
        }
    }
    let candidates = [
        root.join("runtime").join("python").join("python.exe"),
        root.join("runtime").join("python.exe"),
        root.join(".venv").join("Scripts").join("python.exe"),
    ];
    for c in candidates {
        if c.exists() {
            return c;
        }
    }
    // vom Installer angelegte Laufzeit
    if let Ok(local) = std::env::var("LOCALAPPDATA") {
        let p = PathBuf::from(local)
            .join("LGMStudio")
            .join("runtime")
            .join("python.exe");
        if p.exists() {
            return p;
        }
    }
    root.join(".venv").join("Scripts").join("python.exe")
}

fn python_simready(root: &Path) -> PathBuf {
    root.join(".venv-simready")
        .join("Scripts")
        .join("python.exe")
}

#[derive(Serialize)]
struct EnvInfo {
    root: String,
    python: bool,
    python_simready: bool,
    checkpoint: bool,
    ok: bool,
    autorun: bool,
    autodownload: bool,
    setupruntime: bool,
    problems: Vec<String>,
}

#[derive(Serialize, Clone)]
struct Emit {
    line: String,
}

// ---------------------------------------------------------------- Befehle

#[tauri::command]
fn env_info(app: State<App>) -> EnvInfo {
    let root = app.root.clone();
    let py = python(&root).exists();
    let py2 = python_simready(&root).exists();
    let ckpt = root
        .join("LGM")
        .join("pretrained")
        .join("model_fp16_fixrot.safetensors")
        .exists();

    let mut problems = Vec::new();
    if !py {
        problems.push("Python-Umgebung fehlt (.venv)".into());
    }
    if !ckpt {
        problems.push("Modellgewichte fehlen (LGM/pretrained)".into());
    }
    if !py2 {
        problems.push("SimReady-Umgebung fehlt (.venv-simready)".into());
    }

    EnvInfo {
        root: root.display().to_string(),
        python: py,
        python_simready: py2,
        checkpoint: ckpt,
        ok: problems.is_empty(),
        autorun: app.autorun,
        autodownload: std::env::var("LGM_STUDIO_AUTODOWNLOAD").ok().as_deref() == Some("1"),
        setupruntime: std::env::var("LGM_STUDIO_SETUP_RUNTIME").ok().as_deref() == Some("1"),
        problems,
    }
}

/// Bild, das beim Start uebergeben wurde (Doppelklick auf eine Datei).
#[tauri::command]
fn initial_image(app: State<App>) -> Option<String> {
    app.initial_image.clone()
}

/// Nativer Dateidialog -- kein Web-Input, das ist der Unterschied zu einer WebUI.
#[tauri::command]
fn pick_image() -> Option<String> {
    rfd::FileDialog::new()
        .add_filter("Bilder", &["png", "jpg", "jpeg", "webp", "bmp"])
        .set_title("Bild auswaehlen")
        .pick_file()
        .map(|p| p.display().to_string())
}

#[tauri::command]
fn pick_folder() -> Option<String> {
    rfd::FileDialog::new()
        .set_title("Zielordner waehlen")
        .pick_folder()
        .map(|p| p.display().to_string())
}

/// Datei als Bytes lesen. Der Viewer bekommt die Daten direkt aus Rust,
/// damit im WebView kein HTTP-Server noetig ist.
#[tauri::command]
fn read_file(path: String) -> Result<Vec<u8>, String> {
    std::fs::read(&path).map_err(|e| format!("{path}: {e}"))
}

#[tauri::command]
fn reveal(path: String) {
    let p = Path::new(&path);
    let target = if p.is_dir() {
        p.to_path_buf()
    } else {
        p.parent().map(|d| d.to_path_buf()).unwrap_or_else(|| p.to_path_buf())
    };
    // explorer.exe ist auf jedem Windows vorhanden -- keine Extra-Abhaengigkeit
    let _ = Command::new("explorer").arg(target).spawn();
}

#[derive(Serialize)]
struct GenResult {
    ok: bool,
    ply: Option<String>,
    workspace: String,
    seconds: f32,
    log: Vec<String>,
}

/// Startet die Inferenz. Laeuft blockierend in einem eigenen Thread, damit die
/// Oberflaeche bedienbar bleibt; jede Ausgabezeile geht als Event raus.
#[tauri::command]
async fn generate(
    window: tauri::Window,
    app: State<'_, App>,
    image: String,
    steps: u32,
) -> Result<GenResult, String> {
    let root = app.root.clone();
    let py = python(&root);
    if !py.exists() {
        return Err("Python-Umgebung fehlt (.venv)".into());
    }
    if !Path::new(&image).exists() {
        return Err(format!("Bild nicht gefunden: {image}"));
    }

    let steps = steps.clamp(STEPS_MIN, STEPS_MAX);
    let stem = Path::new(&image)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "asset".into());

    // Checkpoint bevorzugt aus dem Download-Ordner, sonst aus dem Projekt.
    let local_ckpt = models_dir(&app).join("lgm").join("model_fp16_fixrot.safetensors");
    let checkpoint = if local_ckpt.exists() {
        local_ckpt
    } else {
        root.join("LGM")
            .join("pretrained")
            .join("model_fp16_fixrot.safetensors")
    };
    let workspace = root.join("out").join("studio").join(&stem);
    let _ = std::fs::create_dir_all(&workspace);

    let win = window.clone();
    let ws = workspace.clone();
    let started = std::time::Instant::now();

    // Lauf-Protokoll zusaetzlich auf Platte: die Oberflaeche kann fehlen,
    // abgeschnitten sein oder abstuerzen -- das Log bleibt.
    let log_path = workspace.join("run.log");
    let log_file = std::sync::Arc::new(std::sync::Mutex::new(
        std::fs::File::create(&log_path).ok(),
    ));

    let result = tauri::async_runtime::spawn_blocking(move || {
        let mut cmd = Command::new(&py);
        cmd.current_dir(&root)
            .arg("LGM/infer.py").arg("big")
            .arg("--resume").arg(&checkpoint)
            .arg("--workspace").arg(&ws)
            .arg("--test_path").arg(&image)
            .env("LGM_STEPS", steps.to_string())
            .env("PYTHONUNBUFFERED", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => return Err(format!("Start fehlgeschlagen: {e}")),
        };

        let mut log = Vec::new();
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();

        let mut push = {
            let lf = log_file.clone();
            move |line: &str| {
                if let Ok(mut f) = lf.lock() {
                    if let Some(f) = f.as_mut() {
                        use std::io::Write;
                        let _ = writeln!(f, "{line}");
                        let _ = f.flush();
                    }
                }
            }
        };
        push(&format!("python: {}", py.display()));
        push(&format!("bild:   {image}"));
        push(&format!("LGM_STEPS={steps}"));

        // stderr in eigenem Thread, sonst blockiert ein voller Puffer
        let err_lines = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let el = err_lines.clone();
        let lf_err = log_file.clone();
        let h = std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                if let Ok(mut f) = lf_err.lock() {
                    if let Some(f) = f.as_mut() {
                        use std::io::Write;
                        let _ = writeln!(f, "{line}");
                        let _ = f.flush();
                    }
                }
                if let Ok(mut v) = el.lock() {
                    v.push(line);
                }
            }
        });

        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let _ = win.emit("pipeline-line", Emit { line: line.clone() });
            push(&line);
            log.push(line);
        }
        let _ = child.wait();
        let _ = h.join();
        if let Ok(v) = err_lines.lock() {
            for l in v.iter() {
                if !l.trim().is_empty() {
                    let _ = win.emit("pipeline-line", Emit { line: l.clone() });
                }
            }
            log.extend(v.iter().cloned());
        }

        Ok::<_, String>((log, ws))
    })
    .await
    .map_err(|e| format!("Thread-Fehler: {e}"))?;

    let (log, workspace) = result?;
    let ply = workspace.join(format!("{stem}.ply"));
    Ok(GenResult {
        ok: ply.exists(),
        ply: if ply.exists() {
            Some(ply.display().to_string())
        } else {
            None
        },
        workspace: workspace.display().to_string(),
        seconds: started.elapsed().as_secs_f32(),
        log,
    })
}

#[derive(Serialize)]
struct ExportResult {
    ok: bool,
    usd: Option<String>,
    log: Vec<String>,
}

/// Splat -> SimReady-USD ueber die bestehende Python-Pipeline.
#[tauri::command]
async fn export_simready(
    window: tauri::Window,
    app: State<'_, App>,
    ply: String,
    obj_class: String,
    target_size: Option<f32>,
) -> Result<ExportResult, String> {
    let root = app.root.clone();
    let py = python_simready(&root);
    if !py.exists() {
        return Err("SimReady-Umgebung fehlt (.venv-simready)".into());
    }
    let out_dir = Path::new(&ply)
        .parent()
        .map(|d| d.join("simready"))
        .unwrap_or_else(|| root.join("out").join("simready"));
    let _ = std::fs::create_dir_all(&out_dir);

    let win = window.clone();
    let ply2 = ply.clone();
    let out2 = out_dir.clone();

    let log = tauri::async_runtime::spawn_blocking(move || {
        let mut cmd = Command::new(&py);
        cmd.current_dir(&root)
            .arg("-m").arg("simready").arg("build").arg(&ply2)
            .arg("-o").arg(&out2)
            .arg("--obj-class").arg(&obj_class)
            .arg("--collider").arg("hull")
            .arg("--preview")
            .env("PYTHONUNBUFFERED", "1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(t) = target_size {
            cmd.arg("--target-size").arg(format!("{t}"));
        }
        let mut child = cmd.spawn().map_err(|e| format!("Start fehlgeschlagen: {e}"))?;
        let mut out = String::new();
        if let Some(o) = child.stdout.take() {
            for line in BufReader::new(o).lines().map_while(Result::ok) {
                let _ = win.emit("pipeline-line", Emit { line: line.clone() });
                out.push_str(&line);
                out.push('\n');
            }
        }
        if let Some(e) = child.stderr.take() {
            for line in BufReader::new(e).lines().map_while(Result::ok) {
                let _ = win.emit("pipeline-line", Emit { line: line.clone() });
                out.push_str(&line);
                out.push('\n');
            }
        }
        let _ = child.wait();
        Ok::<String, String>(out)
    })
    .await
    .map_err(|e| format!("Thread-Fehler: {e}"))??;

    let usd = std::fs::read_dir(&out_dir).ok().and_then(|rd| {
        rd.filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.extension().map(|x| x == "usd" || x == "usda").unwrap_or(false))
    });

    Ok(ExportResult {
        ok: usd.is_some(),
        usd: usd.map(|p| p.display().to_string()),
        log: log.lines().map(|s| s.to_string()).collect(),
    })
}

// ---------------------------------------------------------------- Laufzeit

/// Portables CPython von python-build-standalone. Kein Installer, keine
/// Registry, kein Adminrecht -- einfach entpacken und benutzen.
const PY_URL: &str = "https://github.com/astral-sh/python-build-standalone/releases/download/20261026/cpython-3.11.18%2B20261026-x86_64-pc-windows-msvc-install_only.tar.gz";
const PIP_XPU: &str = "https://download.pytorch.org/whl/xpu";

fn runtime_dir(app: &App) -> PathBuf {
    app.root.join("runtime")
}

fn runtime_python(app: &App) -> PathBuf {
    runtime_dir(app).join("python").join("python.exe")
}

#[derive(Serialize)]
struct RuntimeStatus {
    dir: String,
    python: String,
    installed: bool,
    /// Pfad, den die App fuer die Inferenz benutzt (kann die .venv sein)
    active: String,
}

#[tauri::command]
fn runtime_status(app: State<App>) -> RuntimeStatus {
    let py = runtime_python(&app);
    RuntimeStatus {
        dir: runtime_dir(&app).display().to_string(),
        python: py.display().to_string(),
        installed: py.exists(),
        active: python(&app.root).display().to_string(),
    }
}

#[derive(Serialize, Clone)]
struct SetupLine {
    line: String,
    phase: String,
}

fn emit_line(window: &tauri::Window, phase: &str, line: &str) {
    let _ = window.emit(
        "setup-line",
        SetupLine {
            line: line.to_string(),
            phase: phase.to_string(),
        },
    );
    // Zusaetzlich auf Platte: pip laeuft Minuten, und ein Fehler dort ist
    // sonst nur im (abgeschnittenen) Fenster sichtbar.
    if let Ok(p) = std::env::var("LGM_STUDIO_SETUP_LOG") {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(p) {
            let _ = writeln!(f, "[{phase}] {line}");
        }
    }
}

/// Entpackt ein .tar.gz und liefert die Zahl der Dateien.
fn untar_gz(archive: &Path, dest: &Path) -> Result<usize, String> {
    let f = std::fs::File::open(archive).map_err(|e| format!("{archive:?}: {e}"))?;
    let gz = flate2::read::GzDecoder::new(std::io::BufReader::new(f));
    let mut ar = tar::Archive::new(gz);
    ar.set_preserve_permissions(false);
    ar.unpack(dest).map_err(|e| format!("entpacken: {e}"))?;
    Ok(std::fs::read_dir(dest).map(|d| d.count()).unwrap_or(0))
}

/// Laedt die Python-Laufzeit und installiert die Pakete.
///
/// Zwei Phasen, weil sie sehr unterschiedlich lange dauern: 49 MB entpacken
/// (Sekunden) und ~2,5 GB Wheels (Minuten). Die Oberflaeche bekommt beide
/// getrennt gemeldet.
#[tauri::command]
async fn install_runtime(window: tauri::Window, app: State<'_, App>) -> Result<(), String> {
    let dir = runtime_dir(&app);
    std::fs::create_dir_all(&dir).map_err(|e| format!("Ordner: {e}"))?;
    let root = app.root.clone();
    let win = window.clone();
    let win2 = window.clone();

    tauri::async_runtime::spawn_blocking(move || -> Result<(), String> {
        let py = dir.join("python").join("python.exe");

        // ---- Phase 1: Interpreter
        if !py.exists() {
            emit_line(&win, "python", "Lade CPython 3.11 (48,7 MB) …");
            let tgz = dir.join("cpython.tar.gz");
            let client = reqwest::blocking::Client::builder()
                .user_agent("LGM-Studio/0.1")
                .timeout(std::time::Duration::from_secs(900))
                .build()
                .map_err(|e| format!("HTTP: {e}"))?;
            let mut resp = client.get(PY_URL).send().map_err(|e| format!("Download: {e}"))?;
            if !resp.status().is_success() {
                return Err(format!("Download: HTTP {}", resp.status()));
            }
            let total = resp.content_length().unwrap_or(0);
            let mut f = std::fs::File::create(&tgz).map_err(|e| format!("{tgz:?}: {e}"))?;
            let mut buf = vec![0u8; 1 << 20];
            let mut done = 0u64;
            let mut last = std::time::Instant::now();
            use std::io::{Read, Write};
            loop {
                let n = resp.read(&mut buf).map_err(|e| format!("Download: {e}"))?;
                if n == 0 {
                    break;
                }
                f.write_all(&buf[..n]).map_err(|e| format!("schreiben: {e}"))?;
                done += n as u64;
                if last.elapsed().as_millis() >= 200 {
                    last = std::time::Instant::now();
                    let _ = win.emit(
                        "runtime-progress",
                        serde_json::json!({"phase": "python", "done": done, "total": total}),
                    );
                }
            }
            f.flush().ok();
            drop(f);

            emit_line(&win, "python", "Entpacke …");
            untar_gz(&tgz, &dir)?;
            let _ = std::fs::remove_file(&tgz);
        }
        if !py.exists() {
            return Err("python.exe nach dem Entpacken nicht gefunden".into());
        }
        emit_line(&win, "python", "Interpreter bereit.");

        // ---- Phase 2: Pakete
        let run = |args: &[&str], label: &str| -> Result<(), String> {
            emit_line(&win, "pip", label);
            let mut cmd = Command::new(&py);
            cmd.args(args).current_dir(&root).env("PYTHONUNBUFFERED", "1");
            cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
            let mut child = cmd.spawn().map_err(|e| format!("{label}: {e}"))?;
            let w1 = win2.clone();
            let l1 = label.to_string();
            let t1 = child.stdout.take().unwrap();
            let t2 = child.stderr.take().unwrap();
            let h1 = std::thread::spawn(move || {
                for line in BufReader::new(t1).lines().map_while(Result::ok) {
                    emit_line(&w1, "pip", &line);
                }
            });
            let w2 = win2.clone();
            let l2 = label.to_string();
            let h2 = std::thread::spawn(move || {
                for line in BufReader::new(t2).lines().map_while(Result::ok) {
                    emit_line(&w2, "pip", &line);
                }
            });
            let _ = (l1, l2);
            let status = child.wait().map_err(|e| format!("{label}: {e}"))?;
            let _ = h1.join();
            let _ = h2.join();
            if !status.success() {
                return Err(format!("{label} fehlgeschlagen ({status})"));
            }
            Ok(())
        };

        run(&["-m", "pip", "install", "--upgrade", "pip"], "pip aktualisieren …")?;
        run(
            &[
                "-m", "pip", "install", "torch", "torchvision",
                "--index-url", PIP_XPU,
            ],
            "PyTorch für Intel-XPU installieren (~1,5 GB) …",
        )?;
        run(
            &["-m", "pip", "install", "-r", "requirements-local.txt"],
            "Abhängigkeiten installieren …",
        )?;

        emit_line(&win, "fertig", "Laufzeit eingerichtet.");
        Ok(())
    })
    .await
    .map_err(|e| format!("Thread: {e}"))?
}



/// Windows muss wissen, dass dieser Prozess in echten Pixeln rechnet.
///
/// Ohne das liefert Windows virtuelle Koordinaten (Fenster 1094 px), waehrend
/// WebView2 in physikalischen Pixeln zeichnet (1367 px) -- die Seite ist dann
/// 1,25x breiter als das Fenster und die rechte Spalte wird abgeschnitten.
/// Der Aufruf muss vor dem ersten Fenster passieren, deshalb ganz oben in main.
// ---------------------------------------------------------------- Modelle

fn models_dir(app: &App) -> PathBuf {
    if let Ok(p) = std::env::var("LGM_STUDIO_MODELS") {
        return PathBuf::from(p);
    }
    app.root.join("models")
}

fn file_on_disk(app: &App, f: &MFile) -> u64 {
    let p = models_dir(app).join(&f.path);
    std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0)
}

#[tauri::command]
fn models_status(app: State<App>) -> ModelsStatus {
    let files = manifest().unwrap_or_default();
    let mut out = Vec::new();
    let (mut total, mut present) = (0u64, 0u64);
    for f in &files {
        let on_disk = file_on_disk(&app, f);
        total += f.size;
        present += on_disk.min(f.size);
        out.push(FileStatus {
            path: f.path.clone(),
            size: f.size,
            on_disk,
            complete: on_disk >= f.size,
        });
    }
    ModelsStatus {
        dir: models_dir(&app).display().to_string(),
        complete: out.iter().all(|f| f.complete),
        files: out,
        total,
        present,
    }
}

#[derive(Serialize, Clone)]
struct DlProgress {
    path: String,
    file_done: u64,
    file_total: u64,
    overall_done: u64,
    overall_total: u64,
    done: bool,
    error: Option<String>,
}

#[tauri::command]
fn cancel_download(app: State<App>) {
    app.cancel
        .store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Laedt alle fehlenden Modelldateien nach.
///
/// Fortschritt geht als Event `dl-progress` raus (hoechstens ~20/s). Abbrueche
/// sind unschaedlich: jede Datei liegt waehrend des Ladens als `.part` und
/// wird per Range-Request fortgesetzt, nicht neu begonnen.
fn dlog(dir: &Path, msg: &str) {
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("download.log"))
    {
        let _ = writeln!(f, "{msg}");
    }
}

#[tauri::command]
async fn download_models(window: tauri::Window, app: State<'_, App>) -> Result<(), String> {
    let dir = models_dir(&app);
    std::fs::create_dir_all(&dir).map_err(|e| format!("Ordner: {e}"))?;
    let files = manifest()?;
    app.cancel
        .store(false, std::sync::atomic::Ordering::Relaxed);
    let cancel = app.cancel.clone();
    let win = window.clone();
    // reqwest::blocking darf nicht in der Async-Runtime laufen -> eigener Thread
    tauri::async_runtime::spawn_blocking(move || download_all(win, dir, cancel, files))
        .await
        .map_err(|e| format!("Thread: {e}"))?
}

fn download_all(
    window: tauri::Window,
    dir: PathBuf,
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    files: Vec<MFile>,
) -> Result<(), String> {
    dlog(&dir, &format!("=== Start {} ===", dir.display()));

    let overall_total: u64 = files.iter().map(|f| f.size).sum();
    let mut overall_done: u64 = files
        .iter()
        .map(|f| {
            let p = dir.join(&f.path);
            if p.exists() {
                f.size
            } else {
                std::fs::metadata(dir.join(format!("{}.part", f.path)))
                    .map(|m| m.len())
                    .unwrap_or(0)
            }
        })
        .sum();

    let client = reqwest::blocking::Client::builder()
        .user_agent("LGM-Studio/0.1")
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|e| format!("HTTP: {e}"))?;

    for f in &files {
        let final_path = dir.join(&f.path);
        if final_path.exists() && std::fs::metadata(&final_path).map(|m| m.len()).unwrap_or(0) >= f.size
        {
            continue;
        }
        if let Some(parent) = final_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("Ordner: {e}"))?;
        }
        let part_path = dir.join(format!("{}.part", f.path));
        let mut have = std::fs::metadata(&part_path).map(|m| m.len()).unwrap_or(0);
        if have > f.size {
            let _ = std::fs::remove_file(&part_path);
            have = 0;
        }

        let mut req = client.get(&f.url);
        if have > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={}-", have));
        }
        let resp = match req.send() {
            Ok(r) => r,
            Err(e) => {
                dlog(&dir, &format!("FEHLER {}: {e}", f.url));
                let _ = window.emit(
                    "dl-progress",
                    DlProgress {
                        path: f.path.clone(),
                        file_done: have,
                        file_total: f.size,
                        overall_done,
                        overall_total,
                        done: true,
                        error: Some(format!("{}: {e}", f.id)),
                    },
                );
                return Err(format!("{}: {e}", f.id));
            }
        };
        if !resp.status().is_success() && resp.status().as_u16() != 206 {
            dlog(&dir, &format!("HTTP {} fuer {}", resp.status(), f.url));
            return Err(format!("{}: HTTP {}", f.id, resp.status()));
        }
        dlog(&dir, &format!("OK {} ({} Bytes, ab {})", f.path, f.size, have));
        if resp.status().as_u16() != 206 {
            have = 0; // Server ignoriert Range -> von vorn
        }

        use std::io::{Read, Write};
        let mut out = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&part_path)
            .map_err(|e| format!("{}.part: {e}", f.id))?;
        let mut reader = resp;
        let mut buf = vec![0u8; 1 << 20];
        let mut last_emit = std::time::Instant::now();

        loop {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                let _ = out.flush();
                return Err("abgebrochen".into());
            }
            let n = match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) => return Err(format!("{}: {e}", f.id)),
            };
            out.write_all(&buf[..n]).map_err(|e| format!("{}.part: {e}", f.id))?;
            have += n as u64;
            overall_done += n as u64;
            if last_emit.elapsed().as_millis() >= 50 {
                last_emit = std::time::Instant::now();
                let _ = window.emit(
                    "dl-progress",
                    DlProgress {
                        path: f.path.clone(),
                        file_done: have,
                        file_total: f.size,
                        overall_done,
                        overall_total,
                        done: false,
                        error: None,
                    },
                );
            }
        }
        out.flush().ok();
        drop(out);
        std::fs::rename(&part_path, &final_path).map_err(|e| format!("umbenennen: {e}"))?;
        let _ = window.emit(
            "dl-progress",
            DlProgress {
                path: f.path.clone(),
                file_done: f.size,
                file_total: f.size,
                overall_done,
                overall_total,
                done: true,
                error: None,
            },
        );
    }
    Ok(())
}

#[cfg(windows)]
fn enable_per_monitor_dpi() {
    #[link(name = "user32")]
    extern "system" {
        fn SetProcessDpiAwarenessContext(value: isize) -> i32;
    }
    const PER_MONITOR_AWARE_V2: isize = -4;
    unsafe {
        let _ = SetProcessDpiAwarenessContext(PER_MONITOR_AWARE_V2);
    }
}

#[cfg(not(windows))]
fn enable_per_monitor_dpi() {}

fn main() {
    enable_per_monitor_dpi();

    let root = find_root();

    // Startbild optional: erlaubt "Oeffnen mit" und automatisierte Tests.
    // --auto loest zusaetzlich sofort eine Generierung aus.
    let args: Vec<String> = std::env::args().collect();
    let arg = args.iter().skip(1).find(|a| Path::new(a).is_file()).cloned();
    let autorun = args.iter().any(|a| a == "--auto");

    tauri::Builder::default()
        .manage(App {
            root,
            initial_image: arg,
            autorun,
            cancel: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
        .setup(|_app| Ok(()))
        .invoke_handler(tauri::generate_handler![
            env_info,
            initial_image,
            pick_image,
            pick_folder,
            reveal,
            read_file,
            generate,
            export_simready,
            models_status,
            download_models,
            cancel_download,
            runtime_status,
            install_runtime
        ])
        .run(tauri::generate_context!())
        .expect("Fehler beim Start der Anwendung");
}
