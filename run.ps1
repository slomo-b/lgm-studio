<#
.SYNOPSIS
    Bild -> 3D-Gaussian-Splat (.ply) mit LGM, lokal auf Intel-Arc-iGPU (XPU)
    oder CPU -- ohne NVIDIA-Karte.

.EXAMPLE
    .\run.ps1 -Image LGM\data_test\catstatue_rgba.png

.EXAMPLE
    .\run.ps1 -Image C:\Users\Mo\Pictures\pflanze.png -NoPreview
#>
param(
    # Pfad zum Eingabebild (relativ zum LGM-Ordner oder absolut)
    [string]$Image = "data_test\catstatue_rgba.png",
    # Ausgabeordner, relativ zu LGM\
    [string]$Out = "workspace_test",
    # Vorschau-Turntable nach der Generierung ueberspringen
    [switch]$NoPreview
)

# Es existiert nur der "big"-Checkpoint (model_fp16_fixrot.safetensors).
# Die Configs small/tiny haben eine andere Architektur und wuerden mit diesen
# Gewichten stillschweigend Zufallsrauschen erzeugen.
$Config = "big"

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
$lgm = Join-Path $root "LGM"
$ckpt = Join-Path $lgm "pretrained\model_fp16_fixrot.safetensors"

if (-not (Test-Path $py)) { throw "Venv fehlt: $py - siehe SETUP.md" }
if (-not (Test-Path $ckpt)) { throw "Checkpoint fehlt: $ckpt" }

# HF-Cache im Projekt halten, keine Symlink-Warnungen unter Windows
$env:HF_HOME = Join-Path $root ".hf"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"

$t0 = Get-Date
Push-Location $lgm
try {
    & $py infer.py $Config `
        --resume "pretrained/model_fp16_fixrot.safetensors" `
        --workspace $Out `
        --test_path $Image
}
finally {
    Pop-Location
}

$stem = [System.IO.Path]::GetFileNameWithoutExtension($Image)
$ply = Join-Path $lgm "$Out\$stem.ply"
if (-not (Test-Path $ply)) { throw "Keine .ply erzeugt - siehe Ausgabe oben." }

Write-Host ""
Write-Host ("Fertig in {0:N1} min: {1}" -f ((Get-Date) - $t0).TotalMinutes, $ply) -ForegroundColor Green

if (-not $NoPreview) {
    Write-Host "Rendere Vorschau (eigener Renderer, ohne CUDA)..." -ForegroundColor Cyan
    & $py (Join-Path $root "render_preview.py") --ply $ply --size 448 --elev 8 --frames 24
}
