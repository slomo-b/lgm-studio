use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use splatcage::carve::carve;
use splatcage::hull::Hull;
use splatcage::ply::Ply;
use splatcage::views::{Mask, MaskMode, View};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "splatcage", about = "Raeumliche Schranke fuer Gaussian Splats", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Voxelhuelle aus Silhouetten carven
    Build(BuildArgs),
    /// Splats ausserhalb der Huelle entfernen
    Filter(FilterArgs),
    /// Huelle anzeigen
    Info { hull: PathBuf },
}

#[derive(clap::Args)]
struct BuildArgs {
    /// JSON: mask, width, height, fx, fy, cx, cy, world_to_cam
    #[arg(long)]
    views: Option<PathBuf>,
    /// LGM-Shorthand: Verzeichnis mit <stem>_view0..3.png
    #[arg(long)]
    lgm_dir: Option<PathBuf>,
    #[arg(long)]
    stem: Option<String>,
    #[arg(long, default_value_t = 49.1)]
    fovy: f32,
    #[arg(long, default_value_t = 1.6)]
    radius: f32,
    #[arg(long, default_value_t = 0.0)]
    elev: f32,
    /// Drehung der Ansichtsreihenfolge, falls die Posen anders liegen
    #[arg(long, default_value_t = 0.0)]
    azim_offset: f32,
    /// Voxel pro laengster Achse
    #[arg(long, default_value_t = 256)]
    res: u32,
    /// min:max, je drei Kommawerte
    #[arg(long, default_value = "-1.2,-1.2,-1.2:1.2,1.2,1.2")]
    bounds: String,
    /// Zusaetzlich mit der robusten Bounding-Box dieser PLY einschraenken.
    ///
    /// Vier Ansichten auf Hoehe 0 koennen die Hochachse nicht bestimmen -- die
    /// Silhouettenhuelle wird dann ein Prisma durch die ganze Box. Die eigene
    /// Bounding-Box der Splats liefert genau diese fehlende Information.
    #[arg(long)]
    ply: Option<PathBuf>,
    /// Perzentil fuer die robuste Box (gegen fliegende Splats)
    #[arg(long, default_value_t = 0.005)]
    ply_pad: f32,
    /// Huelle um n Voxel aufblasen (Toleranz fuer Splat-Radien)
    #[arg(long, default_value_t = 0)]
    margin: u32,
    /// Nur die groesste zusammenhaengende Komponente behalten
    #[arg(long, default_value_t = true, action = clap::ArgAction::SetTrue)]
    largest: bool,
    #[arg(long, default_value = "hull.bin")]
    out: PathBuf,
    /// Huelle zusaetzlich als Punktwolke schreiben (Sichtpruefung)
    #[arg(long)]
    obj: Option<PathBuf>,
    #[arg(long, default_value_t = 240)]
    threshold: u8,
    #[arg(long, value_enum, default_value_t = MaskMode::Auto)]
    mask_mode: MaskMode,
}

#[derive(clap::Args)]
struct FilterArgs {
    #[arg(long)]
    hull: PathBuf,
    #[arg(long)]
    ply: PathBuf,
    #[arg(short, long)]
    out: PathBuf,
}

fn parse_bounds(s: &str) -> Result<([f32; 3], [f32; 3])> {
    let (lo, hi) = s
        .split_once(':')
        .context("--bounds erwartet min:max, z.B. -1.2,-1.2,-1.2:1.2,1.2,1.2")?;
    let parse3 = |t: &str| -> Result<[f32; 3]> {
        let v: Vec<f32> = t
            .split(',')
            .map(|x| x.trim().parse::<f32>())
            .collect::<Result<_, _>>()
            .context("Zahl in --bounds erwartet")?;
        if v.len() != 3 {
            bail!("--bounds braucht drei Werte je Seite, bekommen: {}", t);
        }
        Ok([v[0], v[1], v[2]])
    };
    Ok((parse3(lo)?, parse3(hi)?))
}

fn load_views(a: &BuildArgs) -> Result<Vec<View>> {
    if let Some(p) = &a.views {
        return View::load_json(p, a.mask_mode, a.threshold);
    }
    let (dir, stem) = match (&a.lgm_dir, &a.stem) {
        (Some(d), Some(s)) => (d, s),
        _ => bail!("entweder --views <json> oder --lgm-dir <dir> --stem <name>"),
    };
    let mut views = Vec::new();
    for i in 0..4u32 {
        let path = dir.join(format!("{}_{}view{}.png", stem, "", i));
        let alt = dir.join(format!("{}_view{}.png", stem, i));
        let p = if path.exists() { path } else { alt };
        if !p.exists() {
            bail!("{} fehlt", p.display());
        }
        let mask = Mask::load(&p, a.mask_mode, a.threshold)?;
        let view = View::orbit(mask, a.azim_offset + 90.0 * i as f32, a.elev, a.radius, a.fovy);
        views.push(view);
    }
    Ok(views)
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Build(a) => {
            let views = load_views(&a)?;
            for (i, v) in views.iter().enumerate() {
                println!(
                    "[view {i}] {}x{} Maske {:.1}% belegt",
                    v.mask.w,
                    v.mask.h,
                    v.mask.coverage() * 100.0
                );
            }
            let (min, max) = parse_bounds(&a.bounds)?;

            // Voxel ausserhalb des Sichtkegels werden nie gecarvt -- sie
            // ueberleben jede Ansicht und ziehen die Huelle in eine Ecke.
            // Das ist die haeufigste Falle bei zu grossen Bounds.
            let mut vis_half = f32::MAX;
            for v in &views {
                let m = &v.w2c;
                // Kamera-Position = -R^T t
                let t = [m[0][3], m[1][3], m[2][3]];
                let eye = [
                    -(m[0][0] * t[0] + m[1][0] * t[1] + m[2][0] * t[2]),
                    -(m[0][1] * t[0] + m[1][1] * t[1] + m[2][1] * t[2]),
                    -(m[0][2] * t[0] + m[1][2] * t[1] + m[2][2] * t[2]),
                ];
                let dist = (eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]).sqrt();
                let tan_half = (v.mask.h as f32 * 0.5) / v.fy;
                vis_half = vis_half.min(dist * tan_half);
            }
            let box_half = (0..3)
                .map(|k| min[k].abs().max(max[k].abs()))
                .fold(0.0f32, f32::max);
            if box_half > vis_half * 0.95 {
                let rec = (vis_half * 0.9).max(1e-3);
                println!(
                    "[warn] Box-Halbweite {:.2} > sichtbare Halbweite {:.2}: Randbereiche \
                     werden nie gecarvt. Empfohlen: --bounds \"-{:.2},-{:.2},-{:.2}:{:.2},{:.2},{:.2}\"",
                    box_half, vis_half, rec, rec, rec, rec, rec, rec
                );
            }

            let t0 = Instant::now();
            let mut hull = carve(&views, min, max, a.res);
            let dt = t0.elapsed();

            println!(
                "[carve] {:?}: {}x{}x{} Voxel @ {:.4}, {:.2}% belegt, {:.0} ms",
                dt,
                hull.dims[0],
                hull.dims[1],
                hull.dims[2],
                hull.voxel,
                hull.occupancy() * 100.0,
                dt.as_secs_f64() * 1000.0
            );

            if hull.occupied() == 0 {
                bail!("Huelle ist leer -- Masken oder Posen passen nicht");
            }

            if a.margin > 0 {
                let t0 = Instant::now();
                hull = hull.dilate(a.margin);
                println!(
                    "[margin] +{} Voxel -> {:.2}% belegt ({:.0} ms)",
                    a.margin,
                    hull.occupancy() * 100.0,
                    t0.elapsed().as_secs_f64() * 1000.0
                );
            }

            if a.largest {
                let t0 = Instant::now();
                let (before, kept) = hull.keep_largest_component();
                println!(
                    "[largest] {before} -> {kept} Voxel, {:.2}% belegt ({:.0} ms)",
                    hull.occupancy() * 100.0,
                    t0.elapsed().as_secs_f64() * 1000.0
                );
            }

            if let Some(ply_path) = &a.ply {
                let p = Ply::open(ply_path)?;
                let (lo, hi) = p.robust_bounds(a.ply_pad, 1.0 - a.ply_pad);
                // zwei Voxel Luft, damit Splat-Radien nicht abgeschnitten werden
                let pad = 2.0 * hull.voxel;
                let lo = [lo[0] - pad, lo[1] - pad, lo[2] - pad];
                let hi = [hi[0] + pad, hi[1] + pad, hi[2] + pad];
                let before = hull.occupancy() * 100.0;
                hull.intersect_box(lo, hi);
                println!(
                    "[ply-bounds] {:.2}% -> {:.2}% belegt (Box [{:.2} {:.2} {:.2}]..[{:.2} {:.2} {:.2}])",
                    before,
                    hull.occupancy() * 100.0,
                    lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]
                );
            }

            hull.save(&a.out)?;
            println!("[out] {}", a.out.display());

            if let Some(obj) = &a.obj {
                let pts = hull.points();
                Ply::write_points_obj(&pts, obj)?;
                println!("[out] {} ({} Punkte)", obj.display(), pts.len());
            }
            if let Some((lo, hi)) = hull.bounds() {
                println!(
                    "[bounds] [{:.3} {:.3} {:.3}] .. [{:.3} {:.3} {:.3}]",
                    lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]
                );
            }
        }

        Cmd::Filter(a) => {
            let hull = Hull::load(&a.hull)?;
            let p = Ply::open(&a.ply)?;
            println!(
                "[ply] {} Splats, {} Properties",
                p.count,
                p.props.len()
            );
            let t0 = Instant::now();
            let inside = std::cell::Cell::new(0usize);
            let (before, after) = p.write_filtered(&a.out, |xyz| {
                let keep = hull.contains(xyz);
                if keep {
                    inside.set(inside.get() + 1);
                }
                keep
            })?;
            let dt = t0.elapsed();
            println!(
                "[filter] {} -> {} Splats ({:.1}% behalten), {:.0} ms",
                before,
                after,
                after as f64 / before as f64 * 100.0,
                dt.as_secs_f64() * 1000.0
            );
            println!("[out] {}", a.out.display());
        }

        Cmd::Info { hull } => {
            let h = Hull::load(&hull)?;
            println!("Datei      {}", hull.display());
            println!("Dims       {}x{}x{}", h.dims[0], h.dims[1], h.dims[2]);
            println!("Voxel      {:.5}", h.voxel);
            println!("Belegt     {:.2}%", h.occupancy() * 100.0);
            println!("Speicher   {:.1} MB", h.bits.len() as f64 * 8.0 / 1e6);
            if let Some((lo, hi)) = h.bounds() {
                println!(
                    "Bounds     [{:.3} {:.3} {:.3}] .. [{:.3} {:.3} {:.3}]",
                    lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]
                );
            }
        }
    }
    Ok(())
}
