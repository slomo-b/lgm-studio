//! Ansichten: Silhouettenmaske plus Kamera.

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

pub struct Mask {
    pub w: u32,
    pub h: u32,
    /// 1 = Objekt, 0 = Hintergrund
    pub on: Vec<u8>,
}

impl Mask {
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> bool {
        self.on[(y * self.w + x) as usize] != 0
    }

    pub fn load(path: &Path, mode: MaskMode, threshold: u8) -> Result<Self> {
        let file = std::fs::File::open(path).with_context(|| format!("{}", path.display()))?;
        let decoder = png::Decoder::new(file);
        let mut reader = decoder.read_info()?;
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf)?;
        let (w, h) = (info.width, info.height);
        let channels = match info.color_type {
            png::ColorType::Grayscale => 1,
            png::ColorType::GrayscaleAlpha => 2,
            png::ColorType::Rgb => 3,
            png::ColorType::Rgba => 4,
            other => bail!("nicht unterstuetzter PNG-Typ: {:?}", other),
        };
        let bits = info.bit_depth;
        if bits != png::BitDepth::Eight {
            bail!("nur 8-Bit-PNGs, gefunden {:?}", bits);
        }

        let mut on = vec![0u8; (w * h) as usize];
        let alpha_thr = threshold.max(1);
        for i in 0..(w * h) as usize {
            let px = &buf[i * channels..i * channels + channels];
            let v = match mode {
                MaskMode::Alpha => {
                    if channels == 4 || channels == 2 {
                        px[channels - 1] > alpha_thr
                    } else {
                        bail!("--mask-mode alpha, aber kein Alphakanal in {}", path.display());
                    }
                }
                MaskMode::Luma | MaskMode::Auto => {
                    let luma = if channels >= 3 {
                        (0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32) as u8
                    } else {
                        px[0]
                    };
                    // weisser Hintergrund = hell; Objekt = dunkler als die Schwelle
                    luma < threshold
                }
            };
            on[i] = v as u8;
        }

        if matches!(mode, MaskMode::Auto) && (channels == 4 || channels == 2) {
            // Alphakanal benutzen, wenn er wirklich Information traegt
            let mut used = 0usize;
            for i in 0..(w * h) as usize {
                if buf[i * channels + channels - 1] != 255 {
                    used += 1;
                }
            }
            if used > (w * h) as usize / 100 {
                return Self::load(path, MaskMode::Alpha, alpha_thr);
            }
        }

        Ok(Self { w, h, on })
    }

    pub fn coverage(&self) -> f64 {
        self.on.iter().filter(|v| **v != 0).count() as f64 / self.on.len() as f64
    }
}

#[derive(Clone, Copy, Debug, clap::ValueEnum, Default)]
pub enum MaskMode {
    /// Alpha nehmen, falls vorhanden, sonst Helligkeit
    #[default]
    Auto,
    /// Alphakanal
    Alpha,
    /// Helligkeit gegen weissen Hintergrund
    Luma,
}

#[derive(Deserialize)]
pub struct ViewJson {
    pub mask: PathBuf,
    pub width: u32,
    pub height: u32,
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    /// world -> camera (OpenCV: x rechts, y unten, z nach vorn)
    pub world_to_cam: [[f32; 4]; 4],
}

#[derive(Deserialize)]
pub struct ViewsFile {
    pub views: Vec<ViewJson>,
}

pub struct View {
    pub mask: Mask,
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub w2c: [[f32; 4]; 4],
}

impl View {
    /// LGM-Konvention: Y-up, Kamera auf einer Kugel um den Ursprung,
    /// Blickrichtung zum Zentrum, Ausrichtung wie OpenCV.
    #[allow(clippy::too_many_arguments)]
    pub fn orbit(
        mask: Mask,
        azim_deg: f32,
        elev_deg: f32,
        radius: f32,
        fovy_deg: f32,
    ) -> Self {
        let a = azim_deg.to_radians();
        let e = elev_deg.to_radians();
        let eye = [
            radius * e.cos() * a.sin(),
            radius * e.sin(),
            radius * e.cos() * a.cos(),
        ];
        let fwd = normalize([-eye[0], -eye[1], -eye[2]]);
        let right = normalize(cross(fwd, [0.0, 1.0, 0.0]));
        let down = cross(fwd, right);
        let r = [right, down, fwd];
        let t = [
            -(r[0][0] * eye[0] + r[0][1] * eye[1] + r[0][2] * eye[2]),
            -(r[1][0] * eye[0] + r[1][1] * eye[1] + r[1][2] * eye[2]),
            -(r[2][0] * eye[0] + r[2][1] * eye[1] + r[2][2] * eye[2]),
        ];
        let w2c = [
            [r[0][0], r[0][1], r[0][2], t[0]],
            [r[1][0], r[1][1], r[1][2], t[1]],
            [r[2][0], r[2][1], r[2][2], t[2]],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let f = (mask.w as f32 * 0.5) / (0.5 * fovy_deg.to_radians()).tan();
        let (cx, cy) = (mask.w as f32 * 0.5, mask.h as f32 * 0.5);
        Self {
            mask,
            fx: f,
            fy: f,
            cx,
            cy,
            w2c,
        }
    }

    pub fn load_json(path: &Path, mode: MaskMode, threshold: u8) -> Result<Vec<Self>> {
        let text = std::fs::read_to_string(path).with_context(|| format!("{}", path.display()))?;
        let file: ViewsFile = serde_json::from_str(&text)?;
        let base = path.parent().unwrap_or(Path::new("."));
        let mut out = Vec::with_capacity(file.views.len());
        for v in file.views {
            let mask_path = if v.mask.is_absolute() {
                v.mask.clone()
            } else {
                base.join(&v.mask)
            };
            let mask = Mask::load(&mask_path, mode, threshold)?;
            if mask.w != v.width || mask.h != v.height {
                bail!(
                    "{}: Maske ist {}x{}, JSON sagt {}x{}",
                    mask_path.display(),
                    mask.w,
                    mask.h,
                    v.width,
                    v.height
                );
            }
            out.push(Self {
                mask,
                fx: v.fx,
                fy: v.fy,
                cx: v.cx,
                cy: v.cy,
                w2c: v.world_to_cam,
            });
        }
        Ok(out)
    }

    /// Weltpunkt -> Pixel; `None`, wenn hinter der Kamera oder ausserhalb des Bildes.
    #[inline]
    pub fn project(&self, p: [f32; 3]) -> Option<(u32, u32)> {
        let m = &self.w2c;
        let x = m[0][0] * p[0] + m[0][1] * p[1] + m[0][2] * p[2] + m[0][3];
        let y = m[1][0] * p[0] + m[1][1] * p[1] + m[1][2] * p[2] + m[1][3];
        let z = m[2][0] * p[0] + m[2][1] * p[1] + m[2][2] * p[2] + m[2][3];
        if z <= 1e-4 {
            return None;
        }
        let u = self.fx * x / z + self.cx;
        let v = self.fy * y / z + self.cy;
        if u < 0.0 || v < 0.0 {
            return None;
        }
        let (ui, vi) = (u as u32, v as u32);
        if ui >= self.mask.w || vi >= self.mask.h {
            return None;
        }
        Some((ui, vi))
    }
}

#[inline]
pub fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub fn normalize(a: [f32; 3]) -> [f32; 3] {
    let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if n < 1e-12 {
        [0.0, 0.0, 1.0]
    } else {
        [a[0] / n, a[1] / n, a[2] / n]
    }
}
