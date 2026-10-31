//! Minimaler, generischer PLY-Leser/Schreiber.
//!
//! 3DGS-Splats haben 62 Float-Properties (x..z, Normale, f_dc, f_rest, opacity,
//! scale, rot). Wir wollen nur x,y,z lesen und *alle* Properties unveraendert
//! durchreichen -- deshalb generisch statt fest verdrahtet.

use anyhow::{bail, Context, Result};
use std::fs;
use std::io::Write;
use std::path::Path;

pub struct Ply {
    pub header: String,
    pub count: usize,
    pub props: Vec<String>,
    pub body: Vec<u8>,
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl Ply {
    pub fn open(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("{}", path.display()))?;
        let marker = b"end_header\n";
        let pos = bytes
            .windows(marker.len())
            .position(|w| w == marker)
            .ok_or_else(|| anyhow::anyhow!("kein PLY-Header in {}", path.display()))?;
        let header = String::from_utf8_lossy(&bytes[..pos + marker.len()]).to_string();

        if !header.contains("format binary_little_endian") {
            bail!("nur binary_little_endian-PLY wird unterstuetzt");
        }

        let mut count = 0usize;
        let mut props = Vec::new();
        let mut element = String::new();
        let mut elements_seen = Vec::new();
        for line in header.lines() {
            let line = line.trim();
            if let Some(rest) = line.strip_prefix("element ") {
                let mut it = rest.split_whitespace();
                element = it.next().unwrap_or("").to_string();
                elements_seen.push(element.clone());
                if element == "vertex" {
                    count = it.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                }
            } else if let Some(rest) = line.strip_prefix("property ") {
                if element != "vertex" {
                    continue;
                }
                let mut it = rest.split_whitespace();
                let ty = it.next().unwrap_or("");
                let name = it.next().unwrap_or("");
                if ty != "float" && ty != "float32" {
                    bail!("Property '{}' hat Typ '{}' -- nur float32 wird unterstuetzt", name, ty);
                }
                props.push(name.to_string());
            }
        }

        if elements_seen.len() > 1 {
            bail!(
                "PLY mit mehreren Elementen ({:?}) -- bitte vorher reduzieren",
                elements_seen
            );
        }
        if count == 0 || props.is_empty() {
            bail!("PLY ohne Vertex-Element");
        }

        let idx = |n: &str| -> Result<usize> {
            props
                .iter()
                .position(|p| p == n)
                .ok_or_else(|| anyhow::anyhow!("Property '{}' fehlt", n))
        };
        let (x, y, z) = (idx("x")?, idx("y")?, idx("z")?);

        let stride = props.len() * 4;
        let body = bytes[pos + marker.len()..].to_vec();
        if body.len() < count * stride {
            bail!(
                "PLY-Body zu kurz: {} Bytes, erwartet {}",
                body.len(),
                count * stride
            );
        }

        Ok(Self {
            header,
            count,
            props,
            body,
            x,
            y,
            z,
        })
    }

    #[inline]
    pub fn xyz(&self, row: usize) -> [f32; 3] {
        let stride = self.props.len() * 4;
        let base = row * stride;
        let rd = |i: usize| {
            let o = base + i * 4;
            f32::from_le_bytes([
                self.body[o],
                self.body[o + 1],
                self.body[o + 2],
                self.body[o + 3],
            ])
        };
        [rd(self.x), rd(self.y), rd(self.z)]
    }

    /// Robuste Bounding-Box aus einer PLY-Punktwolke.
    ///
    /// Perzentile statt Min/Max: einzelne fliegende Splats wuerden die Box sonst
    /// aufspannen und die Schranke wieder wirkungslos machen.
    pub fn robust_bounds(&self, lo_q: f32, hi_q: f32) -> ([f32; 3], [f32; 3]) {
        let mut axes: [Vec<f32>; 3] = [Vec::with_capacity(self.count), Vec::with_capacity(self.count), Vec::with_capacity(self.count)];
        for r in 0..self.count {
            let p = self.xyz(r);
            for k in 0..3 {
                axes[k].push(p[k]);
            }
        }
        let mut lo = [0f32; 3];
        let mut hi = [0f32; 3];
        for k in 0..3 {
            axes[k].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = axes[k].len();
            lo[k] = axes[k][((n as f32 * lo_q) as usize).min(n - 1)];
            hi[k] = axes[k][((n as f32 * hi_q) as usize).min(n - 1)];
        }
        (lo, hi)
    }

    /// Schreibt nur die Zeilen, fuer die `keep` wahr ist.
    ///
    /// `move_to` darf den Punkt verschieben (Zentrum in die Huelle ziehen);
    /// die Properties x,y,z werden dann neu geschrieben.
    pub fn write_filtered<F>(
        &self,
        dst: &Path,
        keep: F,
    ) -> Result<(usize, usize)>
    where
        F: Fn([f32; 3]) -> bool,
    {
        let stride = self.props.len() * 4;
        let kept: Vec<usize> = (0..self.count).filter(|r| keep(self.xyz(*r))).collect();

        let header = replace_vertex_count(&self.header, kept.len());
        let mut out = fs::File::create(dst).with_context(|| format!("{}", dst.display()))?;
        out.write_all(header.as_bytes())?;
        let mut row = vec![0u8; stride];
        for r in &kept {
            row.copy_from_slice(&self.body[r * stride..(r + 1) * stride]);
            out.write_all(&row)?;
        }
        out.flush()?;
        Ok((self.count, kept.len()))
    }

    /// Schreibt eine Punktwolke als ASCII-OBJ (nur Punkte, fuer die Sichtpruefung).
    pub fn write_points_obj(points: &[[f32; 3]], dst: &Path) -> Result<()> {
        let mut s = String::with_capacity(points.len() * 32);
        for p in points {
            s.push_str(&format!("v {:.5} {:.5} {:.5}\n", p[0], p[1], p[2]));
        }
        fs::write(dst, s).with_context(|| format!("{}", dst.display()))?;
        Ok(())
    }
}

fn replace_vertex_count(header: &str, n: usize) -> String {
    let mut out = String::with_capacity(header.len());
    for line in header.lines() {
        if line.trim_start().starts_with("element vertex") {
            out.push_str(&format!("element vertex {}\n", n));
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}
