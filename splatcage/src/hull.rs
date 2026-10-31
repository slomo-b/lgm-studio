//! Die Voxelhuelle: bit-gepackte Belegtheit mit Punktabfrage.

use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"SPLCAGE1";

/// Belegtes Voxelgitter. Bit-Index = x + nx*(y + ny*z), gepackt in u64-Worte.
///
/// Bit-gepackt statt `Vec<bool>`: 256^3 sind 2 MB statt 16 MB, und die
/// Punktabfrage bleibt ein einziges Wort-AND. Das ist wichtig, weil die
/// Schranke pro Splat einmal abgefragt wird -- bei 60k Splats 60k Mal.
pub struct Hull {
    pub origin: [f32; 3],
    pub voxel: f32,
    pub dims: [u32; 3],
    pub bits: Vec<u64>,
}

impl Hull {
    pub fn new(origin: [f32; 3], voxel: f32, dims: [u32; 3]) -> Self {
        let n = Self::word_count(dims);
        Self {
            origin,
            voxel,
            dims,
            bits: vec![u64::MAX; n], // alles belegt, Carving loescht
        }
    }

    pub fn word_count(dims: [u32; 3]) -> usize {
        let total = dims[0] as usize * dims[1] as usize * dims[2] as usize;
        (total + 63) / 64
    }

    pub fn voxel_count(&self) -> u64 {
        self.dims[0] as u64 * self.dims[1] as u64 * self.dims[2] as u64
    }

    #[inline]
    pub fn index(&self, x: u32, y: u32, z: u32) -> u64 {
        x as u64 + self.dims[0] as u64 * (y as u64 + self.dims[1] as u64 * z as u64)
    }

    #[inline]
    pub fn get(&self, x: u32, y: u32, z: u32) -> bool {
        if x >= self.dims[0] || y >= self.dims[1] || z >= self.dims[2] {
            return false;
        }
        let i = self.index(x, y, z) as usize;
        self.bits[i / 64] & (1u64 << (i % 64)) != 0
    }

    /// Punktabfrage in Weltkoordinaten. Ausserhalb der Box gilt "leer".
    #[inline]
    pub fn contains(&self, p: [f32; 3]) -> bool {
        let mut v = [0u32; 3];
        for k in 0..3 {
            let f = (p[k] - self.origin[k]) / self.voxel;
            if f < 0.0 || f >= self.dims[k] as f32 {
                return false;
            }
            v[k] = f as u32;
        }
        self.get(v[0], v[1], v[2])
    }

    pub fn occupied(&self) -> u64 {
        self.bits.iter().map(|w| w.count_ones() as u64).sum()
    }

    pub fn occupancy(&self) -> f64 {
        self.occupied() as f64 / self.voxel_count() as f64
    }

    /// Bounding-Box der belegten Voxel in Weltkoordinaten.
    pub fn bounds(&self) -> Option<([f32; 3], [f32; 3])> {
        let (mut lo, mut hi) = ([u32::MAX; 3], [0u32; 3]);
        let mut any = false;
        for z in 0..self.dims[2] {
            for y in 0..self.dims[1] {
                for x in 0..self.dims[0] {
                    if self.get(x, y, z) {
                        any = true;
                        for (k, v) in [x, y, z].iter().enumerate() {
                            lo[k] = lo[k].min(*v);
                            hi[k] = hi[k].max(*v);
                        }
                    }
                }
            }
        }
        if !any {
            return None;
        }
        let o = self.origin;
        let v = self.voxel;
        Some((
            [
                o[0] + lo[0] as f32 * v,
                o[1] + lo[1] as f32 * v,
                o[2] + lo[2] as f32 * v,
            ],
            [
                o[0] + (hi[0] + 1) as f32 * v,
                o[1] + (hi[1] + 1) as f32 * v,
                o[2] + (hi[2] + 1) as f32 * v,
            ],
        ))
    }

    /// Nur die groesste zusammenhaengende Komponente behalten.
    ///
    /// Visual Hulls bringen Inseln mit: Regionen, die zufaellig in allen
    /// Silhouetten liegen, aber nicht zum Objekt gehoeren. Sie blaehen Volumen
    /// und Bounding-Box auf und liefern Muell als 3DGS-Initialisierung.
    pub fn keep_largest_component(&mut self) -> (u64, usize) {
        let n = self.voxel_count() as usize;
        let mut seen = vec![0u8; n];
        let mut best: Vec<u32> = Vec::new();
        let (nx, ny, nz) = (self.dims[0] as i64, self.dims[1] as i64, self.dims[2] as i64);
        let mut stack: Vec<u32> = Vec::new();
        let mut comp: Vec<u32> = Vec::new();

        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    if !self.get(x as u32, y as u32, z as u32) {
                        continue;
                    }
                    let start = self.index(x as u32, y as u32, z as u32);
                    if seen[start as usize] != 0 {
                        continue;
                    }
                    comp.clear();
                    stack.push(start as u32);
                    seen[start as usize] = 1;
                    while let Some(cur) = stack.pop() {
                        comp.push(cur);
                        let i = cur as i64;
                        let cz = i / (nx * ny);
                        let rem = i % (nx * ny);
                        let cy = rem / nx;
                        let cx = rem % nx;
                        for (dx, dy, dz) in [
                            (1i64, 0, 0),
                            (-1, 0, 0),
                            (0, 1, 0),
                            (0, -1, 0),
                            (0, 0, 1),
                            (0, 0, -1),
                        ] {
                            let (ax, ay, az) = (cx + dx, cy + dy, cz + dz);
                            if ax < 0 || ay < 0 || az < 0 || ax >= nx || ay >= ny || az >= nz {
                                continue;
                            }
                            let ni = self.index(ax as u32, ay as u32, az as u32);
                            if seen[ni as usize] != 0 || !self.get(ax as u32, ay as u32, az as u32) {
                                continue;
                            }
                            seen[ni as usize] = 1;
                            stack.push(ni as u32);
                        }
                    }
                    if comp.len() > best.len() {
                        best = comp.clone();
                    }
                }
            }
        }

        let before = self.occupied();
        self.bits.iter_mut().for_each(|w| *w = 0);
        for i in &best {
            let i = *i as usize;
            self.bits[i / 64] |= 1u64 << (i % 64);
        }
        (before, best.len())
    }

    /// Alles ausserhalb eines Weltkoordinaten-Kastens leeren.
    pub fn intersect_box(&mut self, min: [f32; 3], max: [f32; 3]) -> u64 {
        let mut cleared = 0u64;
        for z in 0..self.dims[2] {
            let wz = self.origin[2] + (z as f32 + 0.5) * self.voxel;
            for y in 0..self.dims[1] {
                let wy = self.origin[1] + (y as f32 + 0.5) * self.voxel;
                for x in 0..self.dims[0] {
                    let wx = self.origin[0] + (x as f32 + 0.5) * self.voxel;
                    if wx < min[0] || wy < min[1] || wz < min[2] || wx > max[0] || wy > max[1] || wz > max[2] {
                        let i = self.index(x, y, z) as usize;
                        let mask = 1u64 << (i % 64);
                        if self.bits[i / 64] & mask != 0 {
                            self.bits[i / 64] &= !mask;
                            cleared += 1;
                        }
                    }
                }
            }
        }
        cleared
    }

    /// Huelle um `n` Voxel aufblasen.
    ///
    /// Splats sind Ellipsoide: ihr Mittelpunkt darf knapp ausserhalb der
    /// Silhouettenhuelle liegen, waehrend die Flaeche selbst noch innerhalb
    /// ist. Ohne diese Toleranz frisst der Filter die Randschicht weg.
    pub fn dilate(&self, n: u32) -> Hull {
        if n == 0 {
            return Hull {
                origin: self.origin,
                voxel: self.voxel,
                dims: self.dims,
                bits: self.bits.clone(),
            };
        }
        let mut cur = Hull {
            origin: self.origin,
            voxel: self.voxel,
            dims: self.dims,
            bits: self.bits.clone(),
        };
        for _ in 0..n {
            let mut next = cur.bits.clone();
            let d = self.dims;
            for z in 0..d[2] {
                for y in 0..d[1] {
                    for x in 0..d[0] {
                        if cur.get(x, y, z) {
                            continue;
                        }
                        let mut hit = false;
                        'n: for dz in -1i32..=1 {
                            for dy in -1i32..=1 {
                                for dx in -1i32..=1 {
                                    let (nx, ny, nz) =
                                        (x as i32 + dx, y as i32 + dy, z as i32 + dz);
                                    if nx < 0
                                        || ny < 0
                                        || nz < 0
                                        || nx >= d[0] as i32
                                        || ny >= d[1] as i32
                                        || nz >= d[2] as i32
                                    {
                                        continue;
                                    }
                                    if cur.get(nx as u32, ny as u32, nz as u32) {
                                        hit = true;
                                        break 'n;
                                    }
                                }
                            }
                        }
                        if hit {
                            let i = self.index(x, y, z) as usize;
                            next[i / 64] |= 1u64 << (i % 64);
                        }
                    }
                }
            }
            cur.bits = next;
        }
        cur
    }

    /// Belegte Voxel als Punktwolke (Voxelzentren) -- fuer Sichtpruefung und OBJ-Export.
    pub fn points(&self) -> Vec<[f32; 3]> {
        let mut out = Vec::with_capacity(self.occupied() as usize);
        for z in 0..self.dims[2] {
            for y in 0..self.dims[1] {
                for x in 0..self.dims[0] {
                    if self.get(x, y, z) {
                        out.push([
                            self.origin[0] + (x as f32 + 0.5) * self.voxel,
                            self.origin[1] + (y as f32 + 0.5) * self.voxel,
                            self.origin[2] + (z as f32 + 0.5) * self.voxel,
                        ]);
                    }
                }
            }
        }
        out
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let f = File::create(path).with_context(|| format!("{}", path.display()))?;
        let mut w = BufWriter::new(f);
        w.write_all(MAGIC)?;
        for v in self.origin.iter().chain(std::iter::once(&self.voxel)) {
            w.write_all(&v.to_le_bytes())?;
        }
        for d in self.dims {
            w.write_all(&d.to_le_bytes())?;
        }
        for word in &self.bits {
            w.write_all(&word.to_le_bytes())?;
        }
        w.flush()?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let f = File::open(path).with_context(|| format!("{}", path.display()))?;
        let mut r = BufReader::new(f);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            bail!("keine splatcage-Datei: {}", path.display());
        }
        let mut f32buf = [0u8; 4];
        let mut origin = [0f32; 3];
        for o in origin.iter_mut() {
            r.read_exact(&mut f32buf)?;
            *o = f32::from_le_bytes(f32buf);
        }
        r.read_exact(&mut f32buf)?;
        let voxel = f32::from_le_bytes(f32buf);
        let mut dims = [0u32; 3];
        let mut u32buf = [0u8; 4];
        for d in dims.iter_mut() {
            r.read_exact(&mut u32buf)?;
            *d = u32::from_le_bytes(u32buf);
        }
        let n = Self::word_count(dims);
        let mut bits = vec![0u64; n];
        for w in bits.iter_mut() {
            r.read_exact(&mut u32buf)?;
            let lo = u32::from_le_bytes(u32buf) as u64;
            r.read_exact(&mut u32buf)?;
            let hi = u32::from_le_bytes(u32buf) as u64;
            *w = lo | (hi << 32);
        }
        Ok(Self {
            origin,
            voxel,
            dims,
            bits,
        })
    }
}
