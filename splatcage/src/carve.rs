//! Silhouetten-Carving: alles, was in irgendeiner Ansicht ausserhalb der
//! Objektmaske liegt, ist definitiv leer.

use crate::hull::Hull;
use crate::views::View;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};

/// Carvt die Voxelhuelle. `res` = Voxel pro laengster Achse.
///
/// Startet mit "alles belegt" und loescht. Ein Voxel, das in *einer* Ansicht
/// ausserhalb der Silhouette liegt, kann nicht zum Objekt gehoeren -- das ist
/// die einzige Regel, und sie ist beweisbar richtig. Der Preis: Konkavitaeten,
/// die von keiner Ansicht gesehen werden, fuellen sich (Visual Hull).
pub fn carve(views: &[View], min: [f32; 3], max: [f32; 3], res: u32) -> Hull {
    let extent = [
        (max[0] - min[0]).max(1e-6),
        (max[1] - min[1]).max(1e-6),
        (max[2] - min[2]).max(1e-6),
    ];
    let longest = extent[0].max(extent[1]).max(extent[2]);
    let voxel = longest / res as f32;
    let dims = [
        ((extent[0] / voxel).ceil() as u32).max(2),
        ((extent[1] / voxel).ceil() as u32).max(2),
        ((extent[2] / voxel).ceil() as u32).max(2),
    ];

    let hull = Hull::new(min, voxel, dims);
    let n_words = hull.bits.len();
    let bits: Vec<AtomicU64> = (0..n_words).map(|_| AtomicU64::new(u64::MAX)).collect();
    let (nx, ny, nz) = (dims[0], dims[1], dims[2]);

    for view in views {
        let before = count_bits(&bits);
        if before == 0 {
            break;
        }

        (0..nz).into_par_iter().for_each(|z| {
            let wz = min[2] + (z as f32 + 0.5) * voxel;
            for y in 0..ny {
                let wy = min[1] + (y as f32 + 0.5) * voxel;
                for x in 0..nx {
                    let wx = min[0] + (x as f32 + 0.5) * voxel;
                    let Some((u, v)) = view.project([wx, wy, wz]) else {
                        continue; // ausserhalb des Bildes: keine Aussage
                    };
                    if view.mask.get(u, v) {
                        continue;
                    }
                    // ausserhalb der Silhouette -> leer
                    let idx = x as u64 + nx as u64 * (y as u64 + ny as u64 * z as u64);
                    bits[(idx / 64) as usize].fetch_and(!(1u64 << (idx % 64)), Ordering::Relaxed);
                }
            }
        });
    }

    let bits: Vec<u64> = bits.into_iter().map(|a| a.into_inner()).collect();
    Hull {
        origin: min,
        voxel,
        dims,
        bits,
    }
}

fn count_bits(bits: &[AtomicU64]) -> u64 {
    bits.iter()
        .map(|w| w.load(Ordering::Relaxed).count_ones() as u64)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::views::{Mask, MaskMode, View};

    /// Kugel mit Radius r: die Huelle aus vielen Ansichten muss nahe an
    /// 4/3 pi r^3 liegen (Visual Hull ist immer etwas groesser).
    #[test]
    fn sphere_volume_is_close() {
        let size = 128u32;
        let r = 0.35f32;
        let (min, max) = ([-0.6; 3], [0.6; 3]);
        let fovy = 49.1f32;

        let mut views = Vec::new();
        for i in 0..16 {
            let mut m = Mask {
                w: size,
                h: size,
                on: vec![0; (size * size) as usize],
            };
            let azim = i as f32 * 22.5;
            let elev = if i % 2 == 0 { 5.0 } else { 25.0 };
            let v = View::orbit(
                Mask {
                    w: size,
                    h: size,
                    on: vec![0; (size * size) as usize],
                },
                azim,
                elev,
                1.6,
                fovy,
            );
            // Kugel-Silhouette analytisch: projizierter Mittelpunkt plus
            // scheinbarer Winkelradius asin(r/d), also f * r / sqrt(d^2 - r^2).
            let c = v.project([0.0, 0.0, 0.0]).unwrap();
            let d = 1.6f32;
            let rad = v.fx * r / (d * d - r * r).sqrt();
            for y in 0..size {
                for x in 0..size {
                    let dx = x as f32 - c.0 as f32;
                    let dy = y as f32 - c.1 as f32;
                    if dx * dx + dy * dy <= rad * rad {
                        m.on[(y * size + x) as usize] = 1;
                    }
                }
            }
            views.push(View {
                mask: m,
                ..v
            });
        }

        let hull = carve(&views, min, max, 96);
        let vol = hull.occupied() as f64 * (hull.voxel as f64).powi(3);
        let expect = 4.0 / 3.0 * std::f64::consts::PI * (r as f64).powi(3);
        let ratio = vol / expect;
        println!("vol={vol:.4} expect={expect:.4} ratio={ratio:.3}");
        assert!(
            ratio > 0.96,
            "Huelle zu klein, Carving schneidet ins Objekt: {ratio:.3}"
        );
        assert!(ratio < 1.2, "Huelle viel zu gross: {ratio:.3}");
    }

    #[test]
    fn outside_mask_is_empty() {
        // Ein einziger Blick von +Z mit einer Maske, die nur die linke Haelfte
        // abdeckt: die rechte Haelfte des Raums muss leer sein.
        let size = 32u32;
        let mut on = vec![0u8; (size * size) as usize];
        for y in 0..size {
            for x in 0..size / 2 {
                on[(y * size + x) as usize] = 1;
            }
        }
        let view = View::orbit(
            Mask { w: size, h: size, on },
            0.0,
            0.0,
            2.0,
            49.1,
        );
        let hull = carve(&[view], [-1.0; 3], [1.0; 3], 64);
        assert!(!hull.contains([0.4, 0.0, 0.0]), "rechts muss leer sein");
        assert!(hull.contains([-0.3, 0.0, 0.0]), "links muss belegt sein");
    }
}
