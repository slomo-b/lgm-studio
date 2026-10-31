//! splatcage -- raeumliche Schranke fuer Gaussian Splats.
//!
//! Aus Silhouetten mehrerer Ansichten wird eine Voxelhuelle gecarvt. Alles,
//! was in einer Ansicht ausserhalb der Objektmaske liegt, ist beweisbar leer.
//! Splats, die ausserhalb der Huelle liegen, koennen damit entfernt werden,
//! bevor sie gerendert werden -- das killt "Bluten" in den Leerraum, schwebende
//! Artefakte und unscharfe Kanten.

pub mod carve;
pub mod hull;
pub mod ply;
pub mod views;
