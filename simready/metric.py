"""Metrik, Pivot und Achskonvention.

Das ist der Teil, den LGM strukturell nicht liefern kann (ANALYSE_SIMREADY.md,
Abschnitt 3): die Ausgabe liegt in [-1,1]^3, ohne Einheit, ohne Pivot, in Y-up.
SimReady verlangt `metersPerUnit = 1.0` und `upAxis = Z` (Regel UN.006).

Wichtig, ehrlich benannt: die Groesse ist eine **Vorhersage aus einem Prior**,
keine Messung. Ohne Referenzobjekt im Bild gibt es keine Skala.
"""

import numpy as np

# Grobe reale Abmessungen (laengste Kante, Meter). Nur Startwerte -- fuer
# Produktion gehoert hier ein Klassifikator plus Datenblatt hin.
SIZE_PRIORS = {
    "mug": 0.09,
    "cup": 0.09,
    "bottle": 0.25,
    "can": 0.12,
    "bowl": 0.18,
    "plate": 0.26,
    "box": 0.30,
    "crate": 0.50,
    "chair": 0.90,
    "stool": 0.45,
    "table": 1.20,
    "shelf": 1.80,
    "lamp": 0.40,
    "monitor": 0.55,
    "keyboard": 0.45,
    "laptop": 0.35,
    "phone": 0.16,
    "shoe": 0.30,
    "helmet": 0.28,
    "tool": 0.25,
    "valve": 0.15,
    "pump": 0.40,
    "motor": 0.35,
    "pipe": 0.60,
    "toy": 0.15,
}

# grobe Fuellkoerper-Dichten in kg/m^3 fuer die Masenschaetzung
DENSITY_PRIORS = {
    "generic": 500.0,
    "wood": 600.0,
    "metal": 7800.0,
    "plastic": 950.0,
    "ceramic": 2300.0,
    "glass": 2500.0,
    "fabric": 200.0,
}


def guess_size(obj_class, target_size=None):
    if target_size is not None:
        return float(target_size), "explizit"
    key = (obj_class or "").lower().strip()
    if key in SIZE_PRIORS:
        return SIZE_PRIORS[key], f"Prior '{key}'"
    return 0.30, "Fallback 0.30 m (Klasse unbekannt)"


def yup_to_zup(tm):
    """LGM-Welt ist Y-up; USD/SimReady will Z-up. +90 Grad um X bildet +Y auf +Z ab."""
    tm = tm.copy()
    R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    tm.apply_transform(np.block([[R, np.zeros((3, 1))], [np.zeros((1, 3)), 1.0]]))
    return tm


def to_meters(tm, target_size):
    """Laengste Kante auf `target_size` Meter skalieren."""
    ext = float(np.max(tm.extents))
    if ext <= 1e-9:
        raise ValueError("Mesh hat keine Ausdehnung")
    s = float(target_size) / ext
    tm = tm.copy()
    tm.apply_scale(s)
    return tm, s


def pivot_base_center(tm):
    """Pivot in die Mitte der Standflaeche, Unterkante auf z = 0.

    Konvention fuer Props: der Ursprung ist der Punkt, an dem das Asset steht.
    Damit laesst sich ein Asset per Translation platzieren, ohne die
    Bodenhoehe nachzurechnen.
    """
    tm = tm.copy()
    lo, hi = tm.bounds
    shift = np.array([-(lo[0] + hi[0]) / 2.0, -(lo[1] + hi[1]) / 2.0, -lo[2]])
    tm.apply_translation(shift)
    return tm


def normalize(tm, obj_class=None, target_size=None, up_axis="Z", base=True):
    """Kompletter Metrikschritt: Y-up->Z-up, Meter, Pivot."""
    info = {}
    if up_axis.upper() == "Z":
        tm = yup_to_zup(tm)
    size, source = guess_size(obj_class, target_size)
    tm, s = to_meters(tm, size)
    if base:
        tm = pivot_base_center(tm)

    info["target_size_m"] = size
    info["size_source"] = source
    info["scale_factor"] = s
    info["extents_m"] = [float(x) for x in tm.extents]
    return tm, info
