"""Stufe A der SimReady-Pipeline: Splat -> OpenUSD.

Reihenfolge:

    splat.py    3DGS-.ply -> Punktwolke mit Normalen (aus der Gaussian-Kovarianz)
    mesh.py     Poisson-Rekonstruktion + Reparatur
    metric.py   Meter, Pivot, Z-up (LGM liefert Y-up und ohne Einheit)
    collider.py konvexe Zerlegung (CoACD) mit Fallback
    usd_export.py  USD-Authoring inkl. Physik und Semantik
    lint.py     lokaler Spiegel der pruefbaren SimReady-Regeln

Aufruf:

    python -m simready build asset.ply -o out/ --class mug --target-size 0.09
"""

__all__ = ["splat", "mesh", "metric", "collider", "usd_export", "lint"]
