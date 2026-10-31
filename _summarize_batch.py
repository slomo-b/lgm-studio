"""Sammelt die build_report.json aus einem Batch-Ordner zu einer Tabelle."""
import glob
import json
import sys

pattern = sys.argv[1] if len(sys.argv) > 1 else "out/batch/*/build_report.json"
rows = []
for p in sorted(glob.glob(pattern)):
    with open(p, encoding="utf-8") as f:
        r = json.load(f)
    m = r["steps"]["mesh"]
    lint = r["lint"]
    ext = r["steps"]["metric"]["extents_m"]
    rows.append(
        {
            "asset": r["asset"][:26],
            "faces": m["faces"],
            "watertight": m.get("watertight"),
            "h_m": max(ext) if ext else 0.0,
            "pass": lint["n_pass"],
            "fail": lint["n_fail"],
            "sec": r["seconds"],
        }
    )

hdr = f"{'asset':28}{'faces':>7}{'watertight':>12}{'hoehe m':>9}{'lint':>7}{'s':>7}"
print(hdr)
print("-" * len(hdr))
for r in rows:
    print(
        f"{r['asset']:28}{r['faces']:>7}{str(r['watertight']):>12}"
        f"{r['h_m']:>9.3f}{str(r['pass']) + '/' + str(r['pass'] + r['fail']):>7}{r['sec']:>7.1f}"
    )
print()
print(f"{len(rows)} Assets | alle wasserdicht: {all(r['watertight'] for r in rows)}"
      f" | Lint komplett: {all(r['fail'] == 0 for r in rows)}"
      f" | Mittel: {sum(r['sec'] for r in rows) / max(len(rows), 1):.1f} s")
