"""Render the model-size score curve as a Devpost gallery image.

Emits SVG and rasterises with ImageMagick. matplotlib was the obvious route but
would not install here, and hand-written SVG gives exact control over the
bulletin styling anyway, with no dependency beyond `convert`.

The story: the 50 points that are not human-judged split into a throughput term
(30%) and a memory term (20%). Stacking them shows the total AND why the 3B
loses, which is the point: it fails on both terms at once, not just one.

Scored under the published formula, where S_perf is relative to the fastest
candidate. An earlier render used the capped misreading of the provisional
15 tok/s reference and carried a subtitle asserting it, along with a third set of
candidate figures older than either table in the report. curve.jsonl was missing
from the tree, so the data is rebuilt here from the corrected selection figures.

Form: horizontal stacked bar, sorted by total. The job is magnitude comparison
across a small set of named entities with a two-part composition.

Colour: validated categorical slots 1 and 2 (blue, orange), not AGBE's green and
ochre. Green against orange collapses under protanopia (measured adjacent CVD
dE 7.7, inside the warn band); blue against orange measures 24.7, well clear.
Validated with the dataviz validator against this exact cream surface. Every
segment is direct-labelled, which supplies the relief the orange's sub-3:1
contrast on cream requires.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "figures"
OUT.mkdir(exist_ok=True)

W, H = 1500, 1000                 # 3:2, Devpost's recommended ratio

SURFACE = "#F7F2E7"
INK = "#22271F"
INK_SOFT = "#5A6154"
MUTED = "#8A8877"
GRID = "#E2DAC6"
PERF = "#2a78d6"                  # validated slot 1
EFF = "#eb6834"                   # validated slot 2

SANS = "DejaVu Sans"

LABELS = {
    "qwen2.5-0.5b-q4km.gguf": "Qwen2.5 0.5B",
    "gemma3-1b-q4km.gguf": "Gemma 3 1B",
    "llama3.2-1b-q4km.gguf": "Llama 3.2 1B",
    "qwen2.5-1.5b-q4km.gguf": "Qwen2.5 1.5B",
    "qwen2.5-3b-q4km.gguf": "Qwen2.5 3B",
}
SELECTED = "gemma3-1b-q4km.gguf"

# plot geometry
PLOT_L, PLOT_R = 300, 1180
PLOT_T, PLOT_B = 300, 800
X_MAX = 52.0


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, s, size=14, fill=INK, anchor="start", weight="normal", family=SANS):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>')


def load():
    rows = [json.loads(l) for l in (HERE / "curve.jsonl").read_text().splitlines() if l.strip()]
    for r in rows:
        # Points contributed by each weighted term, before the thermal penalty,
        # which applied equally to every run and so cancels in comparison.
        r["perf_pts"] = 0.30 * r["S_perf"]
        r["eff_pts"] = 0.20 * r["S_eff"]
        r["total"] = r["perf_pts"] + r["eff_pts"]
        r["label"] = LABELS.get(r["model"], r["model"])
    rows.sort(key=lambda r: -r["total"])
    return rows


def main() -> None:
    rows = load()
    n = len(rows)
    row_h = (PLOT_B - PLOT_T) / n
    bar_h = 46
    gap_px = 4                    # 2px-equivalent surface gap each side of the join

    def sx(v):                    # value -> x pixels
        return PLOT_L + (v / X_MAX) * (PLOT_R - PLOT_L)

    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}">',
         f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>']

    # titles
    p.append(txt(70, 108, "The biggest model that fits is the wrong answer",
                 size=40, weight="bold"))
    p.append(txt(70, 152, "Throughput is scored against the fastest submission, so speed and memory compound.",
                 size=19, fill=INK_SOFT))
    p.append(txt(70, 180, "Memory is paid for linearly. Both terms punish size.",
                 size=19, fill=INK_SOFT))

    # gridlines + x ticks
    for v in (0, 10, 20, 30, 40, 50):
        x = sx(v)
        p.append(f'<line x1="{x:.1f}" y1="{PLOT_T-30}" x2="{x:.1f}" y2="{PLOT_B+6}" '
                 f'stroke="{GRID}" stroke-width="1"/>')
        p.append(txt(x, PLOT_B + 34, str(v), size=15, fill=MUTED, anchor="middle"))
    p.append(txt((PLOT_L + PLOT_R) / 2, PLOT_B + 72,
                 "Points banked of the 50 available before human accuracy judging",
                 size=16, fill=INK_SOFT, anchor="middle"))

    for i, r in enumerate(rows):
        cy = PLOT_T + row_h * i + row_h / 2
        y = cy - bar_h / 2
        sel = r["model"] == SELECTED

        # row label
        p.append(txt(PLOT_L - 22, cy + 6, r["label"], size=19,
                     fill=INK, anchor="end", weight="bold" if sel else "normal"))
        if sel:
            p.append(txt(PLOT_L - 22, cy + 27, "selected", size=12,
                         fill="#2E4633", anchor="end", weight="bold"))

        w1 = sx(r["perf_pts"]) - PLOT_L
        x2 = sx(r["perf_pts"]) + gap_px
        w2 = (sx(r["total"]) - PLOT_L) - w1 - gap_px

        p.append(f'<rect x="{PLOT_L}" y="{y}" width="{w1:.1f}" height="{bar_h}" fill="{PERF}"/>')
        p.append(f'<rect x="{x2:.1f}" y="{y}" width="{max(w2,0):.1f}" height="{bar_h}" '
                 f'fill="{EFF}" rx="4" ry="4"/>')

        # direct labels inside each segment
        p.append(txt(PLOT_L + w1 / 2, cy + 7, f'{r["perf_pts"]:.1f}',
                     size=17, fill="#FFFFFF", anchor="middle", weight="bold"))
        p.append(txt(x2 + max(w2, 0) / 2, cy + 7, f'{r["eff_pts"]:.1f}',
                     size=17, fill="#FFFFFF", anchor="middle", weight="bold"))

        # total, then the raw measurements that produced it
        tx = sx(r["total"]) + 18
        p.append(txt(tx, cy + 8, f'{r["total"]:.1f}', size=22, fill=INK, weight="bold"))
        p.append(txt(tx + 62, cy + 7, f'{r["tps"]:.0f} tok/s   {r["peak_rss_gb"]:.2f} GB',
                     size=14, fill=MUTED))

    # legend
    ly = 918
    p.append(f'<rect x="70" y="{ly-13}" width="16" height="16" fill="{PERF}"/>')
    p.append(txt(96, ly, "Throughput   0.30 x S_perf", size=16, fill=INK_SOFT))
    p.append(f'<rect x="360" y="{ly-13}" width="16" height="16" fill="{EFF}" rx="3"/>')
    p.append(txt(386, ly, "Memory   0.20 x S_eff", size=16, fill=INK_SOFT))

    p.append(txt(70, 958,
                 "Measured on an i7-10850H held to the ADTC Standard Laptop profile: "
                 "4 threads, no GPU offload, 7 GB cap.", size=13.5, fill=MUTED))
    p.append(txt(70, 980,
                 "The 10-point thermal penalty applied equally to all five runs and is "
                 "excluded here. AGBE / Africa Deep Tech Challenge 2026.",
                 size=13.5, fill=MUTED))

    p.append("</svg>")

    svg_path = OUT / "score-curve.svg"
    png_path = OUT / "score-curve.png"
    svg_path.write_text("\n".join(p))
    subprocess.run(["convert", "-density", "144", "-background", SURFACE,
                    str(svg_path), "-quality", "92", str(png_path)], check=True)

    print(f"wrote {png_path}  ({png_path.stat().st_size/1024:.0f} KB)")
    for r in rows:
        print(f"  {r['label']:<15}{r['perf_pts']:6.1f} + {r['eff_pts']:5.1f} = {r['total']:5.1f}")


if __name__ == "__main__":
    main()
