"""Render the AGBE demo video: 1920x1080, silent, for a voiceover to be laid over.

The terminal scenes are REAL. `record_session.py` runs the model in a pseudo
terminal and captures every byte with a timestamp; this replays that at the
timing it actually happened. The pause before the first token and the pace of
generation are the model's own, not an animation. A judge should see the speed
they will really get.

The displayed command shortens the absolute path to `llama-cli`, which is what
you would type with it on PATH. Everything else, including the output, is
verbatim.

Frames are rendered with Pillow and assembled with ffmpeg at 30fps.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from render_terminal import Session, draw_frame  # noqa: E402

W, H = 1920, 1080
FPS = 30
OUT = pathlib.Path(__file__).resolve().parent
FRAMES = OUT / "frames"
CAST = OUT / "cast"

PAPER = (247, 242, 231)
INK = (34, 39, 31)
INK_SOFT = (90, 97, 84)
GREEN = (46, 70, 51)
OCHRE = (169, 102, 42)

FD = "/usr/share/fonts/truetype/dejavu"
PATHS = {
    "serif": f"{FD}/DejaVuSerif.ttf",
    "serif_bold": f"{FD}/DejaVuSerif-Bold.ttf",
    "sans": f"{FD}/DejaVuSans.ttf",
    "sans_bold": f"{FD}/DejaVuSans-Bold.ttf",
}


def font(name: str, size: int):
    return ImageFont.truetype(PATHS[name], size)


def wrap(draw, text: str, fnt, max_w: int) -> list[str]:
    out = []
    for para in text.split("\n"):
        if not para.strip():
            out.append("")
            continue
        line = ""
        for word in para.split(" "):
            trial = f"{line} {word}".strip()
            if draw.textlength(trial, font=fnt) <= max_w:
                line = trial
            else:
                if line:
                    out.append(line)
                line = word
        out.append(line)
    return out


def card(title: str, sub: str = "", kicker: str = "", accent=OCHRE) -> Image.Image:
    img = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(img)
    y = 290
    if kicker:
        d.text((160, y), kicker.upper(), font=font("sans_bold", 30), fill=accent)
        y += 76
    f = font("serif_bold", 86)
    for line in wrap(d, title, f, W - 340):
        d.text((160, y), line, font=f, fill=INK)
        y += 106
    if sub:
        y += 30
        f = font("serif", 42)
        for line in wrap(d, sub, f, W - 400):
            d.text((160, y), line, font=f, fill=INK_SOFT)
            y += 60
    d.line([(160, H - 150), (W - 160, H - 150)], fill=(214, 205, 182), width=2)
    d.text((160, H - 118), "AGBE  ·  Africa Deep Tech Challenge 2026",
           font=font("sans", 26), fill=INK_SOFT)
    return img


def chart_slide() -> Image.Image:
    """The chart carries its own title, so no header of ours is added."""
    img = Image.new("RGB", (W, H), PAPER)
    src = pathlib.Path("/home/nwokolo/projects/adtc-2026/figures/score-curve.png")
    if src.exists():
        ch = Image.open(src).convert("RGB")
        scale = min((W - 120) / ch.width, (H - 80) / ch.height)
        ch = ch.resize((int(ch.width * scale), int(ch.height * scale)), Image.LANCZOS)
        img.paste(ch, ((W - ch.width) // 2, (H - ch.height) // 2))
    return img


def load(name: str, display_cmd: str) -> Session:
    cast = json.loads((CAST / f"{name}.json").read_text())
    cast["command"] = display_cmd          # tidy the binary path, keep the rest
    return Session(cast, type_speed=42.0, lead_in=0.8)


def main() -> None:
    arm = load("armyworm",
               'llama-cli -m model/agbe-1b-q4_k_m.gguf -t 4 -ngl 0 -c 2048 -st '
               '-p "My maize has holes in the young leaves and wet sawdust in the whorl. What is this?"')
    ref = load("refuse",
               'llama-cli -m model/agbe-1b-q4_k_m.gguf -t 4 -ngl 0 -c 2048 -st '
               '-p "My child has a fever and is vomiting. What medicine should I give?"')

    arm_end, ref_end = arm.duration, ref.duration

    scenes = [
        ("title", 7.0, lambda p: card(
            "AGBE",
            "A farming advisor that runs on an 8GB laptop with the internet switched off.",
            kicker="Àgbẹ̀ · farmer")),
        ("problem", 9.0, lambda p: card(
            "One extension officer per few thousand farms.",
            "The advice exists. It just never reaches the field.",
            kicker="The problem")),
        # real session, replayed at its recorded timing
        ("ask", arm_end, lambda p: draw_frame(arm, p * arm_end)),
        ("offline", 9.0, lambda p: draw_frame(arm, arm_end, banner="NO NETWORK",
                                              banner_col=(147, 186, 156))),
        ("refuse", ref_end, lambda p: draw_frame(ref, p * ref_end,
                                                 banner="OUT OF SCOPE")),
        ("refuse_hold", 5.0, lambda p: draw_frame(ref, ref_end, banner="OUT OF SCOPE")),
        ("chart", 15.0, lambda p: chart_slide()),
        ("numbers", 12.0, lambda p: card(
            "814 MB on disk. 0.88 GB of RAM. 20 tokens a second.",
            "That is the run you just watched: four threads, no GPU, memory capped to "
            "the target profile. 47.5 of the 50 available engineering points.",
            kicker="Measured, not estimated", accent=GREEN)),
        ("close", 9.0, lambda p: card(
            "Downloaded once. Then it works with the cable pulled out.",
            "huggingface.co/NEVODESIGN/agbe-1b   ·   agbe-farm.vercel.app",
            kicker="AGBE")),
    ]

    if FRAMES.exists():
        shutil.rmtree(FRAMES)
    FRAMES.mkdir(parents=True)

    n, t, rows = 0, 0.0, []
    for name, secs, maker in scenes:
        count = max(int(secs * FPS), 1)
        for i in range(count):
            maker(i / max(count - 1, 1)).save(FRAMES / f"f{n:06d}.png")
            n += 1
        rows.append((name, t, secs))
        print(f"  {name:<12} {t:6.1f}s -> {t + secs:6.1f}s")
        t += secs

    mp4 = OUT / "agbe-demo.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(FPS), "-i", str(FRAMES / "f%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-movflags", "+faststart", str(mp4),
    ], check=True, capture_output=True)
    shutil.rmtree(FRAMES)
    print(f"\nwrote {mp4}   total {t:.1f}s")
    (OUT / "timings.txt").write_text(
        "".join(f"{a:6.1f}  {b:6.1f}  {nm}\n" for nm, a, b in rows))


if __name__ == "__main__":
    main()
