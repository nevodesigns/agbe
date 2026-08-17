"""Replay a recorded terminal cast as video frames.

Given a cast from record_session.py, this reproduces the session on screen at the
timing it actually happened: the prompt appears, the command types out at a human
speed, then output streams in at the rate the model genuinely produced it.

Because the cast carries real timestamps, the "thinking" pause before the first
token and the pace of generation are both real. Nothing is sped up to look
better, which matters: a judge should see the speed they will actually get.
"""

from __future__ import annotations

import json
import pathlib
import re

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
TERM_BG = (18, 21, 17)
BAR_BG = (28, 32, 26)
FG = (233, 228, 213)
DIM = (168, 176, 160)
GREEN = (147, 186, 156)
OCHRE = (216, 160, 92)
CURSOR = (233, 228, 213)

FD = "/usr/share/fonts/truetype/dejavu"
MONO = f"{FD}/DejaVuSansMono.ttf"
MONO_B = f"{FD}/DejaVuSansMono-Bold.ttf"
SANS_B = f"{FD}/DejaVuSans-Bold.ttf"

FS = 26
LH = 37
BAR_H = 42                      # GNOME-style desktop bar across the very top
TITLE_H = 58                    # the terminal window's own title strip
PAD_X, PAD_Y = 60, BAR_H + TITLE_H + 38
MAX_LINES = (H - PAD_Y - 56) // LH

ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|\r")



# ---------------------------------------------------------------------------
# Desktop bar. Drawn rather than screenshotted so it renders crisply at 1080p
# and so the network icon can be switched between wifi and aeroplane mode. The
# offline claim is the whole argument of this project, so it should be visible
# in the status bar the way it would be on a real machine, not asserted in a
# caption.
# ---------------------------------------------------------------------------

BAR_BG_TOP = (10, 10, 10)
BAR_FG = (232, 232, 232)


def _wifi(d, cx, cy, col):
    """Three arcs and a dot."""
    for i, r in enumerate((13, 9, 5)):
        d.arc([cx - r, cy - r - 2, cx + r, cy + r - 2], start=215, end=325,
              fill=col, width=2)
    d.ellipse([cx - 2, cy + 4, cx + 2, cy + 8], fill=col)


def _airplane(d, cx, cy, col):
    """Aeroplane silhouette, nose up, as GNOME draws aeroplane mode."""
    d.polygon([(cx, cy - 11), (cx + 3, cy - 5), (cx + 3, cy + 1),
               (cx + 12, cy + 6), (cx + 12, cy + 8), (cx + 3, cy + 6),
               (cx + 3, cy + 10), (cx + 6, cy + 13), (cx + 6, cy + 14),
               (cx, cy + 12),
               (cx - 6, cy + 14), (cx - 6, cy + 13), (cx - 3, cy + 10),
               (cx - 3, cy + 6), (cx - 12, cy + 8), (cx - 12, cy + 6),
               (cx - 3, cy + 1), (cx - 3, cy - 5)], fill=col)


def _volume(d, cx, cy, col):
    d.polygon([(cx - 9, cy - 3), (cx - 4, cy - 3), (cx + 1, cy - 9),
               (cx + 1, cy + 9), (cx - 4, cy + 3), (cx - 9, cy + 3)], fill=col)
    d.arc([cx + 1, cy - 8, cx + 11, cy + 8], start=300, end=60, fill=col, width=2)


def _battery(d, cx, cy, col):
    d.rounded_rectangle([cx - 13, cy - 7, cx + 10, cy + 7], radius=3,
                        outline=col, width=2)
    d.rounded_rectangle([cx + 11, cy - 3, cx + 14, cy + 3], radius=1, fill=col)
    d.rectangle([cx - 10, cy - 4, cx + 7, cy + 4], fill=col)


def _bell(d, cx, cy, col):
    d.pieslice([cx - 8, cy - 9, cx + 8, cy + 7], start=180, end=360, fill=col)
    d.rectangle([cx - 8, cy - 1, cx + 8, cy + 4], fill=col)
    d.rectangle([cx - 10, cy + 4, cx + 10, cy + 6], fill=col)
    d.ellipse([cx - 2, cy + 7, cx + 2, cy + 11], fill=col)


def desktop_bar(d, clock: str = "Aug 17  00:44", airplane: bool = False) -> None:
    d.rectangle([0, 0, W, BAR_H], fill=BAR_BG_TOP)
    f = ImageFont.truetype(f"{FD}/DejaVuSans.ttf", 21)

    # centre: clock, then the notification bell
    tw = d.textlength(clock, font=f)
    x = (W - tw) // 2 - 16
    d.text((x, BAR_H // 2 - 12), clock, font=f, fill=BAR_FG)
    _bell(d, int(x + tw + 20), BAR_H // 2 - 1, BAR_FG)

    # right: network, volume, battery, percentage.
    # Margins are generous because "100 %" was clipping off the right edge.
    pct = "100 %"
    pw = d.textlength(pct, font=f)
    d.text((W - 28 - pw, BAR_H // 2 - 12), pct, font=f, fill=BAR_FG)
    _battery(d, int(W - 46 - pw), BAR_H // 2, BAR_FG)
    _volume(d, int(W - 90 - pw), BAR_H // 2, BAR_FG)
    if airplane:
        _airplane(d, int(W - 132 - pw), BAR_H // 2, BAR_FG)
    else:
        _wifi(d, int(W - 132 - pw), BAR_H // 2, BAR_FG)


def clean(text: str) -> str:
    """Strip escape sequences and llama.cpp's spinner characters."""
    text = ANSI.sub("", text)
    return "".join(c for c in text if c == "\n" or c >= " ")


def wrap_cols(text: str, cols: int) -> list[str]:
    out: list[str] = []
    for para in text.split("\n"):
        if not para:
            out.append("")
            continue
        while len(para) > cols:
            cut = para.rfind(" ", 0, cols)
            cut = cut if cut > cols // 2 else cols
            out.append(para[:cut])
            para = para[cut:].lstrip()
        out.append(para)
    return out


class Session:
    """A cast expanded into (time -> screen text) so frames can be sampled."""

    def __init__(self, cast: dict, type_speed: float = 22.0, lead_in: float = 1.2):
        self.prompt = cast["prompt"]
        self.command = cast["command"]
        self.lead_in = lead_in
        self.type_dur = len(self.command) / type_speed

        body, self.stream = "", []
        for t, chunk in cast["events"]:
            body += clean(chunk)
            self.stream.append((t, body))
        self.duration = self.lead_in + self.type_dur + (
            cast["events"][-1][0] if cast["events"] else 0.0) + 2.0

    def text_at(self, t: float) -> tuple[str, bool]:
        """Screen contents at time t, and whether the cursor should be shown."""
        if t < self.lead_in:
            return "", True
        t -= self.lead_in
        if t < self.type_dur:
            n = int(len(self.command) * (t / self.type_dur))
            return self.command[:n], True
        t -= self.type_dur
        body = ""
        for ts, snapshot in self.stream:
            if ts <= t:
                body = snapshot
            else:
                break
        return self.command + "\n" + body, False


def draw_frame(sess: Session, t: float, banner: str = "",
               banner_col=OCHRE, airplane: bool = False,
               clock: str = "Aug 17  00:44") -> Image.Image:
    img = Image.new("RGB", (W, H), TERM_BG)
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(MONO, FS)
    fb = ImageFont.truetype(MONO_B, FS)

    desktop_bar(d, clock=clock, airplane=airplane)

    # terminal window title strip, below the desktop bar
    d.rectangle([0, BAR_H, W, BAR_H + TITLE_H], fill=BAR_BG)
    d.text((26, BAR_H + 16), "nwokolo@NEVO-ELITE: ~/projects/agbe",
           font=ImageFont.truetype(MONO, 23), fill=DIM)
    if banner:
        bf = ImageFont.truetype(SANS_B, 23)
        bw = d.textlength(banner, font=bf)
        d.rectangle([W - bw - 56, BAR_H + 13, W - 24, BAR_H + 45], fill=banner_col)
        d.text((W - bw - 40, BAR_H + 17), banner, font=bf, fill=(18, 21, 17))

    typed, cursor = sess.text_at(t)
    cols = (W - PAD_X * 2) // int(d.textlength("M", font=f))

    # The command line is PINNED. Long output used to scroll the shell prompt off
    # the top, which hid the working directory, the very thing the terminal scene
    # exists to show. Only the output area scrolls.
    head, _, tail = typed.partition("\n")
    cmd_lines = wrap_cols(sess.prompt + head, cols)

    y = PAD_Y
    for i, ln in enumerate(cmd_lines):
        if i == 0 and ln.startswith(sess.prompt):
            d.text((PAD_X, y), sess.prompt, font=f, fill=GREEN)
            d.text((PAD_X + d.textlength(sess.prompt, font=f), y),
                   ln[len(sess.prompt):], font=fb, fill=FG)
        else:
            d.text((PAD_X, y), ln, font=fb, fill=FG)
        y += LH

    out_lines = wrap_cols(tail, cols) if tail else []
    room = MAX_LINES - len(cmd_lines) - 1
    if len(out_lines) > room:
        out_lines = out_lines[-room:]
    y += 8
    for ln in out_lines:
        d.text((PAD_X, y), ln, font=f, fill=DIM)
        y += LH

    lines = [(cmd_lines[-1] if not out_lines else out_lines[-1], f, DIM)]

    if cursor and int(t * 2) % 2 == 0:
        last = lines[-1][0] if lines else ""
        d.rectangle([PAD_X + d.textlength(last, font=f), y - LH + 4,
                     PAD_X + d.textlength(last + "M", font=f), y - 4], fill=CURSOR)
    return img
