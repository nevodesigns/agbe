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
PAD_X, PAD_Y = 60, 96
MAX_LINES = (H - PAD_Y - 60) // LH

ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07|\r")


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
               banner_col=OCHRE) -> Image.Image:
    img = Image.new("RGB", (W, H), TERM_BG)
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(MONO, FS)
    fb = ImageFont.truetype(MONO_B, FS)

    # title bar
    d.rectangle([0, 0, W, 62], fill=BAR_BG)
    d.text((26, 17), "Terminal", font=ImageFont.truetype(MONO, 24), fill=DIM)
    if banner:
        bf = ImageFont.truetype(SANS_B, 24)
        bw = d.textlength(banner, font=bf)
        d.rectangle([W - bw - 56, 14, W - 24, 48], fill=banner_col)
        d.text((W - bw - 40, 19), banner, font=bf, fill=(18, 21, 17))

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
