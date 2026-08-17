"""Generate subtitles for the demo video and burn them in.

Two outputs, because they serve different purposes:

  agbe-demo.srt            upload alongside the video on YouTube, so captions are
                           toggleable, indexable and screen-reader friendly
  agbe-demo-captioned.mp4  captions burned in, so the video is comprehensible on
                           any player and before the voiceover exists

Cue timings come from video/timings.txt, the same scene boundaries the renderer
produced, so captions and picture cannot drift apart. Within a scene, lines are
allotted time in proportion to their length, which approximates a natural
speaking pace better than dividing evenly.

The narration text is identical to TRANSCRIPT.md, so what a viewer reads is what
the voiceover will say.
"""

from __future__ import annotations

import pathlib
import subprocess

OUT = pathlib.Path(__file__).resolve().parent

# Per scene: the narration, split at the points a speaker would naturally pause.
# Two lines maximum per cue, because three is hard to read at a glance.
SCRIPT: dict[str, list[str]] = {
    "title": [
        "AGBE. Àgbẹ̀ means farmer in Yoruba.",
        "A farming advisor that runs on an ordinary laptop,\nwith the internet switched off.",
    ],
    "problem": [
        "Nigeria has about one agricultural extension officer\nfor every few thousand farming households.",
        "The knowledge that would raise a smallholder's yield\nis not secret, and it is not new.",
        "It sits in extension manuals.\nIt just never reaches the field.",
    ],
    "ask": [
        "So we asked it what farmers actually ask.",
        "Holes in the young maize leaves, and something like\nwet sawdust in the centre of the plant.",
        "It says fall armyworm.",
        "Then it tells you how to check before you spend money:\na pale upside down Y on the head of the caterpillar,",
        "and four dark dots near the tail.",
        "That is real fall armyworm biology, not a guess.",
    ],
    "offline": [
        "This is the part that matters.",
        "Look at the status bar. That is aeroplane mode.",
        "No internet, no API key, no account, no data cost.\nThe model is on the laptop.",
    ],
    "refuse": [
        "Now something it should not answer.",
        "A child with a fever and vomiting.",
        "It declines. It says it does not give medical advice,",
        "tells you to contact a doctor today, and not to give\nany medication without speaking to a professional.",
    ],
    "refuse_hold": [
        "Getting a one billion parameter model to say no\ntook four attempts.",
        "For a tool used by people with no alternative,\nknowing where it stops matters as much as what it knows.",
    ],
    "chart": [
        "The challenge publishes its scoring formula,\nso we read it before writing any code.",
        "Throughput above fifteen tokens a second earns nothing,\nand memory is charged linearly.",
        "That makes the obvious move, running the biggest model\nthat fits in eight gigabytes, exactly backwards.",
        "The three billion model gives up fourteen and a half points\nbefore answering a single question.",
        "So we measured five candidates, and built the one billion.",
    ],
    "numbers": [
        "Eight hundred and fourteen megabytes on disk.\nUnder a gigabyte of memory.",
        "Twenty tokens a second on four CPU threads,\nwith no GPU at all.",
        "Forty seven and a half of the fifty available\nengineering points, measured rather than estimated.",
    ],
    "close": [
        "Downloaded once. After that it works\nwith the cable pulled out.",
        "The weights are public, so you can check\nevery number in this video yourself.",
    ],
}


def timecode(t: float) -> str:
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"


def main() -> None:
    scenes = []
    for line in (OUT / "timings.txt").read_text().splitlines():
        start, dur, name = line.split()
        scenes.append((name, float(start), float(dur)))

    cues: list[tuple[float, float, str]] = []
    for name, start, dur in scenes:
        lines = SCRIPT.get(name)
        if not lines:
            continue
        # allot time by text length, so long lines get longer on screen
        weights = [max(len(l), 20) for l in lines]
        total = sum(weights)
        t = start
        for line, w in zip(lines, weights):
            span = dur * (w / total)
            cues.append((t, t + span, line))
            t += span

    srt = OUT / "agbe-demo.srt"
    with srt.open("w") as fh:
        for i, (a, b, text) in enumerate(cues, 1):
            fh.write(f"{i}\n{timecode(a)} --> {timecode(b)}\n{text}\n\n")
    print(f"wrote {srt}  ({len(cues)} cues)")

    # ---- ASS, with the real resolution declared -------------------------
    #
    # Burning straight from SRT with force_style went wrong: libass assumes a
    # 288-line script when none is declared, so a "21pt" font rendered at nearly
    # 80px on a 1080p frame and swamped the picture. Declaring PlayResX/Y makes
    # every size below a real pixel value.
    def ass_time(t: float) -> str:
        h, rem = divmod(t, 3600)
        m, sec = divmod(rem, 60)
        return f"{int(h)}:{int(m):02d}:{sec:05.2f}"

    ass = OUT / "agbe-demo.ass"
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,DejaVu Sans,38,&H00E9E4D5,&H000000FF,&HD0141712,&H00000000,0,0,0,0,100,100,0,0,3,10,0,2,140,140,54,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with ass.open("w") as fh:
        fh.write(header)
        for a, b, text in cues:
            fh.write(f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Cap,,0,0,0,,"
                     f"{text.replace(chr(10), chr(92) + 'N')}\n")
    print(f"wrote {ass}")

    src = OUT / "agbe-demo.mp4"
    burned = OUT / "agbe-demo-captioned.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"ass={ass}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
        "-movflags", "+faststart", str(burned),
    ], check=True, capture_output=True)
    print(f"wrote {burned}")

    # Soft subtitles too, toggleable in players that support them.
    soft = OUT / "agbe-demo-softsubs.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-i", str(src), "-i", str(srt),
        "-c", "copy", "-c:s", "mov_text", "-metadata:s:s:0", "language=eng",
        str(soft),
    ], check=True, capture_output=True)
    print(f"wrote {soft}")


if __name__ == "__main__":
    main()
