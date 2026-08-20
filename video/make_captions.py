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
        "A farming advisor that runs on an ordinary laptop\nwith the internet switched off.",
    ],
    "problem": [
        "A farmer walks out at first light and finds ragged\nholes in the young maize,",
        "and wet sawdust in the centre of the plant.",
        "They need to know what it is, and whether it is worth\nspending money on, today.",
        "The nearest extension officer covers thousands of\nhouseholds, and there is no signal out here.",
    ],
    "ask": [
        "So we asked it exactly that.",
        "It says fall armyworm.",
        "Then it explains that the wet sawdust is frass,\nthe caterpillar's droppings, packed into the whorl.",
        "And it tells you to open the whorl and count the damage\nacross the field before you spend anything.",
    ],
    "offline": [
        "Now look at the status bar. That is aeroplane mode.",
        "No internet, no API key, no account, no data cost.\nThe model is on the laptop.",
    ],
    "refuse": [
        "Now something it should not answer.",
        "A child with a fever and vomiting.",
        "It declines. It says it does not give medical advice,",
        "tells you to contact a doctor today, and not to give\nany medication without speaking to a professional first.",
    ],
    # refuse_hold is deliberately silent: the refusal reads better without a
    # voice on top of it, so it carries no cue.
    "numbers": [
        "Eight hundred and fourteen megabytes on disk.\nAbout one gigabyte of memory.",
        "Twenty six tokens a second on four CPU threads,\nwith no GPU.",
        "Measured on the target hardware,\nnot taken off a spec sheet.",
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

    # MarginV 132, not 54: the card scenes carry a footer line at H-118 and at the
    # old margin the caption sat straight on top of it, interleaving two texts so
    # that "Challenge 2026" read as "Challenge 2023".
    #
    # Box alpha 20, not D0. In ASS the alpha byte runs 00 = OPAQUE to FF =
    # transparent, which is backwards from every other format here. D0 was 82%
    # transparent, so on the light cards the caption was effectively invisible;
    # it only ever looked fine because it had been tuned against dark terminal
    # frames where the text colour alone carried it.
    ass = OUT / "agbe-demo.ass"
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,DejaVu Sans,38,&H00E9E4D5,&H000000FF,&H20141712,&H00000000,0,0,0,0,100,100,0,0,3,12,0,2,140,140,132,1

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
