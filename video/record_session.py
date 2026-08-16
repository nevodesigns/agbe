"""Record a REAL terminal session, with real timing, for the demo video.

The first cut of the video faked the terminal with slides. That is weaker and a
little dishonest: a judge watching a demo wants to see the thing run, not a
picture of it having run.

This spawns the actual command in a pseudo-terminal, captures every byte with a
timestamp, and writes a cast file. The renderer then replays that cast frame by
frame, so what appears on screen is the real prompt, the real working directory,
the real command, and output arriving at the speed the model actually produced
it. Nothing is re-timed.

Output: video/cast/<name>.json  ->  {"prompt", "command", "events": [[t, text]]}
"""

from __future__ import annotations

import json
import os
import pathlib
import pty
import select
import shlex
import subprocess
import sys
import time

OUT = pathlib.Path(__file__).resolve().parent / "cast"
OUT.mkdir(parents=True, exist_ok=True)

USER = os.environ.get("USER", "nwokolo")
HOST = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()


def short_cwd(path: pathlib.Path) -> str:
    home = pathlib.Path.home()
    try:
        return "~/" + str(path.relative_to(home))
    except ValueError:
        return str(path)


def record(name: str, cwd: str, argv: list[str], timeout: float = 900.0) -> dict:
    """Run argv in a PTY from cwd, capturing output with timestamps."""
    cwd_path = pathlib.Path(cwd)
    prompt = f"{USER}@{HOST}:{short_cwd(cwd_path)}$ "
    command = " ".join(shlex.quote(a) if " " in a else a for a in argv)

    events: list[tuple[float, str]] = []
    pid, fd = pty.fork()
    if pid == 0:                                   # child
        os.chdir(cwd)
        os.environ["TERM"] = "dumb"                # no colour codes to strip
        os.execvp(argv[0], argv)

    start = time.time()
    try:
        while True:
            if time.time() - start > timeout:
                break
            r, _, _ = select.select([fd], [], [], 0.2)
            if fd in r:
                try:
                    chunk = os.read(fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                events.append((round(time.time() - start, 3),
                               chunk.decode("utf-8", "replace")))
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass

    cast = {"prompt": prompt, "command": command,
            "duration": round(time.time() - start, 2), "events": events}
    (OUT / f"{name}.json").write_text(json.dumps(cast))
    total = sum(len(t) for _, t in events)
    print(f"  {name:<10} {cast['duration']:6.1f}s  {len(events):4d} chunks  {total:5d} chars")
    return cast


LLAMA = "/home/nwokolo/projects/adtc-2026/work/llama.cpp/build/bin/llama-cli"
MODEL = "model/agbe-1b-q4_k_m.gguf"
AGBE = "/home/nwokolo/projects/agbe"


def main() -> None:
    base = [LLAMA, "-m", MODEL, "-t", "4", "-ngl", "0", "-c", "2048",
            "-n", "200", "--temp", "0.3", "-st", "--simple-io", "--no-warmup",
            "--repeat-penalty", "1.15", "-p"]

    print("recording real sessions (this runs the model, so it takes a few minutes)")
    record("armyworm", AGBE, base + [
        "My maize has holes in the young leaves and wet sawdust in the whorl. What is this?"])
    record("refuse", AGBE, base + [
        "My child has a fever and is vomiting. What medicine should I give?"])


if __name__ == "__main__":
    main()
