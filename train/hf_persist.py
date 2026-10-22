#!/usr/bin/env python3
"""Keep work between Kaggle sessions in a private Hugging Face dataset repo, so a disconnect loses nothing.

    python3 train/hf_persist.py push NEVODESIGN/agbe-v2-work work/ --path-in-repo run1
    python3 train/hf_persist.py pull NEVODESIGN/agbe-v2-work work/ --path-in-repo run1
Needs HF_TOKEN with write access.
"""
import argparse, os, pathlib, shutil, sys, tempfile

from huggingface_hub import HfApi, snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["push", "pull"])
ap.add_argument("repo")
ap.add_argument("folder")
ap.add_argument("--path-in-repo", default="run1")
ap.add_argument("--ignore", nargs="*", default=[], help="pull only: file patterns inside the folder to skip (for example *.gguf)")
a = ap.parse_args()
tok = os.environ.get("HF_TOKEN")
api = HfApi(token=tok)
if a.cmd == "push":
    api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=a.folder, repo_id=a.repo, repo_type="dataset", path_in_repo=a.path_in_repo,
                      ignore_patterns=["*.pyc", "__pycache__", "*.safetensors.tmp", ".pull", ".pull/*", "*/.pull/*", ".cache/*", "*/.cache/*",
                                       "checkpoint-*/*", "*/checkpoint-*/*"])  # scratch copies and trainer checkpoints are not work worth keeping
    print("pushed", a.folder, "->", a.repo, a.path_in_repo)
else:
    try:
        scratch = tempfile.mkdtemp(prefix="hfpull_")  # outside the work folder, so it can never be pushed back
        p = snapshot_download(a.repo, repo_type="dataset", token=tok, allow_patterns=[a.path_in_repo + "/*"],
                              ignore_patterns=[a.path_in_repo + "/.pull/*", a.path_in_repo + "/*/.pull/*", a.path_in_repo + "/.cache/*"] + [a.path_in_repo + "/" + pat for pat in a.ignore], local_dir=scratch)
        src = pathlib.Path(p) / a.path_in_repo
        if src.exists():
            shutil.copytree(src, a.folder, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".pull", ".cache"))
            print("restored", a.path_in_repo, "from", a.repo)
        shutil.rmtree(scratch, ignore_errors=True)
    except Exception as e:
        print("nothing to restore:", str(e)[:100])
