#!/usr/bin/env python3
"""Keep work between Kaggle sessions in a private Hugging Face dataset repo, so a disconnect loses nothing.

    python3 train/hf_persist.py push NEVODESIGN/agbe-v2-work work/ --path-in-repo run1
    python3 train/hf_persist.py pull NEVODESIGN/agbe-v2-work work/ --path-in-repo run1
Needs HF_TOKEN with write access.
"""
import argparse, os, pathlib, sys

from huggingface_hub import HfApi, snapshot_download

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["push", "pull"])
ap.add_argument("repo")
ap.add_argument("folder")
ap.add_argument("--path-in-repo", default="run1")
a = ap.parse_args()
tok = os.environ.get("HF_TOKEN")
api = HfApi(token=tok)
if a.cmd == "push":
    api.create_repo(a.repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path=a.folder, repo_id=a.repo, repo_type="dataset", path_in_repo=a.path_in_repo,
                      ignore_patterns=["*.pyc", "__pycache__", "*.safetensors.tmp"])
    print("pushed", a.folder, "->", a.repo, a.path_in_repo)
else:
    try:
        p = snapshot_download(a.repo, repo_type="dataset", token=tok, allow_patterns=[a.path_in_repo + "/*"], local_dir=a.folder + "/.pull")
        src = pathlib.Path(p) / a.path_in_repo
        if src.exists():
            import shutil
            shutil.copytree(src, a.folder, dirs_exist_ok=True)
            print("restored", a.path_in_repo, "from", a.repo)
    except Exception as e:
        print("nothing to restore:", str(e)[:100])
