#!/usr/bin/env bash
# Download the AGBE model weight file.
#
# Rules:
#   - Must be idempotent (safe to run multiple times).
#   - Must download without any credentials (public URL only).
#   - The output path must match `_runtime.model_path` in metadata.json.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$HERE/model"
MODEL_FILE="$MODEL_DIR/agbe-1b-q4_k_m.gguf"

MODEL_URL="https://huggingface.co/NEVODESIGN/agbe-1b/resolve/main/agbe-1b-q4_k_m.gguf"

# Exact expected identity of the shipped weights. Updated by tools/lock_model.sh
# when a build is chosen, and checked by tools/check_submission.sh against the
# live URL so these cannot silently drift from what is actually published.
EXPECT_BYTES=814261088
EXPECT_SHA256=d614d6b00aad21990419841bea8dae37502f8c57f1b3a25730ec15c3480d9851

verify() {  # verify <path> -> 0 if this is exactly the shipped file
  local f="$1" sz
  sz=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null || echo 0)
  [[ "$sz" == "$EXPECT_BYTES" ]] || return 1
  if command -v sha256sum > /dev/null 2>&1; then
    [[ "$(sha256sum "$f" | cut -d" " -f1)" == "$EXPECT_SHA256" ]] || return 1
  elif command -v shasum > /dev/null 2>&1; then
    [[ "$(shasum -a 256 "$f" | cut -d" " -f1)" == "$EXPECT_SHA256" ]] || return 1
  fi
  return 0
}

mkdir -p "$MODEL_DIR"

# Idempotent, but only for a file that is actually correct. The previous version
# skipped on mere existence, which would have made a corrupt or half-moved file
# permanent: every later run reported success and llama.cpp reported only
# "failed to load model".
if [[ -f "$MODEL_FILE" ]]; then
  if verify "$MODEL_FILE"; then
    echo "model already present and verified at $MODEL_FILE — skipping download"
    exit 0
  fi
  echo "existing file does not match the expected size or checksum, refetching…"
  rm -f "$MODEL_FILE"
fi

# A resume target that is already at or past full length cannot be repaired by
# appending: curl -C - simply has nothing to add and exits happy, leaving the old
# content at exactly the right size. That is the one case where a size check
# passes a wrong file, which is why the checksum above is not optional.
if [[ -f "$MODEL_FILE.partial" ]]; then
  PSZ=$(stat -c%s "$MODEL_FILE.partial" 2>/dev/null || stat -f%z "$MODEL_FILE.partial" 2>/dev/null || echo 0)
  if [[ "$PSZ" -ge "$EXPECT_BYTES" ]]; then
    echo "discarding an unusable partial file ($PSZ bytes)"
    rm -f "$MODEL_FILE.partial"
  fi
fi

echo "downloading $MODEL_URL → $MODEL_FILE (~0.8 GB)…"

# Resume rather than restart. The target user is on a rural connection where an
# 800 MB transfer routinely drops part way, and starting from zero each time can
# mean never finishing. -C - continues from whatever is already on disk, and we
# retry a few times before giving up.
if command -v curl > /dev/null 2>&1; then
  curl -L --fail --progress-bar -C - --retry 5 --retry-delay 5 \
       -o "$MODEL_FILE.partial" "$MODEL_URL" || true
elif command -v wget > /dev/null 2>&1; then
  wget --continue --tries=5 --show-progress -O "$MODEL_FILE.partial" "$MODEL_URL" || true
else
  echo "error: neither curl nor wget found" >&2
  exit 1
fi

# Verify before committing the name. A truncated GGUF keeps a valid magic number
# at byte 0, so a header check passes on a broken file; only the length catches it.
# We hit exactly this during development: curl exited 0 on three partial models
# and llama.cpp reported only "failed to load model".
ACTUAL=$(stat -c%s "$MODEL_FILE.partial" 2>/dev/null || stat -f%z "$MODEL_FILE.partial" 2>/dev/null || echo 0)
if ! verify "$MODEL_FILE.partial"; then
  if [[ "$ACTUAL" -lt "$EXPECT_BYTES" ]]; then
    echo "error: got $ACTUAL of $EXPECT_BYTES bytes. The transfer was cut short." >&2
    echo "The partial file is kept — re-run this script and it resumes from here." >&2
  else
    echo "error: file is the wrong size or checksum ($ACTUAL bytes)." >&2
    echo "Delete $MODEL_FILE.partial and re-run to fetch it cleanly." >&2
  fi
  exit 1
fi

mv "$MODEL_FILE.partial" "$MODEL_FILE"
echo "done: $MODEL_FILE ($ACTUAL bytes, sha256 verified)"
