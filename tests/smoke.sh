#!/usr/bin/env bash
# Builds hearsay, then checks both engines on real models: Parakeet transcribes tests/utterance.wav
# (LibriSpeech test-clean 1089-134686-0010, CC-BY-4.0) and Pocket speaks one sentence. The models are
# downloaded once into $HEARSAY_MODELS (default .models), pinned by commit and checked by SHA-256.
#   tests/smoke.sh [zig build options...]
set -euo pipefail
cd "$(dirname "$0")/.."
models=${HEARSAY_MODELS:-.models}
mkdir -p "$models"

sha256() { if command -v sha256sum > /dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
fetch() {  # NAME URL SHA256
    if [ ! -f "$models/$1" ] || [ "$(sha256 "$models/$1")" != "$3" ]; then
        curl -fsSL -o "$models/$1.part" "$2"
        [ "$(sha256 "$models/$1.part")" = "$3" ] || { echo "error: $1 does not match its published SHA-256" >&2; exit 1; }
        mv "$models/$1.part" "$models/$1"
    fi
}
fetch parakeet-tdt_ctc-110m-Q4_K_M.gguf \
    https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf/resolve/1c3f0d5378fdd358905cc410a93ed9e64f52d170/parakeet-tdt_ctc-110m-Q4_K_M.gguf \
    486414fd90185a8c8a4ced7c123cfb133ff4f7958426c6b8bd9049946b56b448
fetch pocket-tts-english_2026-09-Q8_0.gguf \
    https://huggingface.co/Serveurperso/pocket-tts-GGUF/resolve/b5a0115aee9658c110d747debb03edd548c4b9c7/pocket-tts-english_2026-09-Q8_0.gguf \
    1c91f8cd54efd089236737a338e998b9cd1bc9c3b1d6bc5bd038a99f2527deba
fetch alba.safetensors \
    https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/575d14372df89a4534bcb989870a34e77544b09f/languages/english_2026-09/embeddings/alba.safetensors \
    d291428b416d6c36a1de7835e51dbe1e334b75e5af512bb18dd23a1047fe8f3b

zig build -Doptimize=ReleaseFast -Dhear=true -Dsay=true "$@"
bench=zig-out/bin/hearsay-bench
[ -x "$bench" ] || bench=zig-out/bin/hearsay-bench.exe
"$bench" --help > /dev/null

out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
echo tests/utterance.wav > "$out/wavs.txt"
heard=$("$bench" hear "$models/parakeet-tdt_ctc-110m-Q4_K_M.gguf" 2 "$out/wavs.txt" 2> /dev/null | tail -n 1)
echo "heard: $heard"
case "$heard" in *declare*stick*) ;; *) echo "error: the transcript is not the recording's" >&2; exit 1 ;; esac

echo "I found the bug in the parser." > "$out/texts.txt"
said=$("$bench" say "$models/pocket-tts-english_2026-09-Q8_0.gguf" "$models/alba.safetensors" alba 2 "$out/texts.txt" "$out/say" 2> /dev/null)
echo "said: $said"
bytes=$(wc -c < "$out/say-0.pcm")
# A second of 24 kHz 16-bit audio is 48,000 bytes; the sentence takes about two.
[ "$bytes" -gt 48000 ] || { echo "error: Pocket made only $bytes bytes of audio" >&2; exit 1; }
echo "ok"
