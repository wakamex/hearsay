#!/usr/bin/env bash
# Builds hearsay with every model family, then checks each on real models: Parakeet and Whisper (from
# a GGUF and from a whisper.cpp .bin file) transcribe tests/utterance.wav (LibriSpeech test-clean
# 1089-134686-0010, CC-BY-4.0), with Parakeet's raw token scores, and Pocket, Phonon (voice Delia) and
# Qwen3-TTS speak one sentence. The models are downloaded once into $HEARSAY_MODELS (default .models),
# pinned by commit and checked by SHA-256.
#   tests/smoke.sh [zig build options...]
set -euo pipefail
cd "$(dirname "$0")/.."
models=${HEARSAY_MODELS:-.models}
mkdir -p "$models/phonon-31m-en"

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
phonon=https://huggingface.co/gradium/phonon-31m-en/resolve/92741c5b07d085ff840e26fea8ce43110b87852f
fetch phonon-31m-en/model.q8.gguf $phonon/model.q8.gguf a7d9aad2cd6dc66473f24a08bc6d2939030ef4d751ef82349361fbef3272dbae
fetch phonon-31m-en/config.json $phonon/config.json b5d0f88353638370d94035b7d73d7d6577c0554d78f8d7ecfd40490aae39ae85
fetch phonon-31m-en/tokenizer.model $phonon/tokenizer.model 82e4572006d64d483317074c22c368e87779640b6f3a9dcd07b7caae32b030ce
fetch phonon-31m-en/delia.safetensors \
    https://huggingface.co/gradium/phonon-voices/resolve/0ee17205f254b5abc43b266c8a0f215565c352f7/voices/nGzMHMPAWGaJFHOo.safetensors \
    a0c3dcb91b2247eca56536be9ae8e8becc1c807b1136c5fe101552d6a155b24a
fetch whisper-tiny.en-Q8_0.gguf \
    https://huggingface.co/handy-computer/whisper-tiny.en-gguf/resolve/9dd46e00ef0577e354e658c07b3ac5580e9cd211/whisper-tiny.en-Q8_0.gguf \
    e8c9b73c06344307d8b346e07fbe93dd88d894627854bcff31523f1ce44394fa
fetch ggml-tiny.en-q8_0.bin \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-tiny.en-q8_0.bin \
    5bc2b3860aa151a4c6e7bb095e1fcce7cf12c7b020ca08dcec0c6d018bb7dd94
fetch qwen-talker-0.6b-customvoice-Q4_K_M.gguf \
    https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF/resolve/b7ee2e8c7459c3bea99da23e3d178125a7d1713c/qwen-talker-0.6b-customvoice-Q4_K_M.gguf \
    b3a7e6613d80f8a703c06267fc1e94d48ce91932ab82ab6e31c50f4ca4868e1e
fetch qwen-tokenizer-12hz-Q4_K_M.gguf \
    https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF/resolve/b7ee2e8c7459c3bea99da23e3d178125a7d1713c/qwen-tokenizer-12hz-Q4_K_M.gguf \
    cf3788b4d50aaa665fb6e57c170396aae03a3555fea52d2b5d0cda902d658039

zig build -Doptimize=ReleaseFast -Dhear=true -Dsay=true -Dwhisper=true -Dqwentts=true "$@"
bench=zig-out/bin/hearsay-bench
[ -x "$bench" ] || bench=zig-out/bin/hearsay-bench.exe
"$bench" --help > /dev/null
# Runs hearsay-bench, keeping its log out of the output unless it fails.
run() { "$bench" "$@" 2> "$out/log" || { echo "error: hearsay-bench $1 failed:" >&2; tail -n 20 "$out/log" >&2; exit 1; }; }

out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
echo tests/utterance.wav > "$out/wavs.txt"
for model in parakeet-tdt_ctc-110m-Q4_K_M.gguf whisper-tiny.en-Q8_0.gguf ggml-tiny.en-q8_0.bin; do
    heard=$(run hear "$models/$model" 2 "$out/wavs.txt" | tail -n 1)
    echo "heard ($model): $heard"
    case "$heard" in *declare*stick*) ;; *) echo "error: $model's transcript is not the recording's" >&2; exit 1 ;; esac
    # Parakeet TDT reports the joint's raw scores for every token.
    case "$model:$heard" in parakeet*null*) echo "error: Parakeet left a token's scores out" >&2; exit 1 ;; esac
done

echo "I found the bug in the parser." > "$out/texts.txt"
said=$(run say "$models/pocket-tts-english_2026-09-Q8_0.gguf" "$models/alba.safetensors" alba 2 "$out/texts.txt" "$out/say")
echo "said: $said"
bytes=$(wc -c < "$out/say-0.pcm")
# A second of 24 kHz 16-bit audio is 48,000 bytes; the sentence takes about two.
[ "$bytes" -gt 48000 ] || { echo "error: Pocket made only $bytes bytes of audio" >&2; exit 1; }

said=$(run say "$models/phonon-31m-en/model.q8.gguf" "$models/phonon-31m-en/delia.safetensors" delia 2 "$out/texts.txt" "$out/phonon")
echo "said (Phonon): $said"
bytes=$(wc -c < "$out/phonon-0.pcm")
[ "$bytes" -gt 48000 ] || { echo "error: Phonon made only $bytes bytes of audio" >&2; exit 1; }

said=$(run qwen "$models/qwen-talker-0.6b-customvoice-Q4_K_M.gguf" "$models/qwen-tokenizer-12hz-Q4_K_M.gguf" ryan 2 "$out/texts.txt" "$out/qwen")
echo "said (Qwen3-TTS): $said"
bytes=$(wc -c < "$out/qwen-0.pcm")
[ "$bytes" -gt 48000 ] || { echo "error: Qwen3-TTS made only $bytes bytes of audio" >&2; exit 1; }
echo "ok"
