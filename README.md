# hearsay

hearsay is the local speech engine for gear, a voice front-end for coding agents: it hears, turning speech into text, and says, turning text into speech, in one small native build, on the CPU or, optionally, on the GPU through [Vulkan](https://www.vulkan.org/). It compiles only the parts of three upstream projects that its callers ask for against a single copy of the [ggml](https://github.com/ggml-org/ggml) tensor library:

- Hearing: NVIDIA's Parakeet models, such as [Parakeet TDT-CTC 110M](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m), through [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp), with only its Parakeet family built, and optionally OpenAI's [Whisper](https://github.com/openai/whisper) models from the same library, from GGUF or whisper.cpp `.bin` files, and [Silero VAD](https://github.com/snakers4/silero-vad), for telling speech from other sound.
- Saying: Kyutai's [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) through [pockettts.cpp](https://github.com/ServeurpersoCom/pockettts.cpp), which streams audio as it is generated, without its command-line tool and HTTP server, and optionally Alibaba's [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (base models that clone a voice from a recording, custom-voice models with named speakers, and voice-design models that take a description of the voice) through [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp), by the same author.

The projects are fetched at pinned commits and changed only by the patch files in `patches/`. The GPU build also fetches the Khronos [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) and [SPIRV-Headers](https://github.com/KhronosGroup/SPIRV-Headers) at pinned releases.

## Result

The combined build matches the two projects built separately and is less than a third of their combined size. One benchmark program holding both engines and ggml is 2.3 MB stripped, against 4.7 MB for transcribe.cpp's command-line program and 2.4 MB for pockettts.cpp's. Every transcript is identical to upstream transcribe.cpp's, and synthesis was bit-identical to upstream pockettts.cpp's when built with the same ggml kernels, before `patches/pockettts/0002` put a comma before every chunk. With the kernels hearsay uses, Pocket gets faster: 0.22 s to first audio instead of 0.29 s on one thread, and 0.50 CPU-seconds per second of audio instead of 0.63. Recognition costs 1.4% more CPU than upstream transcribe.cpp.

## Measured against the separate builds

Each row runs on one thread on an idle Ryzen 7 3800XT (Zen 2 with AVX2, 8 cores) under Fedora 44 in WSL2. Hearing transcribes 57 utterances (460 s) from [LibriSpeech](https://www.openslr.org/12) one at a time with the Q4_K_M weights; saying synthesizes 8 agent-style replies (62 s of audio) with the english_2026-09 Q8_0 model, the alba voice and seed 42, after one warm-up pass. Lower is better in every column except speed.

| Engine | Build | Time per 10 s of speech heard | First audio, median (worst) | Speed vs real time | CPU-seconds per second of audio | Peak memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Hearing | transcribe.cpp alone | 0.65 s | | | 0.066 | 0.31 GB |
| Hearing | hearsay | 0.66 s | | | 0.070 | 0.31 GB |
| Saying | pockettts.cpp alone | | 0.29 s (0.33 s) | 1.59x | 0.63 | 0.28 GB |
| Saying | hearsay | | 0.22 s (0.26 s) | 1.98x | 0.50 | 0.28 GB |

On two threads hearsay's Pocket reaches first audio in 0.12 s (worst 0.14 s) at 3.35x real time.

The hearing rows come from one run each. Three alternating runs of each build put hearsay's recognition compute at 30.8 to 31.0 s against 30.4 to 30.5 s for upstream, the 1.4% above; the benchmark tools start their clocks at slightly different points, which may account for part of it.

Parity checks:

- Hearing: all 556 transcripts from every tenth utterance of LibriSpeech test-clean and test-other are identical to upstream transcribe.cpp at the same commit.
- Saying (before `patches/pockettts/0002`, which changes the text Pocket reads): with llamafile's kernels off, all 8 replies are bit-identical to upstream pockettts.cpp with the same model, voice and seed. With them on, four replies differ by at most 110 in 32,767, and the other four diverge after their first seconds. Pocket feeds each sampled 80 ms frame into the next, so the rounding difference from the other matrix kernels changes the rest of the sample; both are valid samples of the same text.

## Changes to upstream

| Patch | Why |
| --- | --- |
| `patches/transcribe/0001-register-only-built-families.patch` | transcribe.cpp lists all 21 model families in one table, which would link every family. hearsay's copy lists only the families the build includes: Parakeet, Whisper and Silero VAD. |
| `patches/transcribe/0002-stdio-file-io.patch` | Replaces `std::ifstream` and `std::filesystem::status` in model loading with `ggml_fopen`, 64-bit `fseeko` / `_fseeki64` and `stat` / `GetFileAttributesW`, avoiding `std::locale` and `std::filesystem` in the linked binary while preserving UTF-8 path support on Windows. |
| `patches/transcribe/0003-silero-stdio-file-io.patch` | The same change as 0002 for Silero VAD's reader of whisper.cpp-format `.bin` files, which used `std::ifstream` and the `path_from_utf8` that 0002 removes. |
| `patches/transcribe/0004-whisper-bin-stdio-file-io.patch` | The same change as 0002 for the reader of whisper.cpp-format `.bin` files that Whisper uses, `transcribe-bin-loader.cpp`. |
| `patches/ggml/0001` to `0003` | The three commits pockettts.cpp's ggml fork ([ServeurpersoCom/ggml](https://github.com/ServeurpersoCom/ggml)) adds to upstream ggml, by Pascal and copied unchanged (see [License](#license)): a fused snake activation on the CPU and two `conv_1d` fixes for batched input. Both forks start from the same upstream ggml (`d7cb574` and its successor `ffa4e8b`, which changes only a release script), so these apply on top of transcribe.cpp's patched ggml. |
| `patches/pockettts/0001-thread-count-from-POCKET_THREADS.patch` | Reads `POCKET_THREADS` for the CPU thread count instead of always using one thread per physical core, and skips `ggml_backend_load_all()` when `POCKET_STATIC` is defined so static builds do not scan the working directory for dynamic backend plugins. |
| `patches/qwentts/0001-thread-count-from-QWEN_THREADS.patch` | The same change as `patches/pockettts/0001` for qwentts.cpp, with `QWEN_THREADS` and `QWEN_STATIC`. |
| `patches/pockettts/0002-comma-before-every-chunk.patch` | Puts ", " before the text of every chunk. pockettts.cpp drops the audio of each chunk's first frame (80 ms), which decodes to a click, and when the voice starts at once a short first word goes with it: on 480 sentences opening with "I" (30 sentences, 16 seeds), the "I" was lost from 4.4% at the start of a text and 8.1% at the start of a later chunk, and from none of either with the comma, which adds about 0.08 s of quiet before each chunk's speech. A comma added to the text by the caller does not do this: Pocket can cut a long first sentence at it and voice it as a chunk of its own. |

`src/no-bin-loaders.cpp` stands in for the functions transcribe.cpp's shared loader calls for whisper.cpp-format `.bin` files and for the development tensor dumper in `transcribe-debug.cpp`. Whisper's and Silero VAD's `.bin` loaders return `TRANSCRIBE_ERR_NOT_IMPLEMENTED` unless `-Dwhisper` or `-Dsilero` builds that family, whose own loader then reads them; `transcribe-debug.cpp` is omitted, and so is `transcribe-bin-loader.cpp` without Whisper. Parakeet's model code links the Sortformer diarizer for its multi-speaker variant, so that family is compiled as is.

`src/vulkan-loader.cpp`, in GPU builds only, opens the Vulkan loader (`vulkan-1.dll` or `libvulkan.so.1`) when the program starts instead of linking it, and supplies the four Vulkan functions ggml calls directly by forwarding them to the loader's. Without a loader it sets `GGML_DISABLE_VULKAN`, so ggml registers only the CPU and the program runs there.

qwentts.cpp pins its author's ggml fork at [cc5df49](https://github.com/ServeurpersoCom/ggml/commit/cc5df49a6b878db2b68b9a7b45976e279594627d), upstream ggml 0.27.0 with the fused snake activation (`patches/ggml/0001`) and a SYCL fix. It calls nothing that ggml 0.26.0 lacks: the public headers gained only two scheduler callbacks between them, which it does not use. So it builds on hearsay's ggml unchanged. Its sources are copied into the build cache with the patched `backend.h`, because they find their headers beside them and several share names with pockettts.cpp's (`backend.h`, `debug.h`, `weight-ctx.h`).

Build settings: ggml is built with transcribe.cpp's settings, without BLAS or OpenMP and with [llamafile](https://github.com/Mozilla-Ocho/llamafile)'s matrix kernels, which make both engines faster on this CPU. `GGML_MAX_NAME` is 128 for every target, as pockettts.cpp needs; it changes the layout of ggml's tensor struct, so it must match everywhere.

## Scope: speech rather than general inference

hearsay is scoped to speech (`hear` for speech recognition and `say` for speech synthesis) rather than general-purpose model inference for two reasons:

- Shared audio-codec requirements on ggml: speech recognition and neural audio codecs share `GGML_MAX_NAME=128` for long audio tokenizer tensor names, batched `conv_1d` graph layout, and fused CPU snake activations (`patches/ggml/0001` to `0003`), while calling only the stable public ggml C API (`ggml.h`, `ggml-backend.h`, `ggml-alloc.h`, `gguf.h`).
- Coordinated upstream ggml base without llama.cpp churn: on the synthesis side, [ServeurpersoCom](https://github.com/ServeurpersoCom)'s TTS engines (`pockettts.cpp`, `qwentts.cpp`, `omnivoice.cpp`) are maintained by the same author against a single shared `ServeurpersoCom/ggml` submodule commit; on the recognition side, `transcribe.cpp` vendors the same upstream ggml base (`d7cb574`), bridged by hearsay's pinned archive hashes in `build.zig.zon` and `patches/ggml/`. General language model engines such as `llama.cpp` change their internal ggml headers almost daily, so keeping hearsay scoped to speech avoids ggml version drift and patch churn.

## Build

Callers can opt in by capability (`-Dhear=true` for default Parakeet speech recognition, `-Dsay=true` for default Pocket TTS speech synthesis) or by individual model family (`-Dparakeet=true`, `-Dwhisper=true`, `-Dpockettts=true`, `-Dqwentts=true`, `-Dsilero=true` for Silero VAD), compiling and linking only the directions or model families they use. Whisper adds `transcribe/whisper.h` beside `transcribe.h`, Silero `transcribe/vad.h`, and Qwen3-TTS `qwen.h` (with `QWEN_STATIC` defined). qwentts.cpp is a lazy dependency, fetched only by builds that ask for it.

With Zig 0.17.0:

```sh
zig build -Doptimize=ReleaseFast -Dhear=true -Dsay=true
zig build -Doptimize=ReleaseFast -Dhear=true -Dsay=true -Dtarget=x86_64-windows-gnu
zig build -Doptimize=ReleaseFast -Dhear=true -Dsay=true -Dtarget=aarch64-macos -Dmacos_sdk=$HOME/.local/share/macos-sdk/MacOSX.sdk
```

This builds `zig-out/lib/libhearsay.a`, installs the enabled headers (`transcribe.h` and/or `pocket.h`, with `TRANSCRIBE_STATIC` and `POCKET_STATIC` defined) into `zig-out/include/`, and builds `zig-out/bin/hearsay-bench`. An app on [gesso](https://github.com/wakamex/gesso) such as gear can depend on hearsay in `build.zig.zon`, pass `.hear = true` and `.say = true` (or specific model flags such as `.parakeet = true`, `.pockettts = true`), and link the `hearsay` artifact:

It is pinned by commit, so add it with:

```sh
zig fetch --save git+https://github.com/wakamex/hearsay#COMMIT
```

```zig
const hearsay = b.dependency("hearsay", .{
    .target = target,
    .optimize = optimize,
    .hear = true,
    .say = true,
});
mod.linkLibrary(hearsay.artifact("hearsay"));
```

Or with CMake 3.24 or newer and git (`-DHEARSAY_HEAR=ON -DHEARSAY_SAY=ON` or `-DHEARSAY_PARAKEET=ON -DHEARSAY_POCKETTTS=ON`):

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DHEARSAY_HEAR=ON -DHEARSAY_SAY=ON
cmake --build build -j
```

`hearsay-bench` (`zig-out/bin/hearsay-bench` or `build/hearsay-bench`) measures both engines:

```sh
zig-out/bin/hearsay-bench hear parakeet-tdt_ctc-110m-Q4_K_M.gguf 1 wavs.txt
zig-out/bin/hearsay-bench say pocket-tts-english_2026-09-Q8_0.gguf pocket-tts-english_2026-09-voices.gguf alba 1 texts.txt out/say
zig-out/bin/hearsay-bench qwen qwen-talker-1.7b-base-Q8_0.gguf qwen-tokenizer-12hz-Q8_0.gguf voice.wav 8 texts.txt out/qwen
```

`hear` takes Parakeet and Whisper models. For `qwen`, the voice is a speaker name for a custom-voice model, a description of the voice for a voice-design model, or for a base model a 24 kHz mono WAV to clone, with its transcript in a `.txt` file of the same name beside it (`voice.txt`).

Both run on the best device the build has, and print the recognition backend in their first line; `GGML_DISABLE_VULKAN=1` keeps them on the CPU. The voices argument can also be one voice's `.safetensors` file.

The models are [parakeet-tdt_ctc-110m-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf), the [Pocket TTS GGUFs](https://huggingface.co/Serveurperso/pocket-tts-GGUF), the [Whisper GGUFs](https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf) converted for transcribe.cpp or [whisper.cpp's `.bin` files](https://huggingface.co/ggerganov/whisper.cpp), and the [Qwen3-TTS GGUFs](https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF), a talker and the 12 Hz tokenizer.

### GPU

`-Dgpu=vulkan` adds ggml's Vulkan backend, so both engines run on the GPU where one is usable and on the CPU otherwise: with no GPU, no Vulkan driver, or `GGML_DISABLE_VULKAN` set by the app or the user before the models load. transcribe.cpp's default backend request and pockettts.cpp's default device both pick the best device. The build needs the [glslc](https://github.com/google/shaderc) shader compiler (Fedora's `glslc` package) on the `PATH` or named with `-Dglslc=PATH`; nothing is linked against a Vulkan library, so cross-builds for Windows need nothing else. ggml's shader generator is built for the host and compiles each of ggml's 155 shaders with glslc, and the shader features this glslc supports (cooperative matrices, integer dot products, bfloat16, 8-bit floats) are found when the build is configured, as ggml's CMake build does.

```sh
zig build -Doptimize=ReleaseFast -Dhear=true -Dsay=true -Dgpu=vulkan -Dtarget=x86_64-windows-gnu
```

The compiled shaders make `hearsay-bench` 46 MB instead of 2.3 MB. On an RTX 3080 under Windows 11, Parakeet ft-v1 Q4_K_M transcribes 10 s of speech in 0.20 s against 2.97 s on one thread of a Ryzen 7 3800XT, and Pocket reaches first audio in 0.01 to 0.05 s, with word error rates on 556 LibriSpeech utterances matching the CPU's to within rounding. The CMake build is CPU only, and without Silero VAD.

## Limitations

- Only the model families named above are built: Parakeet and Whisper for recognition, Pocket TTS and Qwen3-TTS for synthesis and Silero for speech detection. Other transcribe.cpp families are not.
- The GPU build supports Vulkan only, and needs glslc to build. The CMake build is CPU only and has only Parakeet and Pocket TTS.
- Speed and parity were measured on one machine, a Ryzen 7 3800XT, plus an RTX 3080 for the GPU figures.

## Development checks

`tests/smoke.sh` builds hearsay with every model family, downloads Parakeet TDT-CTC 110M, Whisper tiny.en (as a GGUF and as a whisper.cpp `.bin`), Pocket TTS and Qwen3-TTS 0.6B custom-voice Q4_K_M into `.models` (or `$HEARSAY_MODELS`) at pinned commits, checks their SHA-256, and checks that Parakeet and both Whisper files transcribe `tests/utterance.wav` and that Pocket and Qwen3-TTS speak a sentence. Extra arguments go to `zig build`, such as `tests/smoke.sh -Dgpu=vulkan`. CI runs it on Linux, Windows and macOS, the Vulkan build on Linux (where, without a GPU, it falls back to the CPU), a cross-build for Windows from Linux, and the CMake build.

## License

hearsay's own files, and the patches in `patches/transcribe/`, `patches/pockettts/` and `patches/qwentts/`, are under the MIT license in [LICENSE](LICENSE). The patches change files of transcribe.cpp, pockettts.cpp, qwentts.cpp and ggml, which stay under their own MIT licenses; each patch folder holds the license of the project it patches (`patches/*/LICENSE`).

`patches/ggml/0001` to `0003` are not hearsay's: Pascal (ServeurpersoCom), the author of pockettts.cpp, wrote them as commits [feece7e](https://github.com/ServeurpersoCom/ggml/commit/feece7e1581f752e204048af99a765a70d19bd50), [2eddaf9](https://github.com/ServeurpersoCom/ggml/commit/2eddaf94a8cdbbf1c30017d28313cd661a9da543) and [2be5947](https://github.com/ServeurpersoCom/ggml/commit/2be5947ac234d277714e824a5c7eb23880557543) in his ggml fork, [ServeurpersoCom/ggml](https://github.com/ServeurpersoCom/ggml). They are copied unchanged and are under ggml's MIT license.

`tests/utterance.wav` is utterance 1089-134686-0010 of LibriSpeech test-clean, by Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur ([LibriSpeech](https://www.openslr.org/12), 2015), under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). It was converted from FLAC to a 16-bit WAV at its original 16 kHz, without trimming.

The models are not part of hearsay. The Parakeet and Pocket TTS weights are CC-BY-4.0, and Kyutai's terms for Pocket TTS prohibit cloning a voice without consent and deceptive use; Silero VAD's and Whisper's are MIT, and Qwen3-TTS's Apache-2.0.
