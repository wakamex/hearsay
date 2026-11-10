# hearsay

hearsay is the local speech engine for gear, a voice front-end for coding agents: it hears, turning speech into text, and says, turning text into speech, in one small native build, on the CPU or, optionally, on the GPU through [Vulkan](https://www.vulkan.org/). It compiles only the parts of three upstream projects that its callers ask for against a single copy of the [ggml](https://github.com/ggml-org/ggml) tensor library:

- Hearing: NVIDIA's Parakeet models, such as [Parakeet TDT-CTC 110M](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m), through [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp), with only its Parakeet family built, and optionally OpenAI's [Whisper](https://github.com/openai/whisper) models from the same library, from GGUF or whisper.cpp `.bin` files, and [Silero VAD](https://github.com/snakers4/silero-vad), for telling speech from other sound.
- Saying: Kyutai's [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) through [pockettts.cpp](https://github.com/ServeurpersoCom/pockettts.cpp), which streams audio as it is generated, without its command-line tool and HTTP server; through the same library, Gradium's [Phonon](https://huggingface.co/gradium/phonon-31m-en), a smaller descendant of Pocket TTS, with the voices of [phonon-voices](https://huggingface.co/gradium/phonon-voices); and optionally Alibaba's [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (base models that clone a voice from a recording, custom-voice models with named speakers, and voice-design models that take a description of the voice) through [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp), by the same author.

The projects are fetched at pinned commits and changed only by the patch files in `patches/`. The GPU build also fetches the Khronos [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) and [SPIRV-Headers](https://github.com/KhronosGroup/SPIRV-Headers) at pinned releases.

## Result

The combined build matches the two projects built separately and is less than a third of their combined size. One benchmark program holding both engines and ggml is 2.3 MB stripped, against 4.7 MB for transcribe.cpp's command-line program and 2.4 MB for pockettts.cpp's. Every Parakeet transcript is identical to upstream transcribe.cpp's, and synthesis was bit-identical to upstream pockettts.cpp's when built with the same ggml kernels, before `patches/pockettts/0002` put a comma before every chunk. With the kernels hearsay uses, Pocket gets faster: 0.22 s to first audio instead of 0.29 s on one thread, and 0.50 CPU-seconds per second of audio instead of 0.63. Recognition costs 1.4% more CPU than upstream transcribe.cpp.

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

## Whisper and Qwen3-TTS against the separate builds

These rows compare `-Dwhisper` and `-Dqwentts` with upstream transcribe.cpp and qwentts.cpp built on their own at the same commits, each with its own ggml, on the same idle Ryzen 7 3800XT under WSL2 with 8 threads. Whisper transcribes the same 57 LibriSpeech utterances (460 s) in English, upstream through `transcribe-cli --batch`. Qwen3-TTS clones a voice from `tests/utterance.wav`, resampled to 24 kHz, with its transcript as the reference text, and synthesizes gear's 8 agent-style replies with seed 42, both builds through `hearsay-bench qwen` (upstream's linked against its own libraries), timing the second pass. All models are Q8_0, the Qwen3-TTS talkers with the Q8_0 12 Hz tokenizer.

| Model | Build | Wall time for the 57 utterances | Speed vs real time | First audio, median (worst) | Peak memory |
| --- | --- | ---: | ---: | ---: | ---: |
| Whisper base | transcribe.cpp alone | 37 s | | | 0.19 GB |
| Whisper base | hearsay | 39 s | | | 0.20 GB |
| Whisper large-v3-turbo | transcribe.cpp alone | 481 s | | | 1.02 GB |
| Whisper large-v3-turbo | hearsay | 534 s | | | 1.01 GB |
| Qwen3-TTS 1.7B base | qwentts.cpp alone | | 0.43x | 1.64 s (2.00 s) | 6.0 GB |
| Qwen3-TTS 1.7B base | hearsay | | 0.47x | 1.79 s (1.88 s) | 6.0 GB |
| Qwen3-TTS 0.6B base | qwentts.cpp alone | | 0.58x | 1.14 s (1.18 s) | 3.9 GB |
| Qwen3-TTS 0.6B base | hearsay | | 0.59x | 1.31 s (1.38 s) | 3.9 GB |

Whisper times are the mean of two alternating runs per build, which agreed to within 2%. Whisper is 5% (base) to 11% (large-v3-turbo) slower in hearsay; the cause is not yet known. Neither engine runs in real time on this CPU at these sizes.

Parity checks:

- Whisper: large-v3-turbo's transcripts match upstream's on 55 of the 57 utterances and base's on 50. Each build gives the same transcripts on every run except for one utterance with base, which both builds garble and which changes between runs because Whisper's temperature fallback samples from a random seed. The differences are single words, such as "Hester Prin" for upstream's "Hester print", or punctuation. hearsay links upstream's code unchanged and matches the build settings: with hearsay-bench linked against upstream's own library, the transcripts are upstream's. The differences come from how hearsay compiles transcribe.cpp's own code for the native CPU, where upstream compiles it for baseline x86-64: turning off fused multiply-adds there removes two of the six differences on base. The `conv_1d` patches, `GGML_MAX_NAME` and the compiler (clang or GCC for upstream) change nothing.
- Whisper `.bin`: whisper.cpp's tiny.en `.bin` file transcribes `tests/utterance.wav` as upstream does.
- Qwen3-TTS: the same seed gives different audio from the two builds, as with Pocket: each sampled frame feeds the next, and qwentts.cpp's own ggml (0.27.0, built with OpenMP) rounds differently from hearsay's. Both are valid samples of the same text: Whisper large-v3-turbo transcribes the 8 replies from hearsay's build with 2.7% (0.6B) and 3.6% (1.7B) word errors against the texts, and from upstream's with 4.1% and 3.2%, nearly all of them number and spelling forms such as "11" for "eleven" and "Mac OS".

Size: a `hearsay-bench` with only Whisper is 2.1 MB stripped, with only Qwen3-TTS 2.1 MB, and with Parakeet, Whisper, Pocket TTS and Qwen3-TTS 3.0 MB.

## Phonon against Gradium's runtime

[Phonon](https://huggingface.co/gradium/phonon-31m-en) is Gradium's 31-million-parameter English model built on Pocket TTS: the same flow LM and Mimi decoder at half the width (512), with about 100 voices in [phonon-voices](https://huggingface.co/gradium/phonon-voices). `patches/pockettts/0003` runs it in pockettts.cpp from Gradium's own files, and these rows compare it with Gradium's runtime, the `ptts` 0.6.0 Python package ([xn-ptts](https://github.com/gradium-ai/xn-ptts)), on the Ryzen 7 3800XT under WSL2. Both read the same `model.q8.gguf` (the flow transformer in Q8_0, the rest in BF16), use the Delia voice and seed 42, and synthesize gear's 8 replies, timing the second pass; `ptts` streams through its Python API with `lang="none"`. Each figure is the mean of two alternating runs, which agreed to within 3%.

| Threads | Build | Speed vs real time | First audio, median (worst) | CPU-seconds per second of audio | Peak memory |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Gradium's runtime | 4.9x | 0.08 s (0.09 s) | 0.28 | 0.32 GB |
| 1 | hearsay | 2.0x | 0.08 s (0.10 s) | 0.49 | 0.16 GB |
| 2 | Gradium's runtime | 6.6x | 0.05 s (0.06 s) | 0.35 | 0.34 GB |
| 2 | hearsay | 3.4x | 0.05 s (0.06 s) | 0.55 | 0.16 GB |
| 8 | Gradium's runtime | 8.5x | 0.04 s (0.05 s) | 0.81 | 0.53 GB |
| 8 | hearsay | 6.1x | 0.03 s (0.06 s) | 1.11 | 0.16 GB |

hearsay reaches first audio as fast and with half the memory, but needs 1.4 to 1.8 times the CPU. Its Mimi decoder takes most of it: 31 ms of each 80 ms frame on one thread against 7.5 ms for the flow LM step, and pockettts.cpp's decoder costs the same for Pocket TTS (33 ms per frame). On an RTX 3080 under Windows 11 the Vulkan build runs at 28x real time with first audio in 9 to 11 ms, after a first synthesis of 20 s in which the driver compiles the shaders.

Parity checks:

- At temperature 0, where neither runtime draws noise, the first frame of each of the 8 replies correlates with Gradium's at 0.9998 (33 to 35 dB signal to difference, against 36 dB between Gradium's own Q8_0 and full-precision runs), and every reply ends on the same frame or one frame apart. Later frames drift apart as they do between Gradium's Q8_0 and full-precision runs, since each frame feeds the next.
- Parakeet TDT-CTC 110M transcribes all 8 replies from the CPU and Vulkan builds as their texts, apart from number formats and one "test tasks" for "tests pass" with seed 42, which six other seeds and temperature 0 read correctly.
- Pocket TTS output is byte-identical with the patch and without it.

What the patch changes for Phonon, all of it where the GGUF has no Pocket TTS keys:

- The hyperparameters come from `config.json`, and the tokenizer from `tokenizer.model`, the same SentencePiece Unigram model with byte fallback that pockettts.cpp already encodes.
- Phonon's GELU is the exact one and its time embedding is normalized by the mean square, where Pocket uses the tanh approximation and the variance.
- The summed conditioners (`num_speakers` 1 and `duration_delta` 0, Gradium's defaults) are a constant the reference adds to every generated frame's input, so they load as an `input_linear` bias.
- A voice file's `speaker_wavs` latents go through the checkpoint's speaker projection and are prefilled as the voice prompt; Phonon has no Mimi encoder, so `pt_voice_from_audio` refuses.
- Generation follows Gradium's runtime: temperature 0.3, the EOS threshold from `config.json` (4), EOS counted from the first frame, one frame fewer after it, and no frame dropped at the start of a chunk, so `patches/pockettts/0002`'s comma is not added either. `pt_tts_params.eos_threshold` now defaults to NAN for the model's own threshold, which leaves Pocket at -4.

## Changes to upstream

| Patch | Why |
| --- | --- |
| `patches/transcribe/0001-register-only-built-families.patch` | transcribe.cpp lists all 21 model families in one table, which would link every family. hearsay's copy lists only the families the build includes: Parakeet, Whisper and Silero VAD. |
| `patches/transcribe/0002-stdio-file-io.patch` | Replaces `std::ifstream` and `std::filesystem::status` in model loading with `ggml_fopen`, 64-bit `fseeko` / `_fseeki64` and `stat` / `GetFileAttributesW`, avoiding `std::locale` and `std::filesystem` in the linked binary while preserving UTF-8 path support on Windows. |
| `patches/transcribe/0003-silero-stdio-file-io.patch` | The same change as 0002 for Silero VAD's reader of whisper.cpp-format `.bin` files, which used `std::ifstream` and the `path_from_utf8` that 0002 removes. |
| `patches/transcribe/0004-whisper-bin-stdio-file-io.patch` | The same change as 0002 for the reader of whisper.cpp-format `.bin` files that Whisper uses, `transcribe-bin-loader.cpp`. |
| `patches/transcribe/0005-parakeet-joint-scores.patch` | Adds four fields to `struct transcribe_token`, read by `transcribe_get_token` and `transcribe_batch_get_token`: the joint network's raw scores at the Parakeet TDT decoder step that emitted the token, before any softmax. They are the token's own score (`logit`), blank's (`blank_logit`), the best other non-blank token's (`runner_up_logit`) and the log-sum-exp over every token class, blank included (`logsumexp`), so `logit - logsumexp` is the token's log-probability. gear tells noise that Parakeet writes words into, such as a cough heard as "um", from speech with them: the margin of the token over blank carries information the reported confidence `p` does not. They are NAN for other families and for CTC decoding. The fields sit after `text`, so a caller whose `struct_size` ends there gets the struct as before; this layout is hearsay's, and differs from the one `include/transcribe.abihash` records. They cost one more pass over the token scores per emitted token, the same work as the confidence `p`. |
| `patches/ggml/0001` to `0003` | The three commits pockettts.cpp's ggml fork ([ServeurpersoCom/ggml](https://github.com/ServeurpersoCom/ggml)) adds to upstream ggml, by Pascal and copied unchanged (see [License](#license)): a fused snake activation on the CPU and two `conv_1d` fixes for batched input. Both forks start from the same upstream ggml (`d7cb574` and its successor `ffa4e8b`, which changes only a release script), so these apply on top of transcribe.cpp's patched ggml. |
| `patches/pockettts/0001-thread-count-from-POCKET_THREADS.patch` | Reads `POCKET_THREADS` for the CPU thread count instead of always using one thread per physical core, and skips `ggml_backend_load_all()` when `POCKET_STATIC` is defined so static builds do not scan the working directory for dynamic backend plugins. |
| `patches/qwentts/0001-thread-count-from-QWEN_THREADS.patch` | The same change as `patches/pockettts/0001` for qwentts.cpp, with `QWEN_THREADS` and `QWEN_STATIC`. |
| `patches/pockettts/0002-comma-before-every-chunk.patch` | Puts ", " before the text of every chunk. pockettts.cpp drops the audio of each chunk's first frame (80 ms), which decodes to a click, and when the voice starts at once a short first word goes with it: on 480 sentences opening with "I" (30 sentences, 16 seeds), the "I" was lost from 4.4% at the start of a text and 8.1% at the start of a later chunk, and from none of either with the comma, which adds about 0.08 s of quiet before each chunk's speech. A comma added to the text by the caller does not do this: Pocket can cut a long first sentence at it and voice it as a chunk of its own. Phonon models keep every frame, as Gradium's runtime does, and get no comma. |
| `patches/pockettts/0003-gradium-phonon.patch` | Loads Gradium's Phonon models as published (`model.q8.gguf` with `config.json` and `tokenizer.model` beside it) and their speaker latent voices, and generates as Gradium's runtime does. See [Phonon](#phonon-against-gradiums-runtime). |

transcribe.cpp and pockettts.cpp are each built from one copy of their sources, in the build cache, with every patched file in place: their sources include each other's headers from beside them, so a patched header has to sit where an unpatched one would otherwise be found.

`src/no-bin-loaders.cpp` stands in for the functions transcribe.cpp's shared loader calls for whisper.cpp-format `.bin` files and for the development tensor dumper in `transcribe-debug.cpp`. Whisper's and Silero VAD's `.bin` loaders return `TRANSCRIBE_ERR_NOT_IMPLEMENTED` unless `-Dwhisper` or `-Dsilero` builds that family, whose own loader then reads them; `transcribe-debug.cpp` is omitted, and so is `transcribe-bin-loader.cpp` without Whisper. Parakeet's model code links the Sortformer diarizer for its multi-speaker variant, so that family is compiled as is.

`src/vulkan-loader.cpp`, in GPU builds only, opens the Vulkan loader (`vulkan-1.dll` or `libvulkan.so.1`) when the program starts instead of linking it, and supplies the four Vulkan functions ggml calls directly by forwarding them to the loader's. Without a loader it sets `GGML_DISABLE_VULKAN`, so ggml registers only the CPU and the program runs there.

qwentts.cpp pins its author's ggml fork at [cc5df49](https://github.com/ServeurpersoCom/ggml/commit/cc5df49a6b878db2b68b9a7b45976e279594627d), upstream ggml 0.27.0 with the fused snake activation (`patches/ggml/0001`) and a SYCL fix. It calls nothing that ggml 0.26.0 lacks: the public headers gained only two scheduler callbacks between them, which it does not use. So it builds on hearsay's ggml unchanged. Its sources are copied into the build cache with the patched `backend.h`, because they find their headers beside them and several share names with pockettts.cpp's (`backend.h`, `debug.h`, `weight-ctx.h`).

Build settings: ggml is built with transcribe.cpp's settings, without BLAS or OpenMP and with [llamafile](https://github.com/Mozilla-Ocho/llamafile)'s matrix kernels, which make both engines faster on this CPU. `GGML_MAX_NAME` is 128 for every target, as pockettts.cpp needs; it changes the layout of ggml's tensor struct, so it must match everywhere.

## Scope: speech rather than general inference

hearsay is scoped to speech (`hear` for speech recognition and `say` for speech synthesis) rather than general-purpose model inference for two reasons:

- Shared audio-codec requirements on ggml: speech recognition and neural audio codecs share `GGML_MAX_NAME=128` for long audio tokenizer tensor names, batched `conv_1d` graph layout, and fused CPU snake activations (`patches/ggml/0001` to `0003`), while calling only the stable public ggml C API (`ggml.h`, `ggml-backend.h`, `ggml-alloc.h`, `gguf.h`).
- Coordinated upstream ggml base without llama.cpp churn: on the synthesis side, [ServeurpersoCom](https://github.com/ServeurpersoCom)'s TTS engines (`pockettts.cpp`, `qwentts.cpp`, `omnivoice.cpp`) are maintained by the same author against a single shared `ServeurpersoCom/ggml` submodule commit; on the recognition side, `transcribe.cpp` vendors the same upstream ggml base (`d7cb574`), bridged by hearsay's pinned archive hashes in `build.zig.zon` and `patches/ggml/`. General language model engines such as `llama.cpp` change their internal ggml headers almost daily, so keeping hearsay scoped to speech avoids ggml version drift and patch churn.

## Build

Callers can opt in by capability (`-Dhear=true` for default Parakeet speech recognition, `-Dsay=true` for default Pocket TTS speech synthesis) or by individual model family (`-Dparakeet=true`, `-Dwhisper=true`, `-Dpockettts=true`, `-Dqwentts=true`, `-Dsilero=true` for Silero VAD), compiling and linking only the directions or model families they use. Phonon comes with Pocket TTS, in the same library (`pocket.h`): `pt_init` loads a Phonon model when its GGUF has no Pocket TTS keys, and `pt_voice_load` then reads a voice of phonon-voices. Whisper adds `transcribe/whisper.h` beside `transcribe.h`, Silero `transcribe/vad.h`, and Qwen3-TTS `qwen.h` (with `QWEN_STATIC` defined). qwentts.cpp is a lazy dependency, fetched only by builds that ask for it.

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
zig-out/bin/hearsay-bench say phonon-31m-en/model.q8.gguf phonon-31m-en/delia.safetensors delia 2 texts.txt out/phonon
zig-out/bin/hearsay-bench qwen qwen-talker-1.7b-base-Q8_0.gguf qwen-tokenizer-12hz-Q8_0.gguf voice.wav 8 texts.txt out/qwen
```

`hear` takes Parakeet and Whisper models, and prints each token as `[id, p, logit, blank_logit, runner_up_logit, logsumexp]`, the last four null where the model does not report them. For `qwen`, the voice is a speaker name for a custom-voice model, a description of the voice for a voice-design model, or for a base model a 24 kHz mono WAV to clone, with its transcript in a `.txt` file of the same name beside it (`voice.txt`).

Both run on the best device the build has, and print the recognition backend in their first line; `GGML_DISABLE_VULKAN=1` keeps them on the CPU. The voices argument can also be one voice's `.safetensors` file.

The models are [parakeet-tdt_ctc-110m-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf), the [Pocket TTS GGUFs](https://huggingface.co/Serveurperso/pocket-tts-GGUF), Phonon's `model.q8.gguf`, `config.json` and `tokenizer.model` from [phonon-31m-en](https://huggingface.co/gradium/phonon-31m-en), in one folder, with a voice file from phonon-voices (Delia is `voices/nGzMHMPAWGaJFHOo.safetensors`), the [Whisper GGUFs](https://huggingface.co/handy-computer/whisper-large-v3-turbo-gguf) converted for transcribe.cpp or [whisper.cpp's `.bin` files](https://huggingface.co/ggerganov/whisper.cpp), and the [Qwen3-TTS GGUFs](https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF), a talker and the 12 Hz tokenizer.

### GPU

`-Dgpu=vulkan` adds ggml's Vulkan backend, so both engines run on the GPU where one is usable and on the CPU otherwise: with no GPU, no Vulkan driver, or `GGML_DISABLE_VULKAN` set by the app or the user before the models load. transcribe.cpp's default backend request and pockettts.cpp's default device both pick the best device. The build needs the [glslc](https://github.com/google/shaderc) shader compiler (Fedora's `glslc` package) on the `PATH` or named with `-Dglslc=PATH`; nothing is linked against a Vulkan library, so cross-builds for Windows need nothing else. ggml's shader generator is built for the host and compiles each of ggml's 155 shaders with glslc, and the shader features this glslc supports (cooperative matrices, integer dot products, bfloat16, 8-bit floats) are found when the build is configured, as ggml's CMake build does.

```sh
zig build -Doptimize=ReleaseFast -Dhear=true -Dsay=true -Dgpu=vulkan -Dtarget=x86_64-windows-gnu
```

The compiled shaders make `hearsay-bench` 46 MB instead of 2.3 MB. On an RTX 3080 under Windows 11, Parakeet ft-v1 Q4_K_M transcribes 10 s of speech in 0.20 s against 2.97 s on one thread of a Ryzen 7 3800XT, and Pocket reaches first audio in 0.01 to 0.05 s, with word error rates on 556 LibriSpeech utterances matching the CPU's to within rounding. The CMake build is CPU only, and without Silero VAD.

## Limitations

- Only the model families named above are built: Parakeet and Whisper for recognition, Pocket TTS, Phonon and Qwen3-TTS for synthesis and Silero for speech detection. Other transcribe.cpp families are not.
- The GPU build supports Vulkan only, and needs glslc to build. The CMake build is CPU only and has only Parakeet and Pocket TTS, with Phonon.
- Speed and parity were measured on one machine, a Ryzen 7 3800XT, plus an RTX 3080 for the GPU figures.

## Development checks

`tests/smoke.sh` builds hearsay with every model family, downloads Parakeet TDT-CTC 110M, Whisper tiny.en (as a GGUF and as a whisper.cpp `.bin`), Pocket TTS, Phonon 31M with the Delia voice and Qwen3-TTS 0.6B custom-voice Q4_K_M into `.models` (or `$HEARSAY_MODELS`) at pinned commits, checks their SHA-256, and checks that Parakeet and both Whisper files transcribe `tests/utterance.wav` and that Pocket, Phonon and Qwen3-TTS speak a sentence. Extra arguments go to `zig build`, such as `tests/smoke.sh -Dgpu=vulkan`. CI runs it on Linux, Windows and macOS, the Vulkan build on Linux (where, without a GPU, it falls back to the CPU), a cross-build for Windows from Linux, and the CMake build.

## License

hearsay's own files, and the patches in `patches/transcribe/`, `patches/pockettts/` and `patches/qwentts/`, are under the MIT license in [LICENSE](LICENSE). The patches change files of transcribe.cpp, pockettts.cpp, qwentts.cpp and ggml, which stay under their own MIT licenses; each patch folder holds the license of the project it patches (`patches/*/LICENSE`).

`patches/ggml/0001` to `0003` are not hearsay's: Pascal (ServeurpersoCom), the author of pockettts.cpp, wrote them as commits [feece7e](https://github.com/ServeurpersoCom/ggml/commit/feece7e1581f752e204048af99a765a70d19bd50), [2eddaf9](https://github.com/ServeurpersoCom/ggml/commit/2eddaf94a8cdbbf1c30017d28313cd661a9da543) and [2be5947](https://github.com/ServeurpersoCom/ggml/commit/2be5947ac234d277714e824a5c7eb23880557543) in his ggml fork, [ServeurpersoCom/ggml](https://github.com/ServeurpersoCom/ggml). They are copied unchanged and are under ggml's MIT license.

`tests/utterance.wav` is utterance 1089-134686-0010 of LibriSpeech test-clean, by Vassil Panayotov, Guoguo Chen, Daniel Povey and Sanjeev Khudanpur ([LibriSpeech](https://www.openslr.org/12), 2015), under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/). It was converted from FLAC to a 16-bit WAV at its original 16 kHz, without trimming.

The models are not part of hearsay. The Parakeet and Pocket TTS weights are CC-BY-4.0, and Kyutai's terms for Pocket TTS prohibit cloning a voice without consent and deceptive use; Silero VAD's and Whisper's are MIT, and Phonon's, its voices' and Qwen3-TTS's Apache-2.0.
