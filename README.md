# hearsay

hearsay is the local speech engine for [gear](../gear): it hears, turning speech into text, and says, turning text into speech, in one small native build, on the CPU or, optionally, on the GPU through [Vulkan](https://www.vulkan.org/). It compiles only the parts of two upstream projects that gear uses against a single copy of the [ggml](https://github.com/ggml-org/ggml) tensor library:

- Hearing: NVIDIA's [Parakeet TDT-CTC 110M](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) through [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp), with only its Parakeet family built, and optionally [Silero VAD](https://github.com/snakers4/silero-vad) from the same library, for telling speech from other sound.
- Saying: Kyutai's [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) through [pockettts.cpp](https://github.com/ServeurpersoCom/pockettts.cpp), which streams audio as it is generated, without its command-line tool and HTTP server.

Both projects are fetched at pinned commits and changed only by the patch files in `patches/`. The GPU build also fetches the Khronos [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) and [SPIRV-Headers](https://github.com/KhronosGroup/SPIRV-Headers) at pinned releases.

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
| `patches/transcribe/0001-register-only-built-families.patch` | transcribe.cpp lists all 21 model families in one table, which would link every family. hearsay's copy lists only the families the build includes, Parakeet and Silero VAD. |
| `patches/transcribe/0002-stdio-file-io.patch` | Replaces `std::ifstream` and `std::filesystem::status` in model loading with `ggml_fopen`, 64-bit `fseeko` / `_fseeki64` and `stat` / `GetFileAttributesW`, avoiding `std::locale` and `std::filesystem` in the linked binary while preserving UTF-8 path support on Windows. |
| `patches/transcribe/0003-silero-stdio-file-io.patch` | The same change as 0002 for Silero VAD's reader of whisper.cpp-format `.bin` files, which used `std::ifstream` and the `path_from_utf8` that 0002 removes. |
| `patches/ggml/0001` to `0003` | The three commits pockettts.cpp's ggml fork ([ServeurpersoCom/ggml](https://github.com/ServeurpersoCom/ggml)) adds to upstream ggml: a fused snake activation on the CPU and two `conv_1d` fixes for batched input. Both forks start from the same upstream ggml (`d7cb574` and its successor `ffa4e8b`, which changes only a release script), so these apply on top of transcribe.cpp's patched ggml. |
| `patches/pockettts/0001-thread-count-from-POCKET_THREADS.patch` | Reads `POCKET_THREADS` for the CPU thread count instead of always using one thread per physical core, and skips `ggml_backend_load_all()` when `POCKET_STATIC` is defined so static builds do not scan the working directory for dynamic backend plugins. |
| `patches/pockettts/0002-comma-before-every-chunk.patch` | Puts ", " before the text of every chunk. pockettts.cpp drops the audio of each chunk's first frame (80 ms), which decodes to a click, and when the voice starts at once a short first word goes with it: on 480 sentences opening with "I" (30 sentences, 16 seeds), the "I" was lost from 4.4% at the start of a text and 8.1% at the start of a later chunk, and from none of either with the comma, which adds about 0.08 s of quiet before each chunk's speech. A comma added to the text by the caller does not do this: Pocket can cut a long first sentence at it and voice it as a chunk of its own. |

`src/no-bin-loaders.cpp` stands in for the functions transcribe.cpp's shared loader calls for whisper.cpp-format `.bin` files and for the development tensor dumper in `transcribe-debug.cpp`. Whisper's `.bin` loader returns `TRANSCRIBE_ERR_NOT_IMPLEMENTED`, and so does Silero VAD's unless `-Dsilero` builds that family, whose own loader then reads them; `transcribe-bin-loader.cpp` and `transcribe-debug.cpp` are omitted. Parakeet's model code links the Sortformer diarizer for its multi-speaker variant, so that family is compiled as is.

`src/vulkan-loader.cpp`, in GPU builds only, opens the Vulkan loader (`vulkan-1.dll` or `libvulkan.so.1`) when the program starts instead of linking it, and supplies the four Vulkan functions ggml calls directly by forwarding them to the loader's. Without a loader it sets `GGML_DISABLE_VULKAN`, so ggml registers only the CPU and the program runs there.

Build settings: ggml is built with transcribe.cpp's settings, without BLAS or OpenMP and with [llamafile](https://github.com/Mozilla-Ocho/llamafile)'s matrix kernels, which make both engines faster on this CPU. `GGML_MAX_NAME` is 128 for every target, as pockettts.cpp needs; it changes the layout of ggml's tensor struct, so it must match everywhere.

## Scope: speech rather than general inference

hearsay is scoped to speech (`hear` for speech recognition and `say` for speech synthesis) rather than general-purpose model inference for two reasons:

- Shared audio-codec requirements on ggml: speech recognition and neural audio codecs share `GGML_MAX_NAME=128` for long audio tokenizer tensor names, batched `conv_1d` graph layout, and fused CPU snake activations (`patches/ggml/0001` to `0003`), while calling only the stable public ggml C API (`ggml.h`, `ggml-backend.h`, `ggml-alloc.h`, `gguf.h`).
- Coordinated upstream ggml base without llama.cpp churn: on the synthesis side, [ServeurpersoCom](https://github.com/ServeurpersoCom)'s TTS engines (`pockettts.cpp`, `qwentts.cpp`, `omnivoice.cpp`) are maintained by the same author against a single shared `ServeurpersoCom/ggml` submodule commit; on the recognition side, `transcribe.cpp` vendors the same upstream ggml base (`d7cb574`), bridged by hearsay's pinned archive hashes in `build.zig.zon` and `patches/ggml/`. General language model engines such as `llama.cpp` change their internal ggml headers almost daily, so keeping hearsay scoped to speech avoids ggml version drift and patch churn.

## Build

Callers can opt in by capability (`-Dhear=true` for default Parakeet speech recognition, `-Dsay=true` for default Pocket TTS speech synthesis) or by individual model family (`-Dparakeet=true`, `-Dpockettts=true`, `-Dsilero=true` for Silero VAD), compiling and linking only the directions or model families they use. With Silero, `transcribe/vad.h` is installed beside `transcribe.h`.

With Zig 0.17.0:

```sh
zig build -Doptimize=ReleaseFast -Dhear=true -Dsay=true
zig build -Doptimize=ReleaseFast -Dhear=true -Dsay=true -Dtarget=x86_64-windows-gnu
zig build -Doptimize=ReleaseFast -Dhear=true -Dsay=true -Dtarget=aarch64-macos -Dmacos_sdk=$HOME/.local/share/macos-sdk/MacOSX.sdk
```

This builds `zig-out/lib/libhearsay.a`, installs the enabled headers (`transcribe.h` and/or `pocket.h`, with `TRANSCRIBE_STATIC` and `POCKET_STATIC` defined) into `zig-out/include/`, and builds `zig-out/bin/hearsay-bench`. An app on [gesso](https://github.com/wakamex/gesso) such as gear can depend on hearsay in `build.zig.zon`, pass `.hear = true` and `.say = true` (or specific model flags such as `.parakeet = true`, `.pockettts = true`), and link the `hearsay` artifact:

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
```

Both run on the best device the build has, and print the recognition backend in their first line; `GGML_DISABLE_VULKAN=1` keeps them on the CPU. The voices argument can also be one voice's `.safetensors` file.

The models are [parakeet-tdt_ctc-110m-Q4_K_M.gguf](https://huggingface.co/handy-computer/parakeet-tdt_ctc-110m-gguf) and the [Pocket TTS GGUFs](https://huggingface.co/Serveurperso/pocket-tts-GGUF).

### GPU

`-Dgpu=vulkan` adds ggml's Vulkan backend, so both engines run on the GPU where one is usable and on the CPU otherwise: with no GPU, no Vulkan driver, or `GGML_DISABLE_VULKAN` set by the app or the user before the models load. transcribe.cpp's default backend request and pockettts.cpp's default device both pick the best device. The build needs the [glslc](https://github.com/google/shaderc) shader compiler (Fedora's `glslc` package) on the `PATH` or named with `-Dglslc=PATH`; nothing is linked against a Vulkan library, so cross-builds for Windows need nothing else. ggml's shader generator is built for the host and compiles each of ggml's 155 shaders with glslc, and the shader features this glslc supports (cooperative matrices, integer dot products, bfloat16, 8-bit floats) are found when the build is configured, as ggml's CMake build does.

```sh
zig build -Doptimize=ReleaseFast -Dhear=true -Dsay=true -Dgpu=vulkan -Dtarget=x86_64-windows-gnu
```

The compiled shaders make `hearsay-bench` 46 MB instead of 2.3 MB. On an RTX 3080 under Windows 11, Parakeet ft-v1 Q4_K_M transcribes 10 s of speech in 0.20 s against 2.97 s on one thread of a Ryzen 7 3800XT, and Pocket reaches first audio in 0.01 to 0.05 s, with word error rates on the 556 LibriSpeech utterances matching the CPU's to within rounding; [gear's README](../gear/README.md) has the full comparison. The CMake build is CPU only, and without Silero VAD.

## Licenses

transcribe.cpp, pockettts.cpp and ggml are MIT-licensed. The Parakeet and Pocket TTS weights are CC-BY-4.0, and Kyutai's terms for Pocket TTS prohibit cloning a voice without consent and deceptive use.
