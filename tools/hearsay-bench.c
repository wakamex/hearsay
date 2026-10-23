// hearsay-bench: times speech recognition and speech synthesis through the
// hearsay build and writes JSON, so its output and speed can be compared with
// the separate transcribe.cpp and pockettts.cpp builds.
//
//   hearsay-bench hear MODEL THREADS LIST
//       Transcribes each 16 kHz mono WAV named in LIST (one path per line).
//       Prints one JSON object per line: a header with the load time, then
//       {"file", "text", "total_ms"} per file.
//   hearsay-bench say MODEL VOICES VOICE THREADS TEXTS OUT_PREFIX
//       VOICES is a voices file holding VOICE, or one voice's .safetensors file.
//       Synthesizes each line of TEXTS twice with seed 42, the first pass to
//       warm up, and writes the second pass as OUT_PREFIX-<i>.pcm (s16le,
//       24 kHz). Prints one JSON object with the time to first audio, total
//       time and audio length per line.
//
// Both run on the best device the build has (the GPU in a -Dgpu=vulkan build
// where one is usable); GGML_DISABLE_VULKAN=1 keeps them to the CPU.

#if !defined(HEARSAY_PARAKEET) && !defined(HEARSAY_POCKETTTS)
#define HEARSAY_PARAKEET  1
#define HEARSAY_POCKETTTS 1
#endif

#ifdef HEARSAY_POCKETTTS
#include "pocket.h"
#endif
#ifdef HEARSAY_PARAKEET
#include "transcribe.h"
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"
#endif

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifndef _WIN32
#include <sys/resource.h>
#endif

static double now(void) {
    struct timespec t;
    timespec_get(&t, TIME_UTC);
    return (double) t.tv_sec + (double) t.tv_nsec * 1e-9;
}

static double peak_rss_mb(void) {
#ifdef _WIN32
    return -1;
#else
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
#ifdef __APPLE__
    return (double) r.ru_maxrss / (1024.0 * 1024.0);
#else
    return (double) r.ru_maxrss / 1024.0;
#endif
#endif
}

#ifdef HEARSAY_PARAKEET
static void json_string(const char * s) {
    putchar('"');
    for (; *s; s++) {
        unsigned char c = (unsigned char) *s;
        if (c == '"' || c == '\\') {
            printf("\\%c", c);
        } else if (c < 0x20) {
            printf("\\u%04x", c);
        } else {
            putchar(c);
        }
    }
    putchar('"');
}

static int hear(const char * model, int threads, const char * list) {
    struct transcribe_session_params sp;
    transcribe_session_params_init(&sp);
    sp.n_threads = threads;
    double                      t       = now();
    struct transcribe_session * session = NULL;
    transcribe_status           st      = transcribe_open(model, NULL, &sp, &session);
    if (st != TRANSCRIBE_OK) {
        fprintf(stderr, "transcribe_open: %s\n", transcribe_status_string(st));
        return 1;
    }
    printf("{\"type\": \"batch_header\", \"load_ms\": %.1f, \"backend\": \"%s\"}\n", (now() - t) * 1000,
           transcribe_model_backend(transcribe_get_model(session)));

    struct transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    rp.language = "en";

    FILE * f = fopen(list, "r");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", list);
        return 1;
    }
    char path[4096];
    while (fgets(path, sizeof path, f)) {
        path[strcspn(path, "\r\n")] = 0;
        if (!path[0]) {
            continue;
        }
        unsigned int channels = 0, rate = 0;
        drwav_uint64 frames = 0;
        float *      pcm    = drwav_open_file_and_read_pcm_frames_f32(path, &channels, &rate, &frames, NULL);
        if (!pcm || channels != 1 || rate != 16000) {
            fprintf(stderr, "%s: need a 16 kHz mono WAV\n", path);
            return 1;
        }
        t  = now();
        st = transcribe_run(session, pcm, (int) frames, &rp);
        double ms = (now() - t) * 1000;
        drwav_free(pcm, NULL);
        if (st != TRANSCRIBE_OK) {
            fprintf(stderr, "transcribe_run %s: %s\n", path, transcribe_status_string(st));
            return 1;
        }
        printf("{\"file\": ");
        json_string(path);
        printf(", \"text\": ");
        json_string(transcribe_full_text(session));
        printf(", \"total_ms\": %.1f}\n", ms);
    }
    fclose(f);
    transcribe_close(session);
    fprintf(stderr, "peak_rss_mb=%.0f\n", peak_rss_mb());
    return 0;
}
#endif

#ifdef HEARSAY_POCKETTTS
static void set_threads_env(const char * threads) {
#ifdef _WIN32
    _putenv_s("POCKET_THREADS", threads);
#else
    setenv("POCKET_THREADS", threads, 1);
#endif
}

typedef struct {
    double t0, first;
    long   n;
    int    chunks;
    FILE * pcm;
} chunk_state;

static bool on_chunk(const float * s, int n, void * user) {
    chunk_state * x = user;
    if (x->first < 0 && n > 0) {
        x->first = now() - x->t0;
    }
    x->n += n;
    x->chunks++;
    if (x->pcm) {
        for (int i = 0; i < n; i++) {
            float v = s[i] < -1 ? -1 : s[i] > 1 ? 1 : s[i];
            short q = (short) (v * 32767);
            fwrite(&q, 2, 1, x->pcm);
        }
    }
    return true;
}

static int say(const char * model, const char * voices, const char * voice_name, const char * threads,
               const char * texts, const char * out_prefix) {
    set_threads_env(threads);
    double                 t = now();
    struct pt_init_params ip;
    pt_init_default_params(&ip);
    ip.model_path              = model;
    struct pt_context * ctx    = pt_init(&ip);
    if (!ctx) {
        fprintf(stderr, "pt_init: %s\n", pt_last_error());
        return 1;
    }
    // A .safetensors voice file is loaded as it is; anything else is a voices file holding VOICE.
    size_t            len   = strlen(voices);
    struct pt_voice * voice = len > 12 && !strcmp(voices + len - 12, ".safetensors") ? pt_voice_load(ctx, voices, 0) :
                                                                                       pt_voice_named(ctx, voices, voice_name);
    if (!voice) {
        fprintf(stderr, "pt_voice_named: %s\n", pt_last_error());
        return 1;
    }
    printf("{\"load\": %.3f, \"rows\": [", now() - t);
    char line[8192];
    int  first_row = 1;
    for (int pass = 0; pass < 2; pass++) {
        FILE * f = fopen(texts, "r");
        if (!f) {
            fprintf(stderr, "cannot open %s\n", texts);
            return 1;
        }
        for (int i = 0; fgets(line, sizeof line, f); i++) {
            line[strcspn(line, "\r\n")] = 0;
            struct pt_tts_params p;
            pt_tts_default_params(&p);
            p.text     = line;
            p.voice    = voice;
            p.on_chunk = on_chunk;
            p.seed     = 42;
            char path[4096];
            snprintf(path, sizeof path, "%s-%d.pcm", out_prefix, i);
            chunk_state x = { now(), -1, 0, 0, pass ? fopen(path, "wb") : NULL };
            p.user_data   = &x;
            if (pt_synthesize(ctx, &p, NULL)) {
                fprintf(stderr, "pt_synthesize: %s\n", pt_last_error());
                return 1;
            }
            double total = now() - x.t0;
            if (x.pcm) {
                fclose(x.pcm);
            }
            printf("%s{\"pass\": %d, \"i\": %d, \"ttfa\": %.4f, \"total\": %.4f, \"audio\": %.3f, \"chunks\": %d}",
                   first_row ? "" : ", ", pass, i, x.first, total, (double) x.n / pt_sample_rate(ctx), x.chunks);
            first_row = 0;
        }
        fclose(f);
    }
    printf("], \"rss_mb\": %.1f}\n", peak_rss_mb());
    pt_voice_free(voice);
    pt_free(ctx);
    return 0;
}
#endif

int main(int argc, char ** argv) {
#ifdef HEARSAY_PARAKEET
    if (argc == 5 && strcmp(argv[1], "hear") == 0) {
        return hear(argv[2], atoi(argv[3]), argv[4]);
    }
#endif
#ifdef HEARSAY_POCKETTTS
    if (argc == 8 && strcmp(argv[1], "say") == 0) {
        return say(argv[2], argv[3], argv[4], argv[5], argv[6], argv[7]);
    }
#endif
    bool help = argc == 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0);
    fprintf(help ? stdout : stderr,
            "usage: %s hear MODEL THREADS LIST\n"
            "       %s say MODEL VOICES VOICE THREADS TEXTS OUT_PREFIX\n",
            argv[0], argv[0]);
    return help ? 0 : 2;
}
