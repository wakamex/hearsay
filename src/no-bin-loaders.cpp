// transcribe.cpp's shared loader sends whisper.cpp-format .bin files to the
// Whisper and Silero VAD families. Unless -Dwhisper or -Dsilero builds a
// family, whose own loader then reads them, its entry point reports that .bin
// files are not supported instead of pulling the family into the build.
//
// The per-stage tensor dumper in transcribe-debug.cpp is also stubbed out so
// production builds do not pull in std::ofstream and std::locale formatting.

#include "transcribe.h"
#include "transcribe-debug.h"

namespace transcribe {
#ifndef HEARSAY_WHISPER
namespace whisper {
transcribe_status load_from_bin(const char *, const transcribe_model_load_params *, transcribe_model **) {
    return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
}
}  // namespace whisper
#endif

#ifndef HEARSAY_SILERO
namespace silero_vad {
transcribe_status load_from_bin(const char *, const transcribe_model_load_params *, transcribe_model **) {
    return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
}
}  // namespace silero_vad
#endif

namespace debug {
bool         init() { return false; }
bool         enabled() { return false; }
const char * dump_dir() { return nullptr; }
bool         validation_hooks_enabled() { return false; }
bool         dump_all_blocks_requested() { return false; }
const char * dump_sub_blocks_spec() { return nullptr; }
void         push_name_prefix(const char *) {}
void         pop_name_prefix() {}
void         mark_tensor_for_dump(ggml_tensor *) {}
void         dump_tensor(const char *, const ggml_tensor *, const char *) {}
void         dump_host_f32(const char *, const float *, long long, const long long *, int, const char *) {}
}  // namespace debug
}  // namespace transcribe

