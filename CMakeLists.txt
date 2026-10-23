cmake_minimum_required(VERSION 3.24)
project(hearsay LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF)

option(HEARSAY_HEAR      "Include default speech recognition (enables HEARSAY_PARAKEET)" OFF)
option(HEARSAY_SAY       "Include default speech synthesis (enables HEARSAY_POCKETTTS)"  OFF)
option(HEARSAY_PARAKEET  "Include Parakeet speech recognition (transcribe.cpp)" OFF)
option(HEARSAY_POCKETTTS "Include Pocket TTS speech synthesis (pockettts.cpp)"  OFF)
if(HEARSAY_HEAR)
    set(HEARSAY_PARAKEET ON)
endif()
if(HEARSAY_SAY)
    set(HEARSAY_POCKETTTS ON)
endif()
if(NOT HEARSAY_PARAKEET AND NOT HEARSAY_POCKETTTS)
    message(FATAL_ERROR "Enable at least one capability or model family: -DHEARSAY_HEAR=ON, -DHEARSAY_SAY=ON, -DHEARSAY_PARAKEET=ON, and/or -DHEARSAY_POCKETTTS=ON")
endif()

# Upstream sources, pinned. Each is fetched as is and changed only by the
# patch files in patches/, which the README lists with their reasons.
include(FetchContent)
set(TRANSCRIBE_COMMIT 90cd63d53b3103d137dcf463ec62338a66f56174)
set(POCKETTTS_COMMIT 6187e69a6b3badbfcc677cb71a846f28ca560f54)

# transcribe.cpp carries ggml at upstream d7cb574 with its two patches
# applied (ggml/UPSTREAM). The three ggml commits pockettts.cpp's fork adds
# on top of the same base are applied to that tree, so both engines share
# one ggml.
FetchContent_Declare(transcribe
    GIT_REPOSITORY https://github.com/handy-computer/transcribe.cpp.git
    GIT_TAG        ${TRANSCRIBE_COMMIT}
    GIT_SHALLOW    FALSE
    SOURCE_SUBDIR  hearsay-no-subproject
    PATCH_COMMAND  git checkout -- .
          COMMAND  git -C ggml checkout -- .
          COMMAND  git apply
                       ${CMAKE_SOURCE_DIR}/patches/transcribe/0001-register-only-built-families.patch
                       ${CMAKE_SOURCE_DIR}/patches/transcribe/0002-stdio-file-io.patch
                       ${CMAKE_SOURCE_DIR}/patches/transcribe/0003-silero-stdio-file-io.patch
          COMMAND  git apply --directory=ggml
                       ${CMAKE_SOURCE_DIR}/patches/ggml/0001-cpu-add-fused-snake-activation-mul-sin-sqr-mul-add.patch
                       ${CMAKE_SOURCE_DIR}/patches/ggml/0002-ggml-fix-conv_1d-result-layout-for-batched-input.patch
                       ${CMAKE_SOURCE_DIR}/patches/ggml/0003-ggml-build-the-same-conv_1d-graph-for-every-batch-si.patch
    UPDATE_DISCONNECTED TRUE)
FetchContent_MakeAvailable(transcribe)
set(TR ${transcribe_SOURCE_DIR})

if(HEARSAY_POCKETTTS)
    FetchContent_Declare(pockettts
        GIT_REPOSITORY https://github.com/ServeurpersoCom/pockettts.cpp.git
        GIT_TAG        ${POCKETTTS_COMMIT}
        GIT_SHALLOW    FALSE
        GIT_SUBMODULES ""
        SOURCE_SUBDIR  hearsay-no-subproject
        PATCH_COMMAND  git checkout -- .
              COMMAND  git apply ${CMAKE_SOURCE_DIR}/patches/pockettts/0001-thread-count-from-POCKET_THREADS.patch
                                 ${CMAKE_SOURCE_DIR}/patches/pockettts/0002-comma-before-every-chunk.patch
        UPDATE_DISCONNECTED TRUE)
    FetchContent_MakeAvailable(pockettts)
    set(PT ${pockettts_SOURCE_DIR})
endif()

if(NOT MSVC)
    add_compile_options(-ffunction-sections -fdata-sections -fno-unwind-tables -fno-asynchronous-unwind-tables)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>)
endif()

# ggml: one build for both engines. The settings are transcribe.cpp's
# (no BLAS, no OpenMP, llamafile kernels); GGML_MAX_NAME is pockettts.cpp's,
# which needs longer tensor names, and it must match across every target
# because it changes the layout of struct ggml_tensor.
add_compile_definitions(GGML_MAX_NAME=128)
set(GGML_BLAS           OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE      ON  CACHE BOOL "" FORCE)
set(GGML_OPENMP         OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${TR}/ggml ggml EXCLUDE_FROM_ALL)

# Benchmark and parity tool: transcribes WAV lists and synthesizes text,
# reporting timings as JSON.
add_executable(hearsay-bench tools/hearsay-bench.c)
target_include_directories(hearsay-bench PRIVATE ${TR}/examples/common)
set_target_properties(hearsay-bench PROPERTIES LINKER_LANGUAGE CXX)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(hearsay-bench PRIVATE -Wl,--gc-sections)
endif()

if(HEARSAY_PARAKEET)
    # Speech recognition: transcribe.cpp's shared code and the Parakeet family.
    add_library(hearsay-hear STATIC
        ${TR}/src/transcribe.cpp
        ${TR}/src/transcribe-asr.cpp
        ${TR}/src/transcribe-diarize.cpp
        ${TR}/src/transcribe-langid.cpp
        ${TR}/src/transcribe-vad.cpp
        ${TR}/src/transcribe-loader.cpp
        ${TR}/src/transcribe-arch.cpp
        ${TR}/src/transcribe-meta.cpp
        ${TR}/src/transcribe-kaldi-fbank.cpp
        ${TR}/src/transcribe-mel.cpp
        ${TR}/src/transcribe-model.cpp
        ${TR}/src/transcribe-tokenizer.cpp
        ${TR}/src/transcribe-prompting.cpp
        ${TR}/src/transcribe-unicode.cpp
        ${TR}/src/transcribe-unicode-data.cpp
        ${TR}/src/transcribe-env.cpp
        ${TR}/src/transcribe-flash-policy.cpp
        ${TR}/src/transcribe-backend.cpp
        ${TR}/src/transcribe-load-common.cpp
        ${TR}/src/transcribe-weights-util.cpp
        ${TR}/src/transcribe-batch-util.cpp
        ${TR}/src/conformer/conformer.cpp
        ${TR}/src/arch/parakeet/model.cpp
        ${TR}/src/arch/parakeet/longform.cpp
        ${TR}/src/arch/parakeet/multitalker.cpp
        ${TR}/src/arch/parakeet/capabilities.cpp
        ${TR}/src/arch/parakeet/weights.cpp
        ${TR}/src/arch/parakeet/encoder.cpp
        ${TR}/src/arch/parakeet/decoder.cpp
        # Parakeet's model code links the Sortformer diarizer for its
        # multi-speaker variant.
        ${TR}/src/arch/sortformer/model.cpp
        ${TR}/src/arch/sortformer/stream.cpp
        ${TR}/src/arch/sortformer/weights.cpp
        # Stands in for the Whisper and Silero .bin loaders the shared code calls.
        src/no-bin-loaders.cpp)
    target_include_directories(hearsay-hear PUBLIC ${TR}/include PRIVATE ${TR}/src)
    target_compile_definitions(hearsay-hear
        PUBLIC  TRANSCRIBE_STATIC
        PRIVATE TRANSCRIBE_BUILD TRANSCRIBE_COMMIT="${TRANSCRIBE_COMMIT}" HEARSAY_PARAKEET)
    if(WIN32)
        target_compile_definitions(hearsay-hear PRIVATE _USE_MATH_DEFINES)
    endif()
    target_link_libraries(hearsay-hear PRIVATE ggml)
    target_compile_definitions(hearsay-bench PRIVATE HEARSAY_PARAKEET=1)
    target_link_libraries(hearsay-bench PRIVATE hearsay-hear)
endif()

if(HEARSAY_POCKETTTS)
    # Speech synthesis: pockettts.cpp's library, without its CLI and HTTP server.
    file(WRITE ${CMAKE_BINARY_DIR}/pocket/version.h
        "#pragma once\n#define POCKET_VERSION \"${POCKETTTS_COMMIT}\"\n")
    add_library(hearsay-yyjson STATIC ${PT}/vendor/yyjson/yyjson.c)
    target_include_directories(hearsay-yyjson PUBLIC ${PT}/vendor/yyjson)
    add_library(hearsay-say STATIC ${PT}/src/pocket.cpp)
    target_include_directories(hearsay-say PUBLIC ${PT}/src PRIVATE ${PT} ${CMAKE_BINARY_DIR}/pocket)
    target_compile_definitions(hearsay-say PUBLIC POCKET_STATIC)
    find_package(Threads REQUIRED)
    target_link_libraries(hearsay-say PUBLIC ggml PRIVATE hearsay-yyjson Threads::Threads)
    foreach(lib ggml-base ggml-cpu)
        if(TARGET ${lib})
            target_link_libraries(hearsay-say PRIVATE ${lib})
        endif()
    endforeach()
    target_compile_definitions(hearsay-bench PRIVATE HEARSAY_POCKETTTS=1)
    target_link_libraries(hearsay-bench PRIVATE hearsay-say)
endif()
