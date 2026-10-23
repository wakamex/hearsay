// Builds hearsay (speech recognition on transcribe.cpp's Parakeet family and
// speech synthesis on pockettts.cpp, sharing one ggml) as a static library,
// and the hearsay-bench tool.
//   zig build -Dhear=true -Dsay=true
//   zig build -Dhear=true -Dsay=true -Dtarget=x86_64-windows-gnu
//   zig build -Dhear=true -Dsay=true -Dtarget=aarch64-macos -Dmacos_sdk=PATH
const std = @import("std");

const transcribe_commit = "90cd63d53b3103d137dcf463ec62338a66f56174";
const pockettts_commit = "6187e69a6b3badbfcc677cb71a846f28ca560f54";

fn useSdk(mod: *std.Build.Module, sdk: ?std.Build.LazyPath) void {
    const dir = sdk orelse return;
    mod.addSystemIncludePath(dir.path(mod.owner, "usr/include"));
    mod.addSystemFrameworkPath(dir.path(mod.owner, "System/Library/Frameworks"));
    mod.addLibraryPath(dir.path(mod.owner, "usr/lib"));
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const macos_sdk = b.option(std.Build.LazyPath, "macos_sdk", "Path of a macOS SDK, for building for macOS on another system");
    const hear = b.option(bool, "hear", "Include default speech recognition (enables -Dparakeet=true)") orelse false;
    const say = b.option(bool, "say", "Include default speech synthesis (enables -Dpockettts=true)") orelse false;
    const parakeet = (b.option(bool, "parakeet", "Include Parakeet speech recognition (transcribe.cpp)") orelse false) or hear;
    const pockettts = (b.option(bool, "pockettts", "Include Pocket TTS speech synthesis (pockettts.cpp)") orelse false) or say;
    const silero = b.option(bool, "silero", "Include Silero voice activity detection (transcribe.cpp)") orelse false;
    const gpu = b.option(Gpu, "gpu", "Run on a GPU where one is usable, else on the CPU: vulkan (needs glslc to build)") orelse .none;
    const glslc = b.option([]const u8, "glslc", "The glslc shader compiler, for -Dgpu=vulkan (default: glslc on the PATH)") orelse "glslc";

    if (!parakeet and !pockettts and !silero) {
        b.getInstallStep().dependOn(&b.addFail("enable at least one capability or model family: -Dhear=true, -Dsay=true, -Dparakeet=true, -Dpockettts=true and/or -Dsilero=true").step);
        return;
    }

    const tr_dep = b.dependency("transcribe", .{});
    const tr = tr_dep.path("");
    const vk_features: []const []const u8 = if (gpu == .vulkan) vulkanFeatures(b, tr_dep, glslc) else &.{};

    // Apply the patch files in patches/ to the upstream sources and write the
    // generated headers into a build-cache directory.
    const wf = b.addWriteFiles();
    _ = wf.add("ggml-version.h", "#pragma once\n#define GGML_VERSION \"0.26.0\"\n#define GGML_COMMIT  \"" ++ transcribe_commit[0..8] ++ "\"\n");

    const patch_ggml_1 = @embedFile("patches/ggml/0001-cpu-add-fused-snake-activation-mul-sin-sqr-mul-add.patch");
    const patch_ggml_2 = @embedFile("patches/ggml/0002-ggml-fix-conv_1d-result-layout-for-batched-input.patch");
    const patch_ggml_3 = @embedFile("patches/ggml/0003-ggml-build-the-same-conv_1d-graph-for-every-batch-si.patch");

    const ggml_c = wf.add("ggml.c", patchFile(b, tr_dep, "ggml", "src/ggml.c", &.{ patch_ggml_2, patch_ggml_3 }));
    const ggml_cpu_c = wf.add("ggml-cpu/ggml-cpu.c", patchFile(b, tr_dep, "ggml", "src/ggml-cpu/ggml-cpu.c", &.{patch_ggml_1}));
    const ggml_ops_cpp = wf.add("ggml-cpu/ops.cpp", patchFile(b, tr_dep, "ggml", "src/ggml-cpu/ops.cpp", &.{patch_ggml_1}));
    _ = wf.add("ggml-cpu/ops.h", patchFile(b, tr_dep, "ggml", "src/ggml-cpu/ops.h", &.{patch_ggml_1}));

    // Cross-builds that do not name a CPU model get AVX2/FMA/F16C on x86-64, matching ggml's default.
    var lib_target = target;
    if (target.result.cpu.arch == .x86_64 and target.query.cpu_model == .determined_by_arch_os) {
        var q = target.query;
        for ([_]std.Target.x86.Feature{ .sse4_2, .avx, .avx2, .bmi2, .fma, .f16c }) |f| {
            q.cpu_features_add.addFeature(@intFromEnum(f));
        }
        lib_target = b.resolveTargetQuery(q);
    }

    const mod = b.createModule(.{
        .target = lib_target,
        .optimize = if (optimize == .debug) .fast else optimize,
        .link_libc = true,
        .link_libcpp = true,
        .sanitize_c = .off,
        .pic = true,
    });
    useSdk(mod, macos_sdk);

    var common_flags: std.ArrayList([]const u8) = .empty;
    common_flags.appendSlice(b.allocator, &.{
        "-O3",
        "-fno-fast-math",
        "-ffunction-sections",
        "-fdata-sections",
        "-fno-unwind-tables",
        "-fno-asynchronous-unwind-tables",
        "-w",
        "-DNDEBUG",
        "-DGGML_MAX_NAME=128",
        "-DGGML_SCHED_MAX_COPIES=4",
        "-DGGML_USE_CPU",
        "-DGGML_USE_CPU_REPACK",
        "-DGGML_USE_LLAMAFILE",
        "-DTRANSCRIBE_STATIC",
        "-DTRANSCRIBE_BUILD",
        "-DTRANSCRIBE_COMMIT=\"" ++ transcribe_commit ++ "\"",
        "-DPOCKET_STATIC",
    }) catch @panic("OOM");
    if (gpu == .vulkan) {
        common_flags.append(b.allocator, "-DGGML_USE_VULKAN") catch @panic("OOM");
        common_flags.appendSlice(b.allocator, vk_features) catch @panic("OOM");
    }
    switch (target.result.os.tag) {
        .linux => common_flags.appendSlice(b.allocator, &.{ "-D_GNU_SOURCE", "-D_XOPEN_SOURCE=600" }) catch @panic("OOM"),
        .macos => {
            common_flags.appendSlice(b.allocator, &.{ "-D_DARWIN_C_SOURCE", "-D_XOPEN_SOURCE=600" }) catch @panic("OOM");
            mod.linkFramework("Accelerate", .{});
        },
        .windows => common_flags.appendSlice(b.allocator, &.{ "-D_CRT_SECURE_NO_WARNINGS", "-D_USE_MATH_DEFINES" }) catch @panic("OOM"),
        else => {},
    }

    const c_flags = b.allocator.dupe([]const u8, common_flags.items) catch @panic("OOM");
    var c_flag_list: std.ArrayList([]const u8) = .empty;
    c_flag_list.append(b.allocator, "-std=gnu11") catch @panic("OOM");
    c_flag_list.appendSlice(b.allocator, c_flags) catch @panic("OOM");

    var cxx_flag_list: std.ArrayList([]const u8) = .empty;
    cxx_flag_list.appendSlice(b.allocator, &.{ "-std=gnu++17", "-fno-rtti" }) catch @panic("OOM");
    cxx_flag_list.appendSlice(b.allocator, c_flags) catch @panic("OOM");

    var cxx_noexcept_list: std.ArrayList([]const u8) = .empty;
    cxx_noexcept_list.appendSlice(b.allocator, cxx_flag_list.items) catch @panic("OOM");
    cxx_noexcept_list.append(b.allocator, "-fno-exceptions") catch @panic("OOM");

    mod.addIncludePath(wf.getDirectory());
    mod.addIncludePath(tr.path(b, "ggml/include"));
    mod.addIncludePath(tr.path(b, "ggml/src"));
    mod.addIncludePath(tr.path(b, "ggml/src/ggml-cpu"));

    // ggml C and C++ sources.
    mod.addCSourceFile(.{ .file = ggml_c, .flags = c_flag_list.items });
    mod.addCSourceFile(.{ .file = ggml_cpu_c, .flags = c_flag_list.items });
    mod.addCSourceFile(.{ .file = ggml_ops_cpp, .flags = cxx_noexcept_list.items });
    mod.addCSourceFiles(.{
        .root = tr.path(b, "ggml/src"),
        .files = &.{
            "ggml-alloc.c",
            "ggml-quants.c",
            "ggml-cpu/quants.c",
        },
        .flags = c_flag_list.items,
    });
    mod.addCSourceFiles(.{
        .root = tr.path(b, "ggml/src"),
        .files = &.{
            "ggml.cpp",
            "ggml-backend.cpp",
            "ggml-backend-dl.cpp",
            "ggml-backend-meta.cpp",
            "ggml-opt.cpp",
            "ggml-threading.cpp",
            "ggml-cpu/ggml-cpu.cpp",
            "ggml-cpu/repack.cpp",
            "ggml-cpu/hbm.cpp",
            "ggml-cpu/traits.cpp",
            "ggml-cpu/amx/amx.cpp",
            "ggml-cpu/amx/mmq.cpp",
            "ggml-cpu/tiled/tiled.cpp",
            "ggml-cpu/tiled/tiled-kernel.cpp",
            "ggml-cpu/binary-ops.cpp",
            "ggml-cpu/unary-ops.cpp",
            "ggml-cpu/vec.cpp",
            "ggml-cpu/llamafile/sgemm.cpp",
        },
        .flags = cxx_noexcept_list.items,
    });
    mod.addCSourceFiles(.{
        .root = tr.path(b, "ggml/src"),
        .files = &.{
            "ggml-backend-reg.cpp",
            "gguf.cpp",
        },
        .flags = cxx_flag_list.items,
    });

    switch (target.result.cpu.arch) {
        .x86_64 => {
            mod.addCSourceFile(.{ .file = tr.path(b, "ggml/src/ggml-cpu/arch/x86/quants.c"), .flags = c_flag_list.items });
            mod.addCSourceFile(.{ .file = tr.path(b, "ggml/src/ggml-cpu/arch/x86/repack.cpp"), .flags = cxx_noexcept_list.items });
        },
        .aarch64 => {
            mod.addCSourceFile(.{ .file = tr.path(b, "ggml/src/ggml-cpu/arch/arm/quants.c"), .flags = c_flag_list.items });
            mod.addCSourceFile(.{ .file = tr.path(b, "ggml/src/ggml-cpu/arch/arm/repack.cpp"), .flags = cxx_noexcept_list.items });
        },
        else => {},
    }

    if (gpu == .vulkan) addVulkan(b, mod, tr_dep, glslc, vk_features, cxx_flag_list.items);

    const lib = b.addLibrary(.{ .name = "hearsay", .linkage = .static, .root_module = mod });

    if (parakeet or silero) {
        const patch_tr_arch = @embedFile("patches/transcribe/0001-register-only-built-families.patch");
        const patch_tr_io = @embedFile("patches/transcribe/0002-stdio-file-io.patch");
        const tr_arch_cpp = wf.add("transcribe-arch.cpp", patchFile(b, tr_dep, "", "src/transcribe-arch.cpp", &.{patch_tr_arch}));
        _ = wf.add("transcribe-path.h", patchFile(b, tr_dep, "", "src/transcribe-path.h", &.{patch_tr_io}));
        const tr_cpp = wf.add("transcribe.cpp", patchFile(b, tr_dep, "", "src/transcribe.cpp", &.{patch_tr_io}));
        const tr_load_common_cpp = wf.add("transcribe-load-common.cpp", patchFile(b, tr_dep, "", "src/transcribe-load-common.cpp", &.{patch_tr_io}));
        const tr_loader_cpp = wf.addCopyFile(tr.path(b, "src/transcribe-loader.cpp"), "transcribe-loader.cpp");

        mod.addIncludePath(tr.path(b, "include"));
        mod.addIncludePath(tr.path(b, "src"));
        var arch_flags: std.ArrayList([]const u8) = .empty;
        arch_flags.appendSlice(b.allocator, cxx_noexcept_list.items) catch @panic("OOM");
        if (parakeet) arch_flags.append(b.allocator, "-DHEARSAY_PARAKEET") catch @panic("OOM");
        if (silero) arch_flags.append(b.allocator, "-DHEARSAY_SILERO") catch @panic("OOM");
        mod.addCSourceFile(.{ .file = tr_arch_cpp, .flags = arch_flags.items });
        mod.addCSourceFile(.{ .file = tr_loader_cpp, .flags = cxx_noexcept_list.items });
        mod.addCSourceFile(.{ .file = tr_cpp, .flags = cxx_flag_list.items });
        mod.addCSourceFile(.{ .file = tr_load_common_cpp, .flags = cxx_flag_list.items });
        mod.addCSourceFile(.{ .file = b.path("src/no-bin-loaders.cpp"), .flags = arch_flags.items });
        mod.addCSourceFiles(.{
            .root = tr.path(b, "src"),
            .files = &.{
                "transcribe-meta.cpp",
                "transcribe-kaldi-fbank.cpp",
                "transcribe-model.cpp",
                "transcribe-tokenizer.cpp",
                "transcribe-prompting.cpp",
                "transcribe-unicode-data.cpp",
                "transcribe-env.cpp",
                "transcribe-flash-policy.cpp",
                "transcribe-weights-util.cpp",
            },
            .flags = cxx_noexcept_list.items,
        });
        if (parakeet) mod.addCSourceFiles(.{
            .root = tr.path(b, "src"),
            .files = &.{
                "conformer/conformer.cpp",
                "arch/parakeet/model.cpp",
                "arch/parakeet/longform.cpp",
                "arch/parakeet/multitalker.cpp",
                "arch/parakeet/capabilities.cpp",
                "arch/parakeet/weights.cpp",
                "arch/parakeet/encoder.cpp",
                "arch/parakeet/decoder.cpp",
                "arch/sortformer/model.cpp",
                "arch/sortformer/stream.cpp",
                "arch/sortformer/weights.cpp",
            },
            .flags = cxx_noexcept_list.items,
        });
        mod.addCSourceFiles(.{
            .root = tr.path(b, "src"),
            .files = &.{
                "transcribe-asr.cpp",
                "transcribe-diarize.cpp",
                "transcribe-langid.cpp",
                "transcribe-vad.cpp",
                "transcribe-mel.cpp",
                "transcribe-backend.cpp",
                "transcribe-unicode.cpp",
                "transcribe-batch-util.cpp",
            },
            .flags = cxx_flag_list.items,
        });
        if (silero) {
            const patch_silero_io = @embedFile("patches/transcribe/0003-silero-stdio-file-io.patch");
            const silero_model_cpp = wf.add("silero-vad-model.cpp", patchFile(b, tr_dep, "", "src/arch/silero_vad/model.cpp", &.{patch_silero_io}));
            mod.addIncludePath(tr.path(b, "src/arch/silero_vad"));
            mod.addCSourceFile(.{ .file = silero_model_cpp, .flags = cxx_noexcept_list.items });
            mod.addCSourceFiles(.{
                .root = tr.path(b, "src/arch/silero_vad"),
                .files = &.{ "weights.cpp", "encoder.cpp", "decoder.cpp" },
                .flags = cxx_noexcept_list.items,
            });
        }
        lib.installHeader(wf.add("include/transcribe.h", staticHeader(b, tr_dep, "include/transcribe.h", "TRANSCRIBE_STATIC")), "transcribe.h");
        if (silero) lib.installHeader(tr.path(b, "include/transcribe/vad.h"), "transcribe/vad.h");
    }

    if (pockettts) {
        const pt_dep = b.dependency("pockettts", .{});
        const pt = pt_dep.path("");
        const patch_pt_threads = @embedFile("patches/pockettts/0001-thread-count-from-POCKET_THREADS.patch");
        const patch_pt_comma = @embedFile("patches/pockettts/0002-comma-before-every-chunk.patch");

        _ = wf.add("pocket/version.h", "#pragma once\n#define POCKET_VERSION \"" ++ pockettts_commit ++ "\"\n");
        _ = wf.add("pocket/backend.h", patchFile(b, pt_dep, "", "src/backend.h", &.{patch_pt_threads}));
        const pocket_cpp = wf.add("pocket/pocket.cpp", patchFile(b, pt_dep, "", "src/pocket.cpp", &.{patch_pt_comma}));

        mod.addIncludePath(wf.getDirectory().path(b, "pocket"));
        mod.addIncludePath(pt.path(b, "src"));
        mod.addIncludePath(pt.path(b, "vendor/yyjson"));
        mod.addCSourceFile(.{ .file = pocket_cpp, .flags = cxx_flag_list.items });
        mod.addCSourceFile(.{ .file = pt.path(b, "vendor/yyjson/yyjson.c"), .flags = c_flag_list.items });
        lib.installHeader(wf.add("include/pocket.h", staticHeader(b, pt_dep, "src/pocket.h", "POCKET_STATIC")), "pocket.h");
    }

    b.installArtifact(lib);

    // Benchmark and parity tool: zig-out/bin/hearsay-bench, for the recognition and synthesis families.
    if (!parakeet and !pockettts) return;
    const bench_mod = b.createModule(.{
        .target = target,
        .optimize = if (optimize == .debug) .fast else optimize,
        .link_libc = true,
        .link_libcpp = true,
        .strip = optimize != .debug,
    });
    useSdk(bench_mod, macos_sdk);
    var bench_flags: std.ArrayList([]const u8) = .empty;
    bench_flags.appendSlice(b.allocator, &.{
        "-std=gnu11",
        "-Wall",
        "-Wextra",
        "-ffunction-sections",
        "-fdata-sections",
        "-fno-unwind-tables",
        "-fno-asynchronous-unwind-tables",
    }) catch @panic("OOM");
    if (parakeet) bench_flags.append(b.allocator, "-DHEARSAY_PARAKEET") catch @panic("OOM");
    if (pockettts) bench_flags.append(b.allocator, "-DHEARSAY_POCKETTTS") catch @panic("OOM");
    bench_mod.addCSourceFile(.{
        .file = b.path("tools/hearsay-bench.c"),
        .flags = bench_flags.items,
    });
    bench_mod.addIncludePath(tr.path(b, "examples/common"));
    bench_mod.linkLibrary(lib);
    const bench = b.addExecutable(.{ .name = "hearsay-bench", .root_module = bench_mod });
    bench.link_gc_sections = true;
    b.installArtifact(bench);

    const run = b.addRunArtifact(bench);
    run.step.dependOn(b.getInstallStep());
    run.addPassthruArgs();
    b.step("bench", "Run hearsay-bench").dependOn(&run.step);
}

const Gpu = enum { none, vulkan };

// The optional shader features the glslc in use supports, as ggml's own build finds them: each is a
// -D define for the shader generator and for ggml-vulkan.cpp. Fails the build when glslc cannot run.
fn vulkanFeatures(b: *std.Build, tr_dep: *std.Build.Dependency, glslc: []const u8) []const []const u8 {
    var code: u8 = 0;
    if (b.runAllowFail(&.{ glslc, "--version" }, &code, .ignore)) |out| b.allocator.free(out) else |_| {
        std.debug.panic("-Dgpu=vulkan needs the glslc shader compiler: none runs as {s} (name it with -Dglslc=PATH, or build for the CPU only with -Dgpu=none)", .{glslc});
    }
    const tests = [_][2][]const u8{
        .{ "coopmat", "GGML_VULKAN_COOPMAT_GLSLC_SUPPORT" },
        .{ "coopmat2", "GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT" },
        .{ "coopmat2_decode_vector", "GGML_VULKAN_COOPMAT2_DECODE_VECTOR_GLSLC_SUPPORT" },
        .{ "integer_dot", "GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT" },
        .{ "bfloat16", "GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT" },
        .{ "float_e2m1", "GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT" },
        .{ "float_e4m3", "GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT" },
    };
    var list: std.ArrayList([]const u8) = .empty;
    for (tests) |t| {
        const path = tr_dep.builder.root.joinString(b.allocator, b.fmt("ggml/src/ggml-vulkan/vulkan-shaders/feature-tests/{s}.comp", .{t[0]})) catch @panic("OOM");
        if (b.runAllowFail(&.{ glslc, "-o", "-", "-fshader-stage=compute", "--target-env=vulkan1.3", path }, &code, .ignore)) |out| {
            b.allocator.free(out);
            list.append(b.allocator, b.fmt("-D{s}", .{t[1]})) catch @panic("OOM");
        } else |_| {}
    }
    return list.items;
}

// ggml's Vulkan backend in the library: the shader generator built for this machine, each shader
// compiled to SPIR-V by glslc and embedded as a C++ source, and the Vulkan loader opened at run time
// (src/vulkan-loader.cpp) so the program runs on the CPU where there is none.
fn addVulkan(b: *std.Build, mod: *std.Build.Module, tr_dep: *std.Build.Dependency, glslc: []const u8, features: []const []const u8, cxx_flags: []const []const u8) void {
    const vk = b.lazyDependency("vulkan_headers", .{}) orelse return;
    const spirv = b.lazyDependency("spirv_headers", .{}) orelse return;
    const shaders = "ggml/src/ggml-vulkan/vulkan-shaders";
    var gen_flags: std.ArrayList([]const u8) = .empty;
    gen_flags.appendSlice(b.allocator, &.{ "-std=c++17", "-O2", "-w" }) catch @panic("OOM");
    gen_flags.appendSlice(b.allocator, features) catch @panic("OOM");
    const gen_mod = b.createModule(.{ .target = b.graph.host, .optimize = .ReleaseFast, .link_libc = true, .link_libcpp = true });
    gen_mod.addCSourceFile(.{ .file = tr_dep.path(shaders ++ "/vulkan-shaders-gen.cpp"), .flags = gen_flags.items });
    const gen = b.addExecutable(.{ .name = "vulkan-shaders-gen", .root_module = gen_mod });

    const header = b.addRunArtifact(gen);
    header.addArg("--output-dir");
    _ = header.addOutputDirectoryArg("spv");
    header.addArg("--target-hpp");
    const hpp = header.addOutputFileArg("ggml-vulkan-shaders.hpp");
    mod.addIncludePath(hpp.dirname());

    const shaders_dir = tr_dep.builder.root.joinString(b.allocator, shaders) catch @panic("OOM");
    var dir = std.Io.Dir.cwd().openDir(b.graph.io, shaders_dir, .{ .iterate = true }) catch |err|
        std.debug.panic("cannot open {s}: {t}", .{ shaders, err });
    defer dir.close(b.graph.io);
    var it = dir.iterate();
    while (it.next(b.graph.io) catch |err| std.debug.panic("cannot list {s}: {t}", .{ shaders, err })) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".comp")) continue;
        const name = b.dupe(entry.name);
        const run = b.addRunArtifact(gen);
        run.addArgs(&.{ "--glslc", glslc, "--source" });
        run.addFileArg(tr_dep.path(b.fmt("{s}/{s}", .{ shaders, name })));
        run.addArg("--output-dir");
        _ = run.addOutputDirectoryArg("spv");
        run.addArgs(&.{ "--target-hpp", "ggml-vulkan-shaders.hpp", "--target-cpp" });
        // Arrays of SPIR-V bytes only: nothing to optimise.
        mod.addCSourceFile(.{ .file = run.addOutputFileArg(b.fmt("{s}.cpp", .{name})), .flags = &.{ "-std=gnu++17", "-O0", "-w" } });
    }

    mod.addIncludePath(vk.path("include"));
    mod.addIncludePath(spirv.path("include"));
    mod.addCSourceFiles(.{
        .root = tr_dep.path("ggml/src/ggml-vulkan"),
        .files = &.{ "ggml-vulkan.cpp", "ggml-vulkan-buffers.cpp", "ggml-vulkan-debug.cpp" },
        .flags = cxx_flags,
    });
    mod.addCSourceFile(.{ .file = b.path("src/vulkan-loader.cpp"), .flags = cxx_flags });
}

fn staticHeader(b: *std.Build, dep: *std.Build.Dependency, rel_path: []const u8, macro: []const u8) []const u8 {
    const raw: []const u8 = dep.builder.root.root_dir.handle.readFileAlloc(b.graph.io, rel_path, b.allocator, .unlimited) catch |err|
        std.debug.panic("cannot read {s}: {t}", .{ rel_path, err });
    return b.fmt("#ifndef {s}\n#define {s} 1\n#endif\n{s}", .{ macro, macro, raw });
}

// Applies unified diff hunks for `rel_path` from each patch in `patches` to the
// file at `dir_prefix/rel_path` in `dep`. Panics if any context or removed line
// does not match the upstream source.
fn patchFile(
    b: *std.Build,
    dep: *std.Build.Dependency,
    dir_prefix: []const u8,
    rel_path: []const u8,
    patches: []const []const u8,
) []const u8 {
    const full = if (dir_prefix.len == 0) rel_path else b.fmt("{s}/{s}", .{ dir_prefix, rel_path });
    var current: []const u8 = dep.builder.root.root_dir.handle.readFileAlloc(b.graph.io, full, b.allocator, .unlimited) catch |err|
        std.debug.panic("cannot read {s}: {t}", .{ full, err });
    for (patches) |patch| {
        current = applyUnifiedDiff(b.allocator, current, patch, rel_path);
    }
    return current;
}

fn applyUnifiedDiff(allocator: std.mem.Allocator, src: []const u8, patch: []const u8, target_path: []const u8) []const u8 {
    var src_lines: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, src, '\n');
    while (it.next()) |line| {
        src_lines.append(allocator, line) catch @panic("OOM");
    }
    if (src.len > 0 and src[src.len - 1] == '\n') {
        _ = src_lines.pop();
    }

    var out: std.ArrayList(u8) = .empty;
    var cur: usize = 0;
    var in_target = false;
    var matched_file = false;
    const header_prefix = std.fmt.allocPrint(allocator, "--- a/{s}", .{target_path}) catch @panic("OOM");

    var lines = std.mem.splitScalar(u8, patch, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trimEnd(u8, raw_line, "\r");
        if (std.mem.startsWith(u8, line, "--- a/")) {
            in_target = std.mem.startsWith(u8, line, header_prefix) and
                (line.len == header_prefix.len or line[header_prefix.len] == '\t' or line[header_prefix.len] == ' ');
            if (in_target) matched_file = true;
            continue;
        }
        if (!in_target or !std.mem.startsWith(u8, line, "@@ -")) continue;

        // Parse "@@ -old_start,old_count +new_start,new_count @@"
        const after_minus = line["@@ -".len..];
        const comma1 = std.mem.indexOfScalar(u8, after_minus, ',') orelse @panic("invalid hunk header");
        const space1 = std.mem.indexOfScalar(u8, after_minus, ' ') orelse @panic("invalid hunk header");
        const old_start = std.fmt.parseInt(usize, after_minus[0..comma1], 10) catch @panic("invalid old_start");
        const old_rem = std.fmt.parseInt(usize, after_minus[comma1 + 1 .. space1], 10) catch @panic("invalid old_count");

        const plus_pos = std.mem.indexOf(u8, after_minus, " +") orelse @panic("invalid hunk header");
        const after_plus = after_minus[plus_pos + 2 ..];
        const comma2 = std.mem.indexOfScalar(u8, after_plus, ',') orelse @panic("invalid hunk header");
        const space2 = std.mem.indexOfScalar(u8, after_plus, ' ') orelse @panic("invalid hunk header");
        const new_rem = std.fmt.parseInt(usize, after_plus[comma2 + 1 .. space2], 10) catch @panic("invalid new_count");

        var pre: std.ArrayList([]const u8) = .empty;
        var post: std.ArrayList([]const u8) = .empty;
        while (pre.items.len < old_rem or post.items.len < new_rem) {
            const hraw = lines.next() orelse @panic("unexpected EOF in patch hunk");
            const hline = std.mem.trimEnd(u8, hraw, "\r");
            if (hline.len == 0) @panic("empty hunk line");
            switch (hline[0]) {
                ' ' => {
                    pre.append(allocator, hline[1..]) catch @panic("OOM");
                    post.append(allocator, hline[1..]) catch @panic("OOM");
                },
                '-' => pre.append(allocator, hline[1..]) catch @panic("OOM"),
                '+' => post.append(allocator, hline[1..]) catch @panic("OOM"),
                else => @panic("unexpected hunk prefix"),
            }
        }

        const hint = old_start - 1;
        var match_idx: ?usize = null;
        var i: usize = cur;
        while (i + pre.items.len <= src_lines.items.len) : (i += 1) {
            var ok = true;
            for (pre.items, 0..) |pline, j| {
                if (!std.mem.eql(u8, src_lines.items[i + j], pline)) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                if (match_idx) |prev| {
                    const prev_dist = if (prev > hint) prev - hint else hint - prev;
                    const new_dist = if (i > hint) i - hint else hint - i;
                    if (new_dist < prev_dist) match_idx = i;
                } else {
                    match_idx = i;
                }
            }
        }
        const idx = match_idx orelse std.debug.panic("patch hunk did not match {s} near line {d}", .{ target_path, old_start });
        while (cur < idx) : (cur += 1) {
            out.appendSlice(allocator, src_lines.items[cur]) catch @panic("OOM");
            out.append(allocator, '\n') catch @panic("OOM");
        }
        for (post.items) |pline| {
            out.appendSlice(allocator, pline) catch @panic("OOM");
            out.append(allocator, '\n') catch @panic("OOM");
        }
        cur = idx + pre.items.len;
    }

    if (!matched_file) std.debug.panic("target {s} not found in patch", .{target_path});
    while (cur < src_lines.items.len) : (cur += 1) {
        out.appendSlice(allocator, src_lines.items[cur]) catch @panic("OOM");
        out.append(allocator, '\n') catch @panic("OOM");
    }
    return out.items;
}
