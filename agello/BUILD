# Agello's builds of LiteRT-LM (PowerBeef/agello-litert-lm).
#
#   bazel build --config=agello //agello:CLiteRTLM
#
# One XCFramework with iPhone and Mac slices, built from source with LiteRT's Metal accelerator
# linked statically (//runtime/executor:default_static_gpu_accelerator), for the Swift package's
# single CLiteRTLM binary target. Google's release binaries are built internally instead.

load("@build_bazel_rules_apple//apple:apple.bzl", "apple_xcframework")

# litert_lm_get_build_info's JSON, from the stamp agello/workspace_status.sh prints.
genrule(
    name = "build_info_stamp",
    outs = ["build_info_stamp.cc"],
    cmd = "$(location make_build_info.sh) bazel-out/stable-status.txt > $@",
    stamp = 1,
    tools = ["make_build_info.sh"],
)

cc_library(
    name = "build_info",
    srcs = [":build_info_stamp"],
    visibility = ["//visibility:public"],
    alwayslink = True,
)

apple_xcframework(
    name = "CLiteRTLM",
    bundle_id = "com.patricedery.agello.CLiteRTLM",
    features = ["exported_symbols"],
    infoplists = ["Info.plist"],
    ios = {
        "device": ["arm64"],
    },
    macos = ["arm64"],
    minimum_os_versions = {
        "ios": "27.0",
        "macos": "27.0",
    },
    public_hdrs = [
        "//c:api_export.h",
        "//c:conversation.h",
        "//c:embedding_engine.h",
        "//c:engine.h",
        "//c:error_reporter.h",
        "//c:experimental.h",
        "//c:model_info.h",
        "//c:telemetry.h",
        "//c:build_info.h",
    ],
    deps = ["//swift:CLiteRTLM_objc"],
)
