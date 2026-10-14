# Agello telemetry: low-overhead probes for LiteRT-LM, LiteRT and ML Drift. See agello_telemetry.h.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

cc_library(
    name = "telemetry",
    srcs = ["agello/agello_telemetry.cc"],
    hdrs = ["agello/agello_telemetry.h"],
)

# Metal command-buffer timing (Apple only).
objc_library(
    name = "metal",
    srcs = ["agello/agello_telemetry_metal.mm"],
    hdrs = ["agello/agello_telemetry_metal.h"],
    sdk_frameworks = ["Metal"],
    deps = [":telemetry"],
)

cc_test(
    name = "telemetry_test",
    srcs = ["agello/agello_telemetry_test.cc"],
    deps = [
        ":telemetry",
        "@com_google_googletest//:gtest_main",
    ],
)
