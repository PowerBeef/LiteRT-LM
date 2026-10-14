// swift-tools-version: 6.0
// agello-bench: runs the same measured cases against a LiteRT-LM build (an
// "arm") on Apple silicon. See README.md.
import PackageDescription

// The LiteRT-LM package to measure: this fork by default, or another checkout
// given by LITERTLM_PACKAGE. SwiftPM names a path package after its folder,
// so that checkout's folder must be named LiteRT-LM too.
let litertlm = Context.environment["LITERTLM_PACKAGE"] ?? "../.."

let package = Package(
  name: "agello-bench",
  platforms: [.macOS("27.0"), .iOS("27.0")],
  products: [
    .library(name: "BenchKit", targets: ["BenchKit"]),
    .executable(name: "agello-bench", targets: ["agello-bench"]),
  ],
  dependencies: [.package(path: litertlm)],
  targets: [
    .target(
      name: "BenchKit",
      dependencies: [.product(name: "LiteRTLM", package: "LiteRT-LM")]),
    .executableTarget(name: "agello-bench", dependencies: ["BenchKit"]),
    .testTarget(name: "BenchKitTests", dependencies: ["BenchKit"]),
  ]
)
