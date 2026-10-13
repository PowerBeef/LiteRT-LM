// swift-tools-version: 5.9
// Copy-fidelity runner for LiteRT-LM #3913, built against this repository's
// Swift package (v0.18.0 on this branch) and its released CLiteRTLM binaries.
import PackageDescription

let package = Package(
  name: "runner",
  platforms: [.macOS(.v14)],
  dependencies: [.package(path: "../../..")],
  targets: [
    .executableTarget(
      name: "runner",
      dependencies: [.product(name: "LiteRTLM", package: "LiteRT-LM")])
  ]
)
