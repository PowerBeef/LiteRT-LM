// agello-bench: measure one LiteRT-LM build (an arm) with a fixed set of cases.
import BenchKit
import Foundation

let usage = """
  usage: agello-bench run --model <file.litertlm> (--suite <cases.json> | --case <json>)... [options]
         agello-bench env

  run options:
    --backend gpu|cpu   backend (gpu)
    --fp32              FP32 activations (the GPU default is FP16)
    --kv N              KV cache capacity, maxNumTokens (8192)
    --mtp               speculative decoding with the model's MTP drafter
    --cache DIR         serialization cache folder (default: no cache); builds that report a build
                        id (ours) use DIR/<build id>, and other builds' folders there are deleted
    --cold              empty the cache folder before loading
    --repeat N          passes over the cases (1)
    --warmup N          warm-up runs of the first case, recorded with "warmup": true (0)
    --label NAME        the arm's name in the records (arm)
    --out FILE          append JSONL records to FILE (default: stdout)
    --hash-model        record the model file's SHA-256 (reads the whole file)
    --no-benchmark-mode run without LiteRT-LM's benchmark mode (no busy-waiting on Metal); speeds
                        come from the engine's telemetry (our builds)
    --telemetry N       engine telemetry level (our builds): 0 off, 1 counters (default),
                        2 spans and GPU time, 3 plus a trace, 4 diagnostic modes
    --trace FILE        write the span trace (Chrome/Perfetto JSON) at the end; implies level 3

  env prints the machine record (hardware, OS, GPU, runtime binary and its SHA-256).
  """

func fail(_ message: String) -> Never {
  FileHandle.standardError.write(Data("agello-bench: \(message)\n".utf8))
  exit(2)
}

func note(_ message: String) {
  FileHandle.standardError.write(Data("\(message)\n".utf8))
}

var args = Array(CommandLine.arguments.dropFirst())
guard let command = args.first else { fail(usage) }
args.removeFirst()

switch command {
case "env":
  let encoder = JSONEncoder()
  encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
  print(String(decoding: try encoder.encode(Machine.capture()), as: UTF8.self))
  exit(0)
case "run":
  break
default:
  fail(usage)
}

var settings = EngineSettings(model: URL(fileURLWithPath: "/"))
var model: String?
var cases: [BenchCase] = []
var repeats = 1
var warmups = 0
var label = "arm"
var out: String?
var cold = false
var hashModel = false
var telemetryLevel: Int32?
var tracePath: String?

@MainActor func value(_ flag: String) -> String {
  guard !args.isEmpty else { fail("\(flag) needs a value") }
  return args.removeFirst()
}

@MainActor func number(_ flag: String) -> Int {
  guard let n = Int(value(flag)), n >= 0 else { fail("\(flag) needs a non-negative number") }
  return n
}

while !args.isEmpty {
  let flag = args.removeFirst()
  switch flag {
  case "--model": model = value(flag)
  case "--backend":
    settings.backend = value(flag)
    guard ["gpu", "cpu"].contains(settings.backend) else { fail("--backend is gpu or cpu") }
  case "--fp32": settings.fp32 = true
  case "--kv": settings.maxNumTokens = number(flag)
  case "--mtp": settings.mtp = true
  case "--cache":
    // LiteRT-LM's cache keys ignore the runtime build: builds that report an id get a subfolder.
    // Each holds a 2.2 GB weight copy for E4B, so other builds' subfolders are deleted.
    var dir = URL(fileURLWithPath: value(flag), isDirectory: true)
    if let build = Machine.runtimeBuildInfo()?["build_id"] {
      let root = dir
      dir.appendPathComponent(build, isDirectory: true)
      let others = (try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []
      for name in others where name != build && name.count == 12
        && name.allSatisfy({ $0.isHexDigit && !$0.isUppercase })
      {
        try? FileManager.default.removeItem(at: root.appendingPathComponent(name))
        note("cache: removed another build's folder \(name)")
      }
    }
    settings.cacheDir = dir
  case "--cold": cold = true
  case "--repeat": repeats = max(1, number(flag))
  case "--warmup": warmups = number(flag)
  case "--label": label = value(flag)
  case "--out": out = value(flag)
  case "--hash-model": hashModel = true
  case "--no-benchmark-mode": settings.benchmarkMode = false
  case "--telemetry": telemetryLevel = Int32(number(flag))
  case "--trace": tracePath = value(flag)
  case "--suite":
    let path = value(flag)
    do {
      cases += try BenchCase.load(suite: URL(fileURLWithPath: path))
    } catch {
      fail("can't read suite \(path): \(error)")
    }
  case "--case":
    let json = value(flag)
    do {
      cases.append(try JSONDecoder().decode(BenchCase.self, from: Data(json.utf8)))
    } catch {
      fail("can't read case \(json): \(error)")
    }
  default: fail("unknown option \(flag)\n\(usage)")
  }
}
guard let model else { fail("--model is required\n\(usage)") }
guard !cases.isEmpty else { fail("add cases with --suite or --case") }
settings.model = URL(fileURLWithPath: model)
guard FileManager.default.fileExists(atPath: settings.model.path) else { fail("no model at \(model)") }
if cold, let dir = settings.cacheDir {
  try? FileManager.default.removeItem(at: dir)
}

let output: FileHandle
if let out {
  if !FileManager.default.fileExists(atPath: out) {
    FileManager.default.createFile(atPath: out, contents: nil)
  }
  guard let handle = FileHandle(forWritingAtPath: out) else { fail("can't write \(out)") }
  handle.seekToEndOfFile()
  output = handle
} else {
  output = FileHandle.standardOutput
}

if tracePath != nil { telemetryLevel = max(telemetryLevel ?? 3, 3) }
if let telemetryLevel { EngineTelemetry.level = telemetryLevel }

// Engine benchmarks make their own engine: skip the chat engine when they're all there is.
let needsEngine = cases.contains { $0.kind != .engineBenchmark }
let bench = try await Bench.load(settings, label: label, hashModel: hashModel, loadEngine: needsEngine)
note(
  "\(label): loaded \(settings.model.lastPathComponent) in \(String(format: "%.1f", bench.loadSeconds)) s"
    + " (cache \(bench.cache), \(settings.backend), kv \(settings.maxNumTokens), mtp \(settings.mtp))"
    + " on \(bench.machine.hardware), \(bench.machine.gpu)")

@MainActor func record(_ item: BenchCase, rep: Int, warmup: Bool) async throws {
  let result = try await bench.run(item, rep: rep, warmup: warmup)
  output.write(Data((try result.jsonLine() + "\n").utf8))
  let verdict = result.exact.map { $0 ? "exact" : "wrong at \(result.firstMismatch ?? -1)" } ?? "-"
  note(
    String(
      format: "%@%@ rep %d: %@ · prefill %d @ %.0f t/s · decode %d @ %.1f t/s · %.2f s",
      warmup ? "(warm-up) " : "", item.id, rep, verdict, result.prefill, result.prefillTPS,
      result.decode, result.decodeTPS, result.seconds))
}

for _ in 0..<warmups {
  try await record(cases[0], rep: -1, warmup: true)
}
for rep in 0..<repeats {
  for item in cases {
    try await record(item, rep: rep, warmup: false)
  }
}
if let out, let load = bench.loadTelemetry, let data = try? JSONEncoder().encode(load) {
  try? data.write(to: URL(fileURLWithPath: out + ".load-telemetry.json"))
}
if let tracePath {
  let ok = EngineTelemetry.writeTrace(to: URL(fileURLWithPath: tracePath))
  note(ok ? "trace: \(tracePath)" : "trace: not written (this runtime has no telemetry)")
}
// AGELLO_BENCH_HOLD=<seconds>: stay alive after the last case, for footprint/vmmap/malloc_history.
if let hold = ProcessInfo.processInfo.environment["AGELLO_BENCH_HOLD"], let seconds = Double(hold) {
  note("holding \(seconds) s (pid \(ProcessInfo.processInfo.processIdentifier))")
  try await Task.sleep(for: .seconds(seconds))
}
