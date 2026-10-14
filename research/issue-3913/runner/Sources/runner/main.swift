// LiteRT-LM #3913 copy-fidelity runner (same prompts as Agello's probe 10).
// Usage: runner <model.litertlm> <gpu|cpu> <default|fp32> <cases.json>
//   cases.json: [{"id": "...", "sentences": 120, "late": false, "seed": 1}, ...]
// Prints one JSON object per case.
import Foundation
import LiteRTLM

struct Case: Decodable {
  let id: String
  let sentences: Int
  let late: Bool
  let seed: Int
}

func referenceDigits(seed: Int) -> String {
  var state = UInt64(seed) &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
  return String((0..<48).map { _ -> Character in
    state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
    return Character(String((state >> 33) % 10))
  })
}

func filler(sentences: Int) -> String {
  let subjects = [
    "The harbour master", "A visiting engineer", "The night ferry", "Every lighthouse keeper",
    "The council archive", "A quiet fishing village", "The coastal railway", "Old stone breakwaters",
    "The weather station", "A crew of volunteers", "The market square", "Seasonal tourists",
  ]
  let actions = [
    "kept careful notes about the tides and the changing colour of the sea",
    "described how fog rolled in from the bay long before sunrise",
    "noticed that the gulls gathered near the pier whenever storms approached",
    "repaired the wooden walkway after a winter of strong easterly winds",
    "spoke about the slow work of rebuilding the eastern quay",
    "watched cargo ships wait outside the channel for the evening tide",
    "planted hardy grasses along the dunes to hold the sand in place",
    "painted the signal tower white again before the summer festival",
    "explained why the old canal locks were opened only at high water",
  ]
  return (0..<sentences).map { i in
    "\(subjects[i % subjects.count]) \(actions[(i / subjects.count + i) % actions.count])."
  }.joined(separator: " ")
}

let args = CommandLine.arguments
guard args.count == 5 || args.count == 6 else {
  FileHandle.standardError.write("usage: runner <model> <gpu|cpu> <fp16|fp32> <cases.json> [maxNumTokens]\n".data(using: .utf8)!)
  exit(2)
}
let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: URL(fileURLWithPath: args[4])))

ExperimentalFlags.optIntoExperimentalAPIs()
ExperimentalFlags.enableBenchmark = true
let mtp = ProcessInfo.processInfo.environment["MTP"] == "1"
ExperimentalFlags.enableSpeculativeDecoding = mtp

let cacheDir = URL(fileURLWithPath: args[1]).deletingLastPathComponent().appendingPathComponent("cache").path
try FileManager.default.createDirectory(atPath: cacheDir, withIntermediateDirectories: true)
let loadStarted = Date()
let engine = Engine(
  engineConfig: try EngineConfig(
    modelPath: args[1], backend: args[2] == "gpu" ? .gpu : .cpu(), maxNumTokens: args.count == 6 ? Int(args[5])! : 8_192,
    cacheDir: cacheDir, activationDataType: args[3] == "fp32" ? .float32 : nil))
try await engine.initialize()
FileHandle.standardError.write("engine loaded in \(String(format: "%.1f", Date().timeIntervalSince(loadStarted))) s\n".data(using: .utf8)!)

for item in cases {
  let digits = referenceDigits(seed: item.seed)
  let reference = "REFERENCE: \(digits)"
  let body = item.late
    ? filler(sentences: item.sentences) + "\n\n" + reference
    : reference + "\n\n" + filler(sentences: item.sentences)
  let prompt = body + "\n\nCopy the REFERENCE number exactly, digit for digit. Reply with only the digits."
  let conversation = try await engine.createConversation(
    with: ConversationConfig(samplerConfig: try SamplerConfig(topK: 1, topP: 1, temperature: 0)))
  let started = Date()
  let reply = try await conversation.sendMessage(LiteRTLM.Message(prompt), maxOutputTokens: 80).toString
  let info = try conversation.getBenchmarkInfo()
  let got = Array(reply.filter(\.isNumber))
  let want = Array(digits)
  var mismatch: Int? = nil
  if got != want {
    mismatch = zip(got, want).enumerated().first { $0.element.0 != $0.element.1 }?.offset
      ?? min(got.count, want.count)
  }
  var result: [String: Any] = [
    "id": item.id, "backend": args[2], "activation": args[3], "model": URL(fileURLWithPath: args[1]).lastPathComponent,
    "sentences": item.sentences, "late": item.late, "seed": item.seed,
    "prefill": info.lastPrefillTokenCount, "decode": info.lastDecodeTokenCount,
    "exact": mismatch == nil, "expected": digits, "output": reply,
    "seconds": (Date().timeIntervalSince(started) * 100).rounded() / 100, "mtp": mtp,
    "prefillTPS": (info.lastPrefillTokensPerSecond * 10).rounded() / 10,
    "decodeTPS": (info.lastDecodeTokensPerSecond * 10).rounded() / 10,
  ]
  if let mismatch {
    result["firstMismatch"] = mismatch
    result["firstMismatchPosition"] = info.lastPrefillTokenCount - 1 + mismatch
  }
  let line = try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys])
  print(String(data: line, encoding: .utf8)!)
  fflush(stdout)
}
