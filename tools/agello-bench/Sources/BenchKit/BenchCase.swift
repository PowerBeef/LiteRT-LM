import Foundation

/// One measured prompt. A suite file (`suites/*.json`) holds an array of
/// cases; entries without a `kind` are digit copies, so the files in
/// `research/issue-3913/cases` load as they are.
public struct BenchCase: Codable, Sendable, Equatable {
  public enum Kind: String, Codable, Sendable {
    /// Copy a 48-digit number placed before the filler text (or after it,
    /// `late`): exact or not, and where the first wrong digit lands (#3913).
    case digitCopy = "digit-copy"
    /// Prefill the filler text, then generate `decode` tokens: prefill and
    /// decode speed at that context length.
    case speed
    /// The engine's own benchmark (`LiteRTLM.benchmark`): a fresh engine
    /// sized prefill + decode + 32 tokens, `prefill` synthetic prompt tokens,
    /// then `decode` tokens. Agello's in-app benchmark is this call.
    case engineBenchmark = "engine-benchmark"
    /// The first reply after a helper (T3): a history of `sentences` filler
    /// sentences in the main conversation, a helper prompt in a second
    /// conversation, then a short message. `keep` sends it on the main
    /// conversation (the runtime restores it); otherwise on a new conversation
    /// seeded with the history, as Agello does today. `ttft` is that message's.
    case helperTurn = "helper-turn"
  }

  public var id: String
  public var kind: Kind
  /// Filler sentences, about 15.8 tokens each.
  public var sentences: Int
  public var late: Bool
  public var seed: Int
  /// Tokens to generate in a speed case (digit copies stop by themselves).
  public var decode: Int?
  /// Prompt tokens for an engine benchmark.
  public var prefill: Int?
  /// Helper turn: keep the main conversation instead of reseeding it.
  public var keep: Bool?
  /// Helper turn: run the helper (default); false is the control without one.
  public var helper: Bool?

  public init(
    id: String, kind: Kind = .digitCopy, sentences: Int = 0, late: Bool = false, seed: Int = 1,
    decode: Int? = nil, prefill: Int? = nil
  ) {
    self.id = id
    self.kind = kind
    self.sentences = sentences
    self.late = late
    self.seed = seed
    self.decode = decode
    self.prefill = prefill
  }

  public init(from decoder: any Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    id = try c.decode(String.self, forKey: .id)
    kind = try c.decodeIfPresent(Kind.self, forKey: .kind) ?? .digitCopy
    sentences = try c.decodeIfPresent(Int.self, forKey: .sentences) ?? 0
    late = try c.decodeIfPresent(Bool.self, forKey: .late) ?? false
    seed = try c.decodeIfPresent(Int.self, forKey: .seed) ?? 1
    decode = try c.decodeIfPresent(Int.self, forKey: .decode)
    prefill = try c.decodeIfPresent(Int.self, forKey: .prefill)
    keep = try c.decodeIfPresent(Bool.self, forKey: .keep)
    helper = try c.decodeIfPresent(Bool.self, forKey: .helper)
  }

  /// Load a suite: a JSON array of cases.
  public static func load(suite url: URL) throws -> [BenchCase] {
    try JSONDecoder().decode([BenchCase].self, from: Data(contentsOf: url))
  }
}
