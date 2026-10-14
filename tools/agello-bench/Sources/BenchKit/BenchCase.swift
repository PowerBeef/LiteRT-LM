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
  }

  public var id: String
  public var kind: Kind
  /// Filler sentences, about 15.8 tokens each.
  public var sentences: Int
  public var late: Bool
  public var seed: Int
  /// Tokens to generate in a speed case (digit copies stop by themselves).
  public var decode: Int?

  public init(
    id: String, kind: Kind = .digitCopy, sentences: Int, late: Bool = false, seed: Int = 1,
    decode: Int? = nil
  ) {
    self.id = id
    self.kind = kind
    self.sentences = sentences
    self.late = late
    self.seed = seed
    self.decode = decode
  }

  public init(from decoder: any Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    id = try c.decode(String.self, forKey: .id)
    kind = try c.decodeIfPresent(Kind.self, forKey: .kind) ?? .digitCopy
    sentences = try c.decode(Int.self, forKey: .sentences)
    late = try c.decodeIfPresent(Bool.self, forKey: .late) ?? false
    seed = try c.decodeIfPresent(Int.self, forKey: .seed) ?? 1
    decode = try c.decodeIfPresent(Int.self, forKey: .decode)
  }

  /// Load a suite: a JSON array of cases.
  public static func load(suite url: URL) throws -> [BenchCase] {
    try JSONDecoder().decode([BenchCase].self, from: Data(contentsOf: url))
  }
}
