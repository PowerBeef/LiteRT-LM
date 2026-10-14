import Foundation

/// The prompts behind each case kind. The digit-copy text is the one Agello's
/// probe 10 and the #3913 research runner used, so results stay comparable.
public enum Prompts {
  /// The 48 digits a digit-copy case asks for.
  public static func referenceDigits(seed: Int) -> String {
    var state = UInt64(seed) &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
    return String(
      (0..<48).map { _ -> Character in
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Character(String((state >> 33) % 10))
      })
  }

  /// Plain sentences about a harbour town, about 15.8 tokens each.
  public static func filler(sentences: Int) -> String {
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

  /// A digit-copy prompt and the digits its reply must contain.
  public static func digitCopy(_ item: BenchCase) -> (prompt: String, digits: String) {
    let digits = referenceDigits(seed: item.seed)
    let reference = "REFERENCE: \(digits)"
    let body =
      item.late
      ? filler(sentences: item.sentences) + "\n\n" + reference
      : reference + "\n\n" + filler(sentences: item.sentences)
    return (
      body + "\n\nCopy the REFERENCE number exactly, digit for digit. Reply with only the digits.",
      digits
    )
  }

  /// A speed prompt: the filler, then a request long enough to fill any decode budget.
  public static func speed(_ item: BenchCase) -> String {
    filler(sentences: item.sentences)
      + "\n\nContinue these notes with new sentences in the same plain style. Write at least 1,500 words."
  }

  /// The index of the first wrong digit, or nil when the reply's digits are
  /// exactly `digits`. A reply that stops early is wrong where it stops.
  public static func firstMismatch(reply: String, digits: String) -> Int? {
    let got = Array(reply.filter(\.isNumber))
    let want = Array(digits)
    guard got != want else { return nil }
    return zip(got, want).enumerated().first { $0.element.0 != $0.element.1 }?.offset
      ?? min(got.count, want.count)
  }
}
