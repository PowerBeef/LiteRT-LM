import Foundation
import XCTest

@testable import BenchKit

final class BenchKitTests: XCTestCase {
  /// The digits the #3913 research runner asked for (results/mac/short__*.jsonl).
  func testDigitsMatchTheResearchRunner() {
    XCTAssertEqual(
      Prompts.referenceDigits(seed: 1), "360450296322042052577609842187885426869520014813")
    XCTAssertEqual(
      Prompts.referenceDigits(seed: 2), "264598852324464436528445692378236997464565306650")
  }

  func testDigitCopyPromptPlacesTheReference() {
    let early = Prompts.digitCopy(BenchCase(id: "a", sentences: 2, seed: 1))
    XCTAssertTrue(early.prompt.hasPrefix("REFERENCE: \(early.digits)\n\nThe harbour master kept"))
    XCTAssertTrue(early.prompt.hasSuffix("Reply with only the digits."))
    let late = Prompts.digitCopy(BenchCase(id: "b", sentences: 2, late: true, seed: 1))
    XCTAssertTrue(late.prompt.contains("sunrise.\n\nREFERENCE: \(late.digits)\n\nCopy"))
  }

  func testFirstMismatch() {
    let digits = Prompts.referenceDigits(seed: 3)
    XCTAssertNil(Prompts.firstMismatch(reply: digits, digits: digits))
    XCTAssertNil(Prompts.firstMismatch(reply: "Sure: \(digits).", digits: digits))
    var wrong = Array(digits)
    wrong[5] = wrong[5] == "1" ? "2" : "1"
    XCTAssertEqual(Prompts.firstMismatch(reply: String(wrong), digits: digits), 5)
    XCTAssertEqual(Prompts.firstMismatch(reply: String(digits.prefix(30)), digits: digits), 30)
    XCTAssertEqual(Prompts.firstMismatch(reply: digits + "7", digits: digits), 48)
  }

  func testResearchCasesLoadAsDigitCopies() throws {
    let json = #"[{"id": "s10-seed1", "sentences": 10, "late": false, "seed": 1}, {"id": "t2k", "kind": "speed", "sentences": 129, "decode": 128}]"#
    let cases = try JSONDecoder().decode([BenchCase].self, from: Data(json.utf8))
    XCTAssertEqual(cases[0], BenchCase(id: "s10-seed1", sentences: 10, seed: 1))
    XCTAssertEqual(cases[1], BenchCase(id: "t2k", kind: .speed, sentences: 129, decode: 128))
  }

  func testRecordKeepsTheResearchKeys() throws {
    let machine = Machine(
      hardware: "Mac18,5", machine: "arm64", chip: "Apple M6", os: "macOS 27", memoryGB: 17,
      isiOSAppOnMac: false, lowPowerMode: false, gpu: "Apple M6", gpuAppleFamily: 11,
      gpuWorkingSetGB: 12.7, runtime: "", runtimeSHA256: "")
    let record = CaseRecord(
      id: "t2k", backend: "gpu", activation: "default", model: "m.litertlm", sentences: 129,
      late: false, seed: 1, prefill: 2050, decode: 128, exact: nil, expected: nil, output: "…",
      seconds: 4, mtp: false, prefillTPS: 900, decodeTPS: 40, firstMismatch: nil,
      firstMismatchPosition: nil, kind: "speed", label: "arm", run: "r", rep: 0, warmup: false,
      kv: 8192, ttft: 2.3, thermal: "nominal", footprintMB: 2500, peakFootprintMB: 2600,
      loadSeconds: 7, cache: "warm", modelSHA256: nil, machine: machine)
    let object = try XCTUnwrap(
      JSONSerialization.jsonObject(with: Data(try record.jsonLine().utf8)) as? [String: Any])
    for key in ["id", "backend", "activation", "model", "prefill", "decode", "prefillTPS", "decodeTPS", "mtp"] {
      XCTAssertNotNil(object[key], key)
    }
    XCTAssertNil(object["exact"], "Speed cases have no exact key")
  }
}
