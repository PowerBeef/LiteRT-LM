import CryptoKit
import Darwin
import Foundation
import Metal

/// Where a run happened: hardware, OS, GPU, and the exact runtime binary.
public struct Machine: Codable, Sendable {
  /// `hw.model`: "Mac18,5" on a Mac (also for an iPhone app running there).
  public var hardware: String
  /// `hw.machine`: "arm64" on a Mac, "iPhone18,2" on an iPhone.
  public var machine: String
  /// "Apple M6" on a Mac; empty on iOS, which doesn't expose it.
  public var chip: String
  public var os: String
  public var memoryGB: Double
  public var isiOSAppOnMac: Bool
  public var lowPowerMode: Bool
  public var gpu: String
  /// The newest Apple GPU family supported: 7 for A14 and M1, 9 for A17 Pro and M3, and so on.
  public var gpuAppleFamily: Int
  public var gpuWorkingSetGB: Double
  /// The loaded LiteRT-LM C library (CLiteRTLM): the build being measured.
  public var runtime: String
  public var runtimeSHA256: String
  /// How the runtime was built (`litert_lm_get_build_info`: commits, flags, build id); nil for
  /// builds without it, such as Google's.
  public var runtimeBuild: [String: String]?

  public static func capture() -> Machine {
    let gpu = MTLCreateSystemDefaultDevice()
    let process = ProcessInfo.processInfo
    #if os(macOS)
      let platform = "macOS"
    #else
      let platform = "iOS"
    #endif
    let runtime = runtimePath() ?? ""
    return Machine(
      hardware: sysctl("hw.model"),
      machine: sysctl("hw.machine"),
      chip: sysctl("machdep.cpu.brand_string"),
      os: "\(platform) \(process.operatingSystemVersionString)",
      memoryGB: Double(process.physicalMemory) / 1e9,
      isiOSAppOnMac: process.isiOSAppOnMac,
      lowPowerMode: process.isLowPowerModeEnabled,
      gpu: gpu?.name ?? "none",
      gpuAppleFamily: gpu.map(appleFamily) ?? 0,
      gpuWorkingSetGB: Double(gpu?.recommendedMaxWorkingSetSize ?? 0) / 1e9,
      runtime: runtime,
      runtimeSHA256: runtime.isEmpty ? "" : (try? sha256(of: URL(fileURLWithPath: runtime))) ?? "",
      runtimeBuild: runtimeBuildInfo())
  }

  /// `litert_lm_get_build_info`, looked up at run time so Google's builds still load.
  static func runtimeBuildInfo() -> [String: String]? {
    typealias GetBuildInfo = @convention(c) () -> UnsafePointer<CChar>?
    guard let handle = dlopen(nil, RTLD_NOW), let address = dlsym(handle, "litert_lm_get_build_info"),
      let text = unsafeBitCast(address, to: GetBuildInfo.self)()
    else { return nil }
    return (try? JSONSerialization.jsonObject(with: Data(String(cString: text).utf8))) as? [String: String]
  }

  static func sysctl(_ name: String) -> String {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "" }
    var value = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &value, &size, nil, 0) == 0 else { return "" }
    return value.withUnsafeBufferPointer { String(cString: $0.baseAddress!) }
  }

  static func appleFamily(_ device: any MTLDevice) -> Int {
    let family = (1001...1020).last { MTLGPUFamily(rawValue: $0).map(device.supportsFamily) ?? false }
    return family.map { $0 - 1000 } ?? 0
  }

  /// The image that defines LiteRT-LM's C API: the framework, or this
  /// executable when the runtime is linked statically.
  static func runtimePath() -> String? {
    guard let handle = dlopen(nil, RTLD_NOW),
      let symbol = dlsym(handle, "litert_lm_engine_settings_create")
    else { return nil }
    var info = Dl_info()
    guard dladdr(symbol, &info) != 0, let name = info.dli_fname else { return nil }
    return URL(fileURLWithPath: String(cString: name)).resolvingSymlinksInPath().path
  }

  /// SHA-256 of a file, read in 8 MB chunks (model files are several GB).
  public static func sha256(of url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var hasher = SHA256()
    while let chunk = try handle.read(upToCount: 8 << 20), !chunk.isEmpty {
      hasher.update(data: chunk)
    }
    return hasher.finalize().map { String(format: "%02x", $0) }.joined()
  }
}

/// The process's memory as the system accounts it (GPU buffers included on
/// Apple silicon), and its peak so far.
public enum Footprint {
  public static func current() -> (bytes: UInt64, peakBytes: UInt64) {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(
      MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &info) {
      $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
        task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
      }
    }
    guard result == KERN_SUCCESS else { return (0, 0) }
    return (info.phys_footprint, UInt64(info.ledger_phys_footprint_peak))
  }
}

extension ProcessInfo.ThermalState {
  public var label: String {
    switch self {
    case .nominal: "nominal"
    case .fair: "fair"
    case .serious: "serious"
    case .critical: "critical"
    @unknown default: "unknown"
    }
  }
}
