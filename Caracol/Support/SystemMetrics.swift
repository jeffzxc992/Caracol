import Foundation
import IOKit
import Darwin

struct SystemSnapshot {
    var cpuPercent: Double = 0
    var gpuPercent: Double = 0
    var memoryMB: Double = 0
    var thermalState: ProcessInfo.ThermalState = .nominal
    var timestamp: CFTimeInterval = hostNow()
}

/// Low-overhead CPU / GPU / memory sampler. Call `sample()` at most a few times
/// per second from the main thread.
final class SystemMetrics {
    static let shared = SystemMetrics()

    private var lastCPU: (user: UInt32, system: UInt32, idle: UInt32, nice: UInt32)?
    private var cachedGPU: Double = 0
    private var cachedGPUAt: CFTimeInterval = 0

    private init() {}

    func sample() -> SystemSnapshot {
        var snapshot = SystemSnapshot()
        snapshot.cpuPercent = sampleCPU()
        snapshot.gpuPercent = sampleGPU()
        snapshot.memoryMB = sampleMemoryMB()
        snapshot.thermalState = ProcessInfo.processInfo.thermalState
        snapshot.timestamp = hostNow()
        return snapshot
    }

    private func sampleCPU() -> Double {
        var load = host_cpu_load_info()
        var count = mach_msg_type_number_t(MemoryLayout<host_cpu_load_info_data_t>.stride / MemoryLayout<integer_t>.stride)
        let result = withUnsafeMutablePointer(to: &load) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }

        let user = load.cpu_ticks.0
        let system = load.cpu_ticks.1
        let idle = load.cpu_ticks.2
        let nice = load.cpu_ticks.3

        defer { lastCPU = (user, system, idle, nice) }
        guard let previous = lastCPU else { return 0 }

        let dUser = Double(user &- previous.user)
        let dSystem = Double(system &- previous.system)
        let dIdle = Double(idle &- previous.idle)
        let dNice = Double(nice &- previous.nice)
        let total = dUser + dSystem + dIdle + dNice
        guard total > 0 else { return 0 }
        return clamp((dUser + dSystem + dNice) / total * 100.0, 0, 100)
    }

    private func sampleGPU() -> Double {
        let now = hostNow()
        if now - cachedGPUAt < 0.75 { return cachedGPU }
        cachedGPUAt = now

        var iterator: io_iterator_t = 0
        let matching = IOServiceMatching("IOAccelerator")
        guard IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator) == KERN_SUCCESS else {
            return cachedGPU
        }
        defer { IOObjectRelease(iterator) }

        var best: Double = 0
        var service = IOIteratorNext(iterator)
        while service != 0 {
            defer {
                IOObjectRelease(service)
                service = IOIteratorNext(iterator)
            }
            guard let property = IORegistryEntryCreateCFProperty(
                service,
                "PerformanceStatistics" as CFString,
                kCFAllocatorDefault,
                0
            )?.takeRetainedValue() as? [String: Any] else { continue }

            for key in ["Device Utilization %", "GPU Activity(%)", "Renderer Utilization %"] {
                if let value = property[key] as? NSNumber {
                    best = max(best, value.doubleValue)
                }
            }
        }
        cachedGPU = clamp(best, 0, 100)
        return cachedGPU
    }

    private func sampleMemoryMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info_data_t>.stride / MemoryLayout<natural_t>.stride)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Double(info.resident_size) / (1024.0 * 1024.0)
    }
}

extension ProcessInfo.ThermalState {
    var displayName: String {
        switch self {
        case .nominal: return "Nominal"
        case .fair: return "Fair"
        case .serious: return "Serious"
        case .critical: return "Critical"
        @unknown default: return "Unknown"
        }
    }

    var shortName: String {
        switch self {
        case .nominal: return "OK"
        case .fair: return "Warm"
        case .serious: return "Hot"
        case .critical: return "Critical"
        @unknown default: return "?"
        }
    }
}
