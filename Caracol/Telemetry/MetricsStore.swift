import Foundation
import Combine

struct FPSMeter {
    private var timestamps: [CFTimeInterval] = []
    private let window: CFTimeInterval

    init(window: CFTimeInterval = 1.0) {
        self.window = window
    }

    mutating func record(_ time: CFTimeInterval = hostNow()) {
        timestamps.append(time)
        if timestamps.count > 512 { timestamps.removeFirst(timestamps.count - 512) }
    }

    mutating func value(at time: CFTimeInterval = hostNow()) -> Double {
        let cutoff = time - window
        if let firstValid = timestamps.firstIndex(where: { $0 >= cutoff }) {
            if firstValid > 0 { timestamps.removeFirst(firstValid) }
        } else {
            timestamps.removeAll(keepingCapacity: true)
        }
        return Double(timestamps.count) / window
    }

    mutating func reset() { timestamps.removeAll(keepingCapacity: true) }
}

@MainActor
final class MetricsStore: ObservableObject {
    @Published private(set) var sourceFPS: Double = 0
    @Published private(set) var realFPS: Double = 0
    @Published private(set) var generatedFPS: Double = 0
    @Published private(set) var presentedFPS: Double = 0
    @Published private(set) var frameTimeMs: Double = 0
    @Published private(set) var gpuMs: Double = 0
    @Published private(set) var flowMs: Double = 0
    @Published private(set) var aneMs: Double = 0
    @Published private(set) var cpuPercent: Double = 0
    @Published private(set) var gpuPercent: Double = 0
    @Published private(set) var memoryMB: Double = 0
    @Published private(set) var latencyP50: Double = 0
    @Published private(set) var latencyP95: Double = 0
    @Published private(set) var thermal: String = "OK"
    @Published private(set) var modeLabel: String = "Fast"
    @Published private(set) var scaleLabel: String = "2×"
    @Published private(set) var targetFPS: Int = 60
    @Published private(set) var status: String = "Idle"
    @Published private(set) var sourceTitle: String = ""
    @Published private(set) var calibrationProgress: Double = 0
    @Published private(set) var calibrationText: String = ""

    private var sourceMeter = FPSMeter()
    private var realCounter = 0
    private var generatedCounter = 0
    private var presentedCounter = 0
    private var intervalStart = hostNow()
    private var latencyRing = RingBuffer<Double>(capacity: 240)
    private var frameTimeEMA = EMA(alpha: 0.15)
    private var gpuEMA = EMA(alpha: 0.2)
    private var flowEMA = EMA(alpha: 0.2)
    private var aneEMA = EMA(alpha: 0.2)
    private var lastTick = hostNow()

    func noteSourceFrame(at time: CFTimeInterval = hostNow()) {
        sourceMeter.record(time)
        sourceFPS = sourceMeter.value(at: time)
    }

    func notePresented(kind: PresentedKind, latency: Double) {
        switch kind {
        case .real: realCounter += 1
        case .generated: generatedCounter += 1
        }
        presentedCounter += 1
        if latency > 0 {
            latencyRing.append(latency * 1000.0)
        }
    }

    func noteFrameTime(ms: Double) {
        frameTimeEMA.update(ms)
    }

    func noteFlowTime(ms: Double) {
        flowEMA.update(ms)
    }

    func noteGPUTime(ms: Double) {
        gpuEMA.update(ms)
    }

    func noteInferenceTime(ms: Double) {
        aneEMA.update(ms)
    }

    func setStatus(_ text: String) {
        status = text
    }

    func setSource(_ info: CaptureSourceInfo) {
        sourceTitle = info.title
    }

    func setMode(_ settings: CaracolSettings) {
        modeLabel = settings.mode.label
        scaleLabel = settings.upscale.label
        targetFPS = settings.targetFPS
    }

    func setCalibration(progress: Double, text: String) {
        calibrationProgress = clamp(progress, 0, 1)
        calibrationText = text
    }

    func finishCalibration() {
        calibrationProgress = 1
        calibrationText = "Ready"
    }

    /// Called from the render loop at roughly 4 Hz.
    func tick(force: Bool = false) {
        let now = hostNow()
        guard force || now - lastTick >= 0.25 else { return }
        let elapsed = max(0.001, now - intervalStart)

        realFPS = Double(realCounter) / elapsed
        generatedFPS = Double(generatedCounter) / elapsed
        presentedFPS = Double(presentedCounter) / elapsed
        frameTimeMs = frameTimeEMA.value ?? 0
        gpuMs = gpuEMA.value ?? 0
        flowMs = flowEMA.value ?? 0
        aneMs = aneEMA.value ?? 0

        let latencies = latencyRing.elements
        latencyP50 = Stats.percentile(latencies, 0.5)
        latencyP95 = Stats.percentile(latencies, 0.95)

        let system = SystemMetrics.shared.sample()
        cpuPercent = system.cpuPercent
        gpuPercent = system.gpuPercent
        memoryMB = system.memoryMB
        thermal = system.thermalState.shortName

        sourceFPS = sourceMeter.value(at: now)

        realCounter = 0
        generatedCounter = 0
        presentedCounter = 0
        intervalStart = now
        lastTick = now
    }

    func reset() {
        sourceMeter.reset()
        latencyRing.removeAll()
        realCounter = 0
        generatedCounter = 0
        presentedCounter = 0
        intervalStart = hostNow()
        lastTick = intervalStart
        cpuPercent = 0
        gpuPercent = 0
        memoryMB = 0
        latencyP50 = 0
        latencyP95 = 0
        realFPS = 0
        generatedFPS = 0
        presentedFPS = 0
        frameTimeMs = 0
        gpuMs = 0
        flowMs = 0
        aneMs = 0
    }
}

enum PresentedKind {
    case real
    case generated
}
