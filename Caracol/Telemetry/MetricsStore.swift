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

/// Thread-safe metrics collector. Counters are touched from capture, render and
/// ML callbacks; published values are only mutated on the main thread so the
/// SwiftUI HUD stays consistent.
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

    private let lock = NSLock()
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
        lock.lock()
        sourceMeter.record(time)
        let fps = sourceMeter.value(at: time)
        lock.unlock()
        publish { self.sourceFPS = fps }
    }

    func notePresented(kind: PresentedKind, latency: Double) {
        lock.lock()
        switch kind {
        case .real: realCounter += 1
        case .generated: generatedCounter += 1
        case .duplicate: break
        }
        presentedCounter += 1
        // Latency is an input-to-photon measure for real frames. Generated
        // frames are shown later by design and would distort the p50/p95 that
        // the controller and HUD use to make decisions.
        if kind == .real, latency > 0 {
            latencyRing.append(latency * 1000.0)
        }
        lock.unlock()
    }

    func noteFrameTime(ms: Double) {
        lock.lock(); _ = frameTimeEMA.update(ms); lock.unlock()
    }

    func noteFlowTime(ms: Double) {
        lock.lock(); _ = flowEMA.update(ms); lock.unlock()
    }

    func noteGPUTime(ms: Double) {
        lock.lock(); _ = gpuEMA.update(ms); lock.unlock()
    }

    func noteInferenceTime(ms: Double) {
        lock.lock(); _ = aneEMA.update(ms); lock.unlock()
    }

    func setStatus(_ text: String) {
        publish { self.status = text }
    }

    func setSource(_ info: CaptureSourceInfo) {
        publish { self.sourceTitle = info.title }
    }

    func setMode(_ settings: CaracolSettings) {
        publish {
            self.modeLabel = settings.mode.label
            self.scaleLabel = settings.upscale.label
            self.targetFPS = settings.targetFPS
        }
    }

    func setCalibration(progress: Double, text: String) {
        publish {
            self.calibrationProgress = clamp(progress, 0, 1)
            self.calibrationText = text
        }
    }

    func finishCalibration() {
        publish {
            self.calibrationProgress = 1
            self.calibrationText = "Ready"
        }
    }

    /// Called from the render loop at roughly 4 Hz.
    func tick(force: Bool = false) {
        let now = hostNow()
        lock.lock()
        guard force || now - lastTick >= 0.25 else {
            lock.unlock()
            return
        }
        let elapsed = max(0.001, now - intervalStart)
        let real = Double(realCounter) / elapsed
        let generated = Double(generatedCounter) / elapsed
        let presented = Double(presentedCounter) / elapsed
        let frame = frameTimeEMA.value ?? 0
        let gpu = gpuEMA.value ?? 0
        let flow = flowEMA.value ?? 0
        let ane = aneEMA.value ?? 0
        let latencies = latencyRing.elements
        let p50 = Stats.percentile(latencies, 0.5)
        let p95 = Stats.percentile(latencies, 0.95)
        let source = sourceMeter.value(at: now)
        realCounter = 0
        generatedCounter = 0
        presentedCounter = 0
        intervalStart = now
        lastTick = now
        lock.unlock()

        let system = SystemMetrics.shared.sample()
        publish {
            self.realFPS = real
            self.generatedFPS = generated
            self.presentedFPS = presented
            self.frameTimeMs = frame
            self.gpuMs = gpu
            self.flowMs = flow
            self.aneMs = ane
            self.latencyP50 = p50
            self.latencyP95 = p95
            self.sourceFPS = source
            self.cpuPercent = system.cpuPercent
            self.gpuPercent = system.gpuPercent
            self.memoryMB = system.memoryMB
            self.thermal = system.thermalState.shortName
        }
    }

    func reset() {
        lock.lock()
        sourceMeter.reset()
        latencyRing.removeAll()
        realCounter = 0
        generatedCounter = 0
        presentedCounter = 0
        intervalStart = hostNow()
        lastTick = intervalStart
        frameTimeEMA.reset()
        gpuEMA.reset()
        flowEMA.reset()
        aneEMA.reset()
        lock.unlock()
        publish {
            self.cpuPercent = 0
            self.gpuPercent = 0
            self.memoryMB = 0
            self.latencyP50 = 0
            self.latencyP95 = 0
            self.realFPS = 0
            self.generatedFPS = 0
            self.presentedFPS = 0
            self.frameTimeMs = 0
            self.gpuMs = 0
            self.flowMs = 0
            self.aneMs = 0
            self.sourceFPS = 0
            self.calibrationProgress = 0
            self.calibrationText = ""
        }
    }

    private func publish(_ block: @escaping () -> Void) {
        if Thread.isMainThread {
            block()
        } else {
            DispatchQueue.main.async(execute: block)
        }
    }
}

enum PresentedKind {
    case real
    case generated
    case duplicate
}
