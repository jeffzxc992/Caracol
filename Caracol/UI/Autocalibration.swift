import Foundation

struct FlowModelInfo {
    var trained: Bool = false
    var width: Int = 256
    var height: Int = 144

    static func load() -> FlowModelInfo {
        guard let url = Bundle.main.url(forResource: "IFRNetLite.model-info", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return FlowModelInfo()
        }
        var info = FlowModelInfo()
        info.trained = object["trained"] as? Bool ?? false
        info.width = object["width"] as? Int ?? 256
        info.height = object["height"] as? Int ?? 144
        return info
    }
}

/// Runs a short, measurable calibration: fast GPU flow at two resolutions,
/// then the Core ML route when a trained model is present. It changes settings
/// live so the measurement reflects the real capture + presentation pipeline.
@MainActor
final class Autocalibrator {
    private struct Sample {
        var sourceFPS: Double = 0
        var realFPS: Double = 0
        var generatedFPS: Double = 0
        var latencyP50: Double = 0
        var latencyP95: Double = 0
        var gpuMs: Double = 0
        var flowMs: Double = 0
        var aneMs: Double = 0
        var cpu: Double = 0
        var gpu: Double = 0
        var ram: Double = 0
        var thermal: String = "OK"

        static func from(_ metrics: MetricsStore) -> Sample {
            Sample(
                sourceFPS: metrics.sourceFPS,
                realFPS: metrics.realFPS,
                generatedFPS: metrics.generatedFPS,
                latencyP50: metrics.latencyP50,
                latencyP95: metrics.latencyP95,
                gpuMs: metrics.gpuMs,
                flowMs: metrics.flowMs,
                aneMs: metrics.aneMs,
                cpu: metrics.cpuPercent,
                gpu: metrics.gpuPercent,
                ram: metrics.memoryMB,
                thermal: metrics.thermal
            )
        }
    }

    private let metrics: MetricsStore
    private let baseSettings: CaracolSettings
    private let displayRefresh: Int
    private let modelInfo: FlowModelInfo
    private let apply: (CaracolSettings) -> Void
    private let progress: (Double, String) -> Void
    private let completion: (CaracolSettings, String) -> Void

    private var timer: Timer?
    private var startTime: CFTimeInterval = 0
    private var stageStart: CFTimeInterval = 0
    private var stage = 0
    private var warmupSamples: [Sample] = []
    private var quarterSamples: [Sample] = []
    private var eighthSamples: [Sample] = []
    private var qualitySamples: [Sample] = []
    private var currentSettings: CaracolSettings

    private let warmupDuration: CFTimeInterval = 2.0
    private let quarterDuration: CFTimeInterval = 3.0
    private let eighthDuration: CFTimeInterval = 1.5
    private let qualityDuration: CFTimeInterval = 2.5

    init(settings: CaracolSettings,
         displayRefresh: Int,
         metrics: MetricsStore,
         modelInfo: FlowModelInfo = .load(),
         apply: @escaping (CaracolSettings) -> Void,
         progress: @escaping (Double, String) -> Void,
         completion: @escaping (CaracolSettings, String) -> Void) {
        self.baseSettings = settings
        self.currentSettings = settings
        self.displayRefresh = displayRefresh
        self.metrics = metrics
        self.modelInfo = modelInfo
        self.apply = apply
        self.progress = progress
        self.completion = completion
    }

    var includesQualityStage: Bool {
        modelInfo.trained &&
            Bundle.main.url(forResource: "IFRNetLite", withExtension: "mlmodelc") != nil
    }

    func start() {
        guard timer == nil else { return }
        startTime = hostNow()
        stageStart = startTime
        stage = 0
        warmupSamples.removeAll()
        quarterSamples.removeAll()
        eighthSamples.removeAll()
        qualitySamples.removeAll()

        var baseline = baseSettings
        baseline.mode = .fast
        baseline.targetFPS = recommendedTarget
        baseline.upscale = .x2
        baseline.flowQuality = .quarter
        baseline.frameGeneration = true
        baseline.interpolation = false
        baseline.inferenceRoute = .gpuOnly
        currentSettings = baseline
        apply(baseline)
        progress(0.02, "Calibrating: measuring source")
        LogStore.info("calibrate", "Started (source \(metrics.sourceFPS) fps, display \(displayRefresh) Hz, trained model \(modelInfo.trained))")

        let timer = Timer(timeInterval: 0.25, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.tick()
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
    }

    func cancel() {
        timer?.invalidate()
        timer = nil
        LogStore.info("calibrate", "Cancelled")
    }

    private func tick() {
        let now = hostNow()
        let stageElapsed = now - stageStart
        let totalEstimate = warmupDuration + quarterDuration + eighthDuration +
            (includesQualityStage ? qualityDuration : 0)

        switch stage {
        case 0:
            warmupSamples.append(.from(metrics))
            let fraction = 0.05 + 0.15 * clamp(stageElapsed / warmupDuration, 0, 1)
            progress(fraction, String(format: "Calibrating: source %.0f fps", metrics.sourceFPS))
            if stageElapsed >= warmupDuration {
                enterQuarter(now: now)
            }
        case 1:
            quarterSamples.append(.from(metrics))
            let fraction = 0.20 + 0.35 * clamp(stageElapsed / quarterDuration, 0, 1)
            progress(fraction, "Calibrating: GPU flow 1/4")
            if stageElapsed >= quarterDuration {
                enterEighth(now: now)
            }
        case 2:
            eighthSamples.append(.from(metrics))
            let base = 0.55
            let span = includesQualityStage ? 0.20 : 0.43
            let fraction = base + span * clamp(stageElapsed / eighthDuration, 0, 1)
            progress(fraction, "Calibrating: GPU flow 1/8")
            if stageElapsed >= eighthDuration {
                if includesQualityStage {
                    enterQuality(now: now)
                } else {
                    finish()
                }
            }
        case 3:
            qualitySamples.append(.from(metrics))
            let fraction = 0.75 + 0.23 * clamp(stageElapsed / qualityDuration, 0, 1)
            progress(fraction, String(format: "Calibrating: Neural Engine %.1f ms", metrics.aneMs))
            if stageElapsed >= qualityDuration {
                finish()
            }
        default:
            finish()
        }
        _ = totalEstimate
    }

    private func enterQuarter(now: CFTimeInterval) {
        stage = 1
        stageStart = now
        var settings = currentSettings
        settings.mode = .fast
        settings.flowQuality = .quarter
        settings.upscale = .x2
        settings.inferenceRoute = .gpuOnly
        currentSettings = settings
        apply(settings)
        LogStore.info("calibrate", "Stage: fast flow 1/4")
    }

    private func enterEighth(now: CFTimeInterval) {
        stage = 2
        stageStart = now
        var settings = currentSettings
        settings.flowQuality = .eighth
        currentSettings = settings
        apply(settings)
        LogStore.info("calibrate", "Stage: fast flow 1/8")
    }

    private func enterQuality(now: CFTimeInterval) {
        stage = 3
        stageStart = now
        var settings = currentSettings
        settings.mode = .quality
        settings.flowQuality = .quarter
        settings.inferenceRoute = .coreML
        currentSettings = settings
        apply(settings)
        LogStore.info("calibrate", "Stage: Core ML quality")
    }

    private var recommendedTarget: Int {
        let source = metrics.sourceFPS > 1 ? metrics.sourceFPS : baseSettings.sourceFPSGuess
        let cap = Double(min(60, max(30, displayRefresh)))
        let desired = min(cap, max(source * 2.0, source + 1.0))
        return nearestFPSChoice(desired)
    }

    private func finish() {
        timer?.invalidate()
        timer = nil

        let source = median(warmupSamples.map(\.sourceFPS).filter { $0 > 1 })
        let quarter = average(quarterSamples)
        let eighth = average(eighthSamples)
        let quality = average(qualitySamples)

        var result = baseSettings
        result.targetFPS = nearestFPSChoice(min(60, max(30, (source > 1 ? source * 2 : Double(result.targetFPS)))))
        if displayRefresh > 0 {
            result.targetFPS = min(result.targetFPS, min(60, displayRefresh))
        }
        result.frameGeneration = true
        result.interpolation = false

        let quarterOK = !quarterSamples.isEmpty &&
            (quarter?.flowMs ?? 99) <= 3.2 &&
            (quarter?.latencyP95 ?? 99) <= 34
        result.flowQuality = quarterOK ? .quarter : .eighth

        let gpuMs = quarter?.gpuMs ?? eighth?.gpuMs ?? result.sharpness * 0 + 6
        if (quarter?.thermal == "Critical") || (quarter?.thermal == "Hot") {
            result.upscale = .off
            result.targetFPS = min(result.targetFPS, 30)
        } else if gpuMs <= 4.5 {
            result.upscale = .x2
        } else if gpuMs <= 7.0 {
            result.upscale = .x15
        } else {
            result.upscale = .off
        }

        let qualityUsable = modelInfo.trained &&
            !qualitySamples.isEmpty &&
            (quality?.aneMs ?? 99) > 0 &&
            (quality?.aneMs ?? 99) <= 3.5 &&
            (quality?.gpuMs ?? 99) <= 6.0 &&
            (quality?.latencyP95 ?? 99) <= 40
        result.mode = qualityUsable ? .quality : .fast
        result.inferenceRoute = qualityUsable ? .coreML : .gpuOnly
        result.calibrated = true
        currentSettings = result
        apply(result)
        metrics.finishCalibration()

        var notes: [String] = []
        notes.append(String(format: "source %.0f fps", source > 1 ? source : metrics.sourceFPS))
        notes.append("target \(result.targetFPS)")
        notes.append(result.mode.label)
        notes.append(result.upscale.label)
        notes.append("flow \(result.flowQuality.label)")
        if let quarter { notes.append(String(format: "flow %.1f ms", quarter.flowMs)) }
        if let quality, quality.aneMs > 0 { notes.append(String(format: "ANE %.1f ms", quality.aneMs)) }
        if !qualityUsable {
            notes.append(modelInfo.trained ? "quality too slow" : "stub model, fast selected")
        }
        let summary = notes.joined(separator: " · ")
        progress(1, "Ready · \(summary)")
        LogStore.info("calibrate", "Completed: \(summary)")

        let completion = self.completion
        let final = result
        DispatchQueue.main.async {
            completion(final, summary)
        }
    }

    private func nearestFPSChoice(_ value: Double) -> Int {
        let choices = CaracolSettings.targetFPSChoices
        return choices.min(by: { abs(Double($0) - value) < abs(Double($1) - value) }) ?? 60
    }

    private func average(_ samples: [Sample]) -> Sample? {
        guard !samples.isEmpty else { return nil }
        let count = Double(samples.count)
        var result = Sample()
        result.sourceFPS = samples.reduce(0) { $0 + $1.sourceFPS } / count
        result.realFPS = samples.reduce(0) { $0 + $1.realFPS } / count
        result.generatedFPS = samples.reduce(0) { $0 + $1.generatedFPS } / count
        result.latencyP50 = samples.reduce(0) { $0 + $1.latencyP50 } / count
        result.latencyP95 = samples.reduce(0) { $0 + $1.latencyP95 } / count
        result.gpuMs = samples.reduce(0) { $0 + $1.gpuMs } / count
        result.flowMs = samples.reduce(0) { $0 + $1.flowMs } / count
        result.aneMs = samples.reduce(0) { $0 + $1.aneMs } / count
        result.cpu = samples.reduce(0) { $0 + $1.cpu } / count
        result.gpu = samples.reduce(0) { $0 + $1.gpu } / count
        result.ram = samples.reduce(0) { $0 + $1.ram } / count
        result.thermal = samples.last?.thermal ?? "OK"
        return result
    }

    private func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return metrics.sourceFPS }
        return Stats.median(values)
    }
}

private extension CaracolSettings {
    /// Best guess before the first measurement. Allows a calibration run to
    /// start before ScreenCaptureKit has delivered its first source frame.
    var sourceFPSGuess: Double {
        switch targetFPS {
        case 40, 50: return 30
        default: return 30
        }
    }
}
