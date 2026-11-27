import Foundation

struct BenchmarkSample: Codable {
    var t: Double
    var sourceFPS: Double
    var realFPS: Double
    var generatedFPS: Double
    var presentedFPS: Double
    var latencyP50: Double
    var latencyP95: Double
    var gpuMs: Double
    var flowMs: Double
    var aneMs: Double
    var cpuPercent: Double
    var gpuPercent: Double
    var memoryMB: Double
    var thermal: String
}

struct BenchmarkResult: Codable {
    var startedAt: Date
    var duration: Double
    var source: String
    var settings: CaracolSettings
    var samples: [BenchmarkSample]
    var averageSourceFPS: Double
    var averageRealFPS: Double
    var averageGeneratedFPS: Double
    var averagePresentedFPS: Double
    var latencyP50: Double
    var latencyP95: Double
    var averageGPUms: Double
    var averageFlowms: Double
    var averageANEms: Double
    var averageCPU: Double
    var averageGPU: Double
    var peakMemoryMB: Double
    var thermal: String

    var fileURL: URL?

    var summary: String {
        String(
            format: "%.0f fps source · %.0f real + %.0f generated = %.0f shown\nlatency p50 %.1f ms / p95 %.1f ms · GPU %.1f · flow %.1f · ANE %.1f ms\nCPU %.0f%% · GPU %.0f%% · peak %.0f MB · %@",
            averageSourceFPS,
            averageRealFPS,
            averageGeneratedFPS,
            averagePresentedFPS,
            latencyP50,
            latencyP95,
            averageGPUms,
            averageFlowms,
            averageANEms,
            averageCPU,
            averageGPU,
            peakMemoryMB,
            thermal
        )
    }
}

@MainActor
final class BenchmarkRunner {
    private var timer: Timer?
    private var samples: [BenchmarkSample] = []
    private var startedAt = Date()
    private var startTime: CFTimeInterval = 0
    private var duration: Double = 20
    private var metrics: MetricsStore?
    private var settings: CaracolSettings = CaracolSettings()
    private var source = ""
    private var completion: ((BenchmarkResult) -> Void)?
    private var progress: ((Double, String) -> Void)?

    var isRunning: Bool { timer != nil }

    func start(duration: Double,
               metrics: MetricsStore,
               settings: CaracolSettings,
               source: String,
               progress: @escaping (Double, String) -> Void,
               completion: @escaping (BenchmarkResult) -> Void) {
        cancel()
        self.duration = max(3, duration)
        self.metrics = metrics
        self.settings = settings
        self.source = source
        self.progress = progress
        self.completion = completion
        self.startedAt = Date()
        self.startTime = hostNow()
        self.samples = []

        let timer = Timer(timeInterval: 0.5, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated {
                self?.tick()
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        self.timer = timer
        LogStore.info("benchmark", "Started \(Int(self.duration)) s benchmark for \(source)")
        tick()
    }

    func cancel() {
        if timer != nil {
            timer?.invalidate()
            timer = nil
            LogStore.info("benchmark", "Cancelled")
        }
        progress = nil
        completion = nil
    }

    private func tick() {
        guard let metrics else { return }
        let now = hostNow()
        let elapsed = now - startTime
        metrics.tick(force: true)
        samples.append(BenchmarkSample(
            t: elapsed,
            sourceFPS: metrics.sourceFPS,
            realFPS: metrics.realFPS,
            generatedFPS: metrics.generatedFPS,
            presentedFPS: metrics.presentedFPS,
            latencyP50: metrics.latencyP50,
            latencyP95: metrics.latencyP95,
            gpuMs: metrics.gpuMs,
            flowMs: metrics.flowMs,
            aneMs: metrics.aneMs,
            cpuPercent: metrics.cpuPercent,
            gpuPercent: metrics.gpuPercent,
            memoryMB: metrics.memoryMB,
            thermal: metrics.thermal
        ))
        progress?(min(1, elapsed / duration), "Benchmark \(Int(elapsed)) / \(Int(duration)) s")
        if elapsed >= duration {
            finish()
        }
    }

    private func finish() {
        timer?.invalidate()
        timer = nil
        guard !samples.isEmpty else {
            completion?(emptyResult())
            return
        }

        let count = Double(samples.count)
        let result = BenchmarkResult(
            startedAt: startedAt,
            duration: hostNow() - startTime,
            source: source,
            settings: settings,
            samples: samples,
            averageSourceFPS: samples.reduce(0) { $0 + $1.sourceFPS } / count,
            averageRealFPS: samples.reduce(0) { $0 + $1.realFPS } / count,
            averageGeneratedFPS: samples.reduce(0) { $0 + $1.generatedFPS } / count,
            averagePresentedFPS: samples.reduce(0) { $0 + $1.presentedFPS } / count,
            latencyP50: Stats.percentile(samples.map(\.latencyP50), 0.5),
            latencyP95: Stats.percentile(samples.map(\.latencyP95), 0.95),
            averageGPUms: samples.reduce(0) { $0 + $1.gpuMs } / count,
            averageFlowms: samples.reduce(0) { $0 + $1.flowMs } / count,
            averageANEms: samples.reduce(0) { $0 + $1.aneMs } / count,
            averageCPU: samples.reduce(0) { $0 + $1.cpuPercent } / count,
            averageGPU: samples.reduce(0) { $0 + $1.gpuPercent } / count,
            peakMemoryMB: samples.map(\.memoryMB).max() ?? 0,
            thermal: samples.last?.thermal ?? "OK",
            fileURL: nil
        )
        let stored = store(result)
        LogStore.info("benchmark", "Finished: \(stored.summary.replacingOccurrences(of: "\n", with: " | "))")
        let completion = self.completion
        self.completion = nil
        self.progress = nil
        completion?(stored)
    }

    private func emptyResult() -> BenchmarkResult {
        BenchmarkResult(
            startedAt: startedAt,
            duration: 0,
            source: source,
            settings: settings,
            samples: [],
            averageSourceFPS: 0,
            averageRealFPS: 0,
            averageGeneratedFPS: 0,
            averagePresentedFPS: 0,
            latencyP50: 0,
            latencyP95: 0,
            averageGPUms: 0,
            averageFlowms: 0,
            averageANEms: 0,
            averageCPU: 0,
            averageGPU: 0,
            peakMemoryMB: 0,
            thermal: "?",
            fileURL: nil
        )
    }

    private func store(_ result: BenchmarkResult) -> BenchmarkResult {
        var result = result
        let fileManager = FileManager.default
        guard let base = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            return result
        }
        let directory = base.appendingPathComponent("Caracol/Benchmarks", isDirectory: true)
        try? fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let url = directory.appendingPathComponent("benchmark-\(formatter.string(from: Date())).json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        if let data = try? encoder.encode(result) {
            try? data.write(to: url, options: .atomic)
            result.fileURL = url
        }
        return result
    }
}
