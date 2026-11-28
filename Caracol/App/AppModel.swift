import AppKit
import Metal
import QuartzCore
import ScreenCaptureKit
import UniformTypeIdentifiers

@MainActor
final class AppModel: NSObject, ObservableObject {
    static let shared = AppModel()

    let settingsStore: SettingsStore
    let metrics: MetricsStore
    let capture: CaptureEngine

    @Published private(set) var statusText = "Idle"
    @Published private(set) var isRunning = false
    @Published private(set) var isCalibrating = false
    @Published private(set) var calibrationSummary = ""
    @Published private(set) var overlayVisible = true
    @Published private(set) var videoTestActive = false
    @Published private(set) var benchmarkRunning = false
    @Published private(set) var benchmarkProgress: Double = 0
    @Published private(set) var benchmarkText = ""
    @Published private(set) var metalMLStatus = "Not tested"
    @Published var permissionGuide: String?

    var settings: CaracolSettings { settingsStore.settings }

    private let metalDevice: MTLDevice?
    private var renderer: Renderer?
    private var overlay: OverlayController?
    private var hud: HUDController?
    private let hotkey = GlobalHotkey()
    private let videoSource = VideoTestSource()
    private let benchmark = BenchmarkRunner()
    private var autocalibrator: Autocalibrator?
    private var activeLayout: CaptureLayout?
    private var currentOverlayRect: NSRect?
    private var calibratedSources: Set<String> = []
    private var isApplyingProfile = false
    private var isNormalizingSettings = false
    private var isStopping = false
    private var didActivate = false
    private var profileSaveWork: DispatchWorkItem?
    private var autoVideoURL: URL?
    private var autoBenchmarkSeconds: Double?
    private var autoExitAfterBenchmark = false

    override init() {
        let store = SettingsStore()
        let metrics = MetricsStore()
        self.settingsStore = store
        self.metrics = metrics
        self.capture = CaptureEngine(settings: store.settings,
                                     displayRefresh: store.settings.displayRefresh)
        self.metalDevice = MTLCreateSystemDefaultDevice()
        super.init()
        capture.delegate = self
        store.onChange = { [weak self] settings in
            MainActor.assumeIsolated {
                self?.settingsChanged(settings)
            }
        }
        hotkey.onPress = { [weak self] in
            self?.toggleOverlay()
        }
        metalMLStatus = MetalMLFlowRefiner.cachedAvailability ? "Available" : "Not tested"
    }

    func activate() {
        guard !didActivate else { return }
        didActivate = true
        let screen = NSScreen.main ?? NSScreen.screens.first
        let refresh = Int(screen?.maximumFramesPerSecond ?? 60)
        settingsStore.refreshDisplayRate(refresh > 0 ? refresh : 60)
        if settingsStore.settings.hotkeyEnabled {
            _ = hotkey.register()
        }
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(screensChanged),
            name: NSApplication.didChangeScreenParametersNotification,
            object: nil
        )
        metrics.setStatus("Choose a window")
        parseLaunchArguments()
        if let autoVideoURL {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { [weak self] in
                self?.startVideoTest(url: autoVideoURL)
            }
        } else if !settingsStore.settings.calibrated {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) { [weak self] in
                self?.chooseWindow()
            }
        }
    }

    private func parseLaunchArguments() {
        let arguments = CommandLine.arguments
        if arguments.contains("--force-fullscreen") || arguments.contains("--target-fps") {
            var overridden = settingsStore.settings
            if arguments.contains("--force-fullscreen") {
                overridden.forceFullScreen = true
            }
            if let index = arguments.firstIndex(of: "--target-fps"),
               index + 1 < arguments.count,
               let fps = Int(arguments[index + 1]) {
                overridden.targetFPS = fps
            }
            settingsStore.settings = overridden.sanitized(displayRefresh: overridden.displayRefresh)
        }
        if let index = arguments.firstIndex(of: "--video-test"),
           index + 1 < arguments.count {
            autoVideoURL = URL(fileURLWithPath: arguments[index + 1])
        }
        if let index = arguments.firstIndex(of: "--auto-benchmark"),
           index + 1 < arguments.count,
           let seconds = Double(arguments[index + 1]) {
            autoBenchmarkSeconds = clamp(seconds, 3, 120)
        }
        autoExitAfterBenchmark = arguments.contains("--auto-exit")
    }

    func shutdown() {
        stop()
        hotkey.unregister()
        NotificationCenter.default.removeObserver(self)
    }

    // MARK: - Public actions

    func chooseWindow() {
        if videoTestActive {
            videoSource.stop()
            videoTestActive = false
        }
        metrics.setStatus("Pick a window or display")
        statusText = "Pick a window or display"
        capture.presentPicker()
    }

    func stop() {
        isStopping = true
        autocalibrator?.cancel()
        autocalibrator = nil
        isCalibrating = false
        benchmark.cancel()
        benchmarkRunning = false
        benchmarkProgress = 0
        videoSource.stop()
        videoTestActive = false
        capture.stop()
        teardownPipeline()
        isRunning = false
        statusText = "Idle"
        metrics.setStatus("Idle")
        metrics.reset()
        isStopping = false
    }

    func toggleOverlay() {
        guard overlay != nil else { return }
        overlayVisible.toggle()
        overlay?.setVisible(overlayVisible)
        hud?.setVisible(overlayVisible && settings.showHUD)
        if overlayVisible {
            metrics.setStatus("Overlay visible (⌃⌥⌘G to hide)")
        } else {
            metrics.setStatus("Overlay hidden (⌃⌥⌘G to show)")
        }
    }

    func setOverlayVisible(_ visible: Bool) {
        guard overlayVisible != visible else { return }
        overlayVisible = visible
        overlay?.setVisible(visible)
        hud?.setVisible(visible && settings.showHUD)
    }

    func openScreenRecordingSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture") {
            NSWorkspace.shared.open(url)
        }
    }

    func dismissPermissionGuide() {
        permissionGuide = nil
    }

    func exportLog() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "Caracol-log.txt"
        panel.allowedContentTypes = [.plainText]
        panel.canCreateDirectories = true
        panel.begin { response in
            guard response == .OK, let url = panel.url else { return }
            do {
                try LogStore.shared.export(to: url)
                LogStore.info("app", "Log exported to \(url.path)")
            } catch {
                LogStore.error("app", "Log export failed: \(error)")
            }
        }
    }

    func openProfilesFolder() {
        let url = ProfileStore.shared.storageURL
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    func recalibrate() {
        guard isRunning, !videoTestActive, let layout = activeLayout else { return }
        autocalibrator?.cancel()
        autocalibrator = nil
        isCalibrating = false
        calibratedSources.remove(layout.profileKey)
        startCalibration()
    }

    func runMetalMLSelfTest() {
        guard renderer == nil || !isRunning else {
            metalMLStatus = "Stop capture first"
            return
        }
        metalMLStatus = "Testing in child process…"
        MetalMLFlowRefiner.runSelfTestInSubprocess { [weak self] ok in
            guard let self else { return }
            self.metalMLStatus = ok ? "Available" : "Unavailable (child crashed or errored)"
            self.renderer?.refreshMetalMLAvailability()
        }
    }

    func runBenchmark(seconds: Double = 20, autoStarted: Bool = false) {
        guard isRunning else {
            statusText = "Start a capture before benchmarking"
            return
        }
        guard !benchmarkRunning else { return }
        benchmarkRunning = true
        benchmarkProgress = 0
        benchmarkText = "Running…"
        benchmark.start(
            duration: seconds,
            metrics: metrics,
            settings: settingsStore.settings,
            source: activeLayout?.title ?? metrics.sourceTitle,
            progress: { [weak self] fraction, text in
                guard let self else { return }
                self.benchmarkProgress = fraction
                self.benchmarkText = text
            },
            completion: { [weak self] result in
                guard let self else { return }
                self.benchmarkRunning = false
                self.benchmarkProgress = 1
                self.benchmarkText = result.summary
                self.statusText = "Benchmark saved"
                if let url = result.fileURL {
                    LogStore.info("benchmark", "Saved \(url.path)")
                }
                if autoStarted {
                    print("CARACOL_BENCHMARK_BEGIN")
                    print(result.summary)
                    if let url = result.fileURL {
                        print("JSON \(url.path)")
                    }
                    print("CARACOL_BENCHMARK_END")
                    fflush(stdout)
                    if self.autoExitAfterBenchmark {
                        DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) {
                            NSApp.terminate(nil)
                        }
                    }
                }
            }
        )
    }

    func cancelBenchmark() {
        benchmark.cancel()
        benchmarkRunning = false
        benchmarkProgress = 0
        benchmarkText = "Cancelled"
    }

    func testWithRecordedVideo() {
        if videoTestActive {
            stop()
        }
        let panel = NSOpenPanel()
        panel.title = "Choose a recorded game video"
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.movie, .quickTimeMovie, .mpeg4Movie]
        NSApp.activate(ignoringOtherApps: true)
        panel.begin { [weak self] response in
            guard let self, response == .OK, let url = panel.url else { return }
            self.startVideoTest(url: url)
        }
    }

    private func startVideoTest(url: URL) {
        videoTestActive = true
        autocalibrator?.cancel()
        autocalibrator = nil
        isCalibrating = false
        capture.stop()
        teardownPipeline()
        metrics.reset()
        statusText = "Loading \(url.lastPathComponent)…"
        videoSource.start(url: url, mailbox: capture.mailbox) { [weak self] result in
            guard let self else { return }
            switch result {
            case .failure(let error):
                self.videoTestActive = false
                self.statusText = "Movie could not be loaded"
                self.metrics.setStatus("Movie error: \(error.localizedDescription)")
                LogStore.error("video", "Load failed: \(error)")
            case .success(let info):
                self.startVideoPipeline(info: info)
            }
        }
    }

    private func startVideoPipeline(info: VideoTestInfo) {
        guard let screen = NSScreen.main ?? NSScreen.screens.first else { return }
        let visible = screen.visibleFrame
        let fullScreen = settingsStore.settings.forceFullScreen
        let overlayRect: NSRect
        if fullScreen {
            overlayRect = screen.frame
        } else {
            let maximum = CGSize(width: visible.width * 0.82, height: visible.height * 0.82)
            let aspect = info.size.width / max(1, info.size.height)
            var size = maximum
            if maximum.width / maximum.height > aspect {
                size.width = (maximum.height * aspect).rounded()
            } else {
                size.height = (maximum.width / aspect).rounded()
            }
            overlayRect = NSRect(
                x: (visible.midX - size.width / 2).rounded(),
                y: (visible.midY - size.height / 2).rounded(),
                width: size.width,
                height: size.height
            )
        }
        let outputSize = CGSize(
            width: (overlayRect.width * screen.backingScaleFactor).rounded(),
            height: (overlayRect.height * screen.backingScaleFactor).rounded()
        )
        let layout = CaptureLayout(
            contentRect: ScreenGeometry.captureRect(fromAppKitRect: overlayRect),
            overlayRect: overlayRect,
            inputSize: info.size,
            outputSize: outputSize,
            title: info.title,
            bundleID: "com.caracol.video-test",
            windowID: nil,
            isDisplay: false,
            fullScreenOverlay: fullScreen
        )
        activeLayout = layout
        metrics.setSource(CaptureSourceInfo(
            title: info.title,
            bundleID: "com.caracol.video-test",
            windowID: nil,
            contentRect: layout.contentRect,
            nativeSize: info.size,
            isDisplay: false
        ))
        do {
            try ensurePipeline(layout: layout)
        } catch {
            statusText = "Renderer error"
            metrics.setStatus(error.localizedDescription)
            LogStore.error("render", "Pipeline failed: \(error)")
            return
        }
        isRunning = true
        statusText = "Test source: \(info.title)"
        metrics.setStatus("Recorded-video test mode")
        metrics.setCalibration(progress: 1, text: "Test source")
        if let seconds = autoBenchmarkSeconds {
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
                self?.runBenchmark(seconds: seconds, autoStarted: true)
            }
        }
    }

    // MARK: - Settings

    private func settingsChanged(_ settings: CaracolSettings) {
        if !isNormalizingSettings,
           settings.mode == .quality,
           settings.inferenceRoute == .gpuOnly {
            isNormalizingSettings = true
            var normalized = settings
            normalized.inferenceRoute = .coreML
            settingsStore.settings = normalized
            isNormalizingSettings = false
            return
        }

        metrics.setMode(settings)
        capture.update(settings: settings, displayRefresh: settings.displayRefresh)
        if let layout = activeLayout, let renderer {
            renderer.configure(settings: settings, layout: layout)
            overlay?.updatePacing(targetFPS: settings.targetFPS,
                                  displayRefresh: settings.displayRefresh)
            hud?.update(settings: settings, overlayRect: layout.overlayRect)
            hud?.setVisible(overlayVisible && settings.showHUD)
        }
        if settings.hotkeyEnabled {
            if !hotkey.isRegistered { _ = hotkey.register() }
        } else {
            hotkey.unregister()
        }
        if !isCalibrating && !videoTestActive {
            scheduleProfileSave()
        }
    }

    private func applyCalibrationSettings(_ settings: CaracolSettings) {
        settingsStore.settings = settings
    }

    private func scheduleProfileSave() {
        guard isRunning, let layout = activeLayout, !videoTestActive else { return }
        profileSaveWork?.cancel()
        let work = DispatchWorkItem { [weak self] in
            guard let self else { return }
            self.saveProfile(settings: self.settingsStore.settings, layout: layout)
        }
        profileSaveWork = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0, execute: work)
    }

    private func saveProfile(settings: CaracolSettings, layout: CaptureLayout) {
        guard !videoTestActive else { return }
        let now = Date()
        let existing = ProfileStore.shared.profile(for: layout.profileKey)
        let profile = GameProfile(
            key: layout.profileKey,
            title: layout.title,
            bundleID: layout.bundleID,
            windowID: layout.windowID,
            settings: settings,
            measuredSourceFPS: metrics.sourceFPS,
            calibratedAt: existing?.calibratedAt ?? now,
            lastUsedAt: now
        )
        ProfileStore.shared.upsert(profile)
        LogStore.info("profile", "Saved \(layout.title): \(settings.mode.label), \(settings.upscale.label), flow \(settings.flowQuality.label)")
    }

    // MARK: - Pipeline

    private func ensurePipeline(layout: CaptureLayout) throws {
        let device: MTLDevice
        if let metalDevice {
            device = metalDevice
        } else if let fallback = MTLCreateSystemDefaultDevice() {
            device = fallback
        } else {
            throw RendererError.texture("No Metal device")
        }

        let renderer: Renderer
        if let existing = self.renderer {
            renderer = existing
        } else {
            renderer = try Renderer(device: device, mailbox: capture.mailbox, metrics: metrics)
            self.renderer = renderer
        }

        if overlay == nil || currentOverlayRect != layout.overlayRect {
            overlay?.teardown()
            let newOverlay = OverlayController(layout: layout, renderer: renderer, device: device)
            overlay = newOverlay
            currentOverlayRect = layout.overlayRect
            hud?.teardown()
            hud = HUDController(metrics: metrics, settings: settingsStore.settings,
                                overlayRect: layout.overlayRect)
        }
        renderer.configure(settings: settingsStore.settings, layout: layout)
        overlay?.updatePacing(targetFPS: settingsStore.settings.targetFPS,
                              displayRefresh: settingsStore.settings.displayRefresh)
        overlayVisible = true
        overlay?.show()
        hud?.update(settings: settingsStore.settings, overlayRect: layout.overlayRect)
        hud?.setVisible(settingsStore.settings.showHUD && overlayVisible)
        activeLayout = layout
    }

    private func teardownPipeline() {
        overlay?.teardown()
        overlay = nil
        hud?.teardown()
        hud = nil
        renderer?.stop()
        renderer = nil
        activeLayout = nil
        currentOverlayRect = nil
        overlayVisible = true
        metrics.setCalibration(progress: 0, text: "")
    }

    private func handleLayout(_ layout: CaptureLayout) {
        if isApplyingProfile {
            activeLayout = layout
            try? ensurePipeline(layout: layout)
            return
        }
        if videoTestActive {
            videoSource.stop()
            videoTestActive = false
        }
        activeLayout = layout
        metrics.setSource(CaptureSourceInfo(
            title: layout.title,
            bundleID: layout.bundleID,
            windowID: layout.windowID,
            contentRect: layout.contentRect,
            nativeSize: layout.inputSize,
            isDisplay: layout.isDisplay
        ))
        do {
            try ensurePipeline(layout: layout)
        } catch {
            statusText = "Renderer error"
            metrics.setStatus(error.localizedDescription)
            LogStore.error("render", "Pipeline failed: \(error)")
            permissionGuide = "Caracol could not start Metal rendering: \(error.localizedDescription)"
            return
        }
        isRunning = true
        metrics.setStatus("Capturing \(layout.title)")

        let key = layout.profileKey
        if let profile = ProfileStore.shared.profile(for: key), profile.settings.calibrated {
            calibratedSources.insert(key)
            applyProfile(profile)
            ProfileStore.shared.touch(key: key)
            calibrationSummary = "Profile: \(profile.title)"
            metrics.finishCalibration()
        } else if !calibratedSources.contains(key), !isCalibrating {
            startCalibration()
        }
        statusText = "Capturing \(layout.title)"
    }

    private func applyProfile(_ profile: GameProfile) {
        isApplyingProfile = true
        var loaded = profile.settings
        loaded.displayRefresh = settingsStore.settings.displayRefresh
        loaded = loaded.sanitized(displayRefresh: loaded.displayRefresh)
        if loaded != settingsStore.settings {
            settingsStore.settings = loaded
        }
        isApplyingProfile = false
        if let layout = activeLayout, let renderer {
            renderer.configure(settings: settingsStore.settings, layout: layout)
        }
    }

    private func startCalibration() {
        guard let layout = activeLayout, !isCalibrating else { return }
        isCalibrating = true
        calibrationSummary = ""
        metrics.setCalibration(progress: 0.01, text: "Calibrating…")
        let calibrator = Autocalibrator(
            settings: settingsStore.settings,
            displayRefresh: settingsStore.settings.displayRefresh,
            metrics: metrics,
            apply: { [weak self] settings in
                self?.applyCalibrationSettings(settings)
            },
            progress: { [weak self] fraction, text in
                self?.metrics.setCalibration(progress: fraction, text: text)
            },
            completion: { [weak self] settings, summary in
                self?.finishCalibration(settings: settings, summary: summary, layout: layout)
            }
        )
        autocalibrator = calibrator
        calibrator.start()
        statusText = "Calibrating…"
    }

    private func finishCalibration(settings: CaracolSettings,
                                   summary: String,
                                   layout: CaptureLayout) {
        guard isCalibrating else { return }
        isCalibrating = false
        autocalibrator = nil
        calibratedSources.insert(layout.profileKey)
        isApplyingProfile = true
        settingsStore.settings = settings
        isApplyingProfile = false
        if let renderer, let activeLayout {
            renderer.configure(settings: settings, layout: activeLayout)
        }
        saveProfile(settings: settings, layout: layout)
        calibrationSummary = summary
        statusText = "Ready — \(summary)"
        metrics.setCalibration(progress: 1, text: "Ready · \(summary)")
        metrics.setStatus("Ready · \(summary)")
        LogStore.info("calibrate", "Ready: \(summary)")
    }

    @objc private func screensChanged() {
        let refresh = Int((NSScreen.main ?? NSScreen.screens.first)?.maximumFramesPerSecond ?? 60)
        settingsStore.refreshDisplayRate(refresh > 0 ? refresh : 60)
    }
}

// MARK: - CaptureEngineDelegate

extension AppModel: @preconcurrency CaptureEngineDelegate {
    func captureEngine(_ engine: CaptureEngine, didProduce layout: CaptureLayout) {
        handleLayout(layout)
    }

    func captureEngine(_ engine: CaptureEngine, didFail error: Error) {
        LogStore.error("capture", "Capture failed: \(error)")
        statusText = "Capture failed"
        metrics.setStatus("Capture failed: \(error.localizedDescription)")
        let nsError = error as NSError
        if nsError.domain.contains("ScreenCaptureKit") {
            permissionGuide = """
            macOS blocked screen capture. Caracol normally uses the system picker so no \
            Screen Recording permission is needed. If capture still fails, open System Settings → \
            Privacy & Security → Screen Recording and enable Caracol, then choose the window again.
            """
        } else {
            permissionGuide = "Capture failed: \(error.localizedDescription)"
        }
    }

    func captureEngineDidStop(_ engine: CaptureEngine) {
        if isStopping || videoTestActive { return }
        teardownPipeline()
        isRunning = false
        statusText = "Idle"
        metrics.setStatus("Idle")
    }

    func captureEngine(_ engine: CaptureEngine, pickerCancelled: Bool) {
        statusText = isRunning ? "Capturing" : "Idle"
        metrics.setStatus(isRunning ? "Capturing" : "Choose a window")
    }
}
