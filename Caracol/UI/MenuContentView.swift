import SwiftUI
import AppKit

struct MenuContentView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var store: SettingsStore
    @ObservedObject var metrics: MetricsStore
    @State private var advancedExpanded = false

    init(model: AppModel) {
        self.model = model
        self.store = model.settingsStore
        self.metrics = model.metrics
    }

    private var fpsChoices: [Int] {
        let maximum = max(30, min(60, store.settings.displayRefresh))
        return CaracolSettings.targetFPSChoices.filter { $0 <= maximum }
    }

    private var permissionBinding: Binding<Bool> {
        Binding(
            get: { model.permissionGuide != nil },
            set: { if !$0 { model.dismissPermissionGuide() } }
        )
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                header
                Divider()
                sourceControls
                Divider()
                renderControls
                Divider()
                liveMetrics
                if model.isCalibrating || !model.calibrationSummary.isEmpty {
                    Divider()
                    calibrationSection
                }
                Divider()
                DisclosureGroup("Benchmark & diagnostics", isExpanded: $advancedExpanded) {
                    diagnostics
                        .padding(.top, 8)
                }
                .font(.system(size: 12, weight: .semibold))
                footer
            }
            .padding(16)
        }
        .frame(width: 390, height: 680)
        .alert("Capture permission", isPresented: permissionBinding) {
            Button("Open Screen Recording") {
                model.openScreenRecordingSettings()
                model.dismissPermissionGuide()
            }
            Button("Dismiss", role: .cancel) {
                model.dismissPermissionGuide()
            }
        } message: {
            Text(model.permissionGuide ?? "")
        }
    }

    private var header: some View {
        HStack(spacing: 10) {
            Image(systemName: "hourglass")
                .font(.system(size: 20, weight: .semibold))
                .foregroundStyle(.tint)
            VStack(alignment: .leading, spacing: 1) {
                Text("Caracol")
                    .font(.system(size: 16, weight: .bold, design: .rounded))
                Text("Minimal-latency frame generation")
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Circle()
                .fill(model.isRunning ? Color.green : Color.secondary.opacity(0.4))
                .frame(width: 8, height: 8)
            Text(model.statusText)
                .font(.system(size: 10))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .frame(maxWidth: 120, alignment: .trailing)
        }
    }

    private var sourceControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(metrics.sourceTitle.isEmpty ? "No window selected" : metrics.sourceTitle)
                    .font(.system(size: 12, weight: .medium))
                    .lineLimit(1)
                Spacer()
            }
            HStack(spacing: 8) {
                Button {
                    model.chooseWindow()
                } label: {
                    Label("Choose window", systemImage: "macwindow")
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)

                Button("Stop") {
                    model.stop()
                }
                .controlSize(.small)
                .disabled(!model.isRunning)

                Spacer()

                Toggle("Overlay", isOn: Binding(
                    get: { model.overlayVisible },
                    set: { model.setOverlayVisible($0) }
                ))
                .toggleStyle(.switch)
                .controlSize(.mini)
                .disabled(!model.isRunning)
            }
            HStack(spacing: 8) {
                Button {
                    model.testWithRecordedVideo()
                } label: {
                    Label("Recorded-video test…", systemImage: "film")
                }
                .controlSize(.small)
                Text("Feeds a movie through the same render path")
                    .font(.system(size: 9))
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var renderControls: some View {
        VStack(alignment: .leading, spacing: 9) {
            row("Target fps") {
                Picker("", selection: $store.settings.targetFPS) {
                    ForEach(fpsChoices, id: \.self) { fps in
                        Text("\(fps)").tag(fps)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .frame(width: 190)
            }

            row("Mode") {
                Picker("", selection: $store.settings.mode) {
                    ForEach(RenderMode.allCases) { mode in
                        Text(mode.label).tag(mode)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .frame(width: 190)
            }

            row("Upscale") {
                Picker("", selection: $store.settings.upscale) {
                    ForEach(UpscaleMode.allCases) { mode in
                        Text(mode.label).tag(mode)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .frame(width: 190)
            }

            HStack(spacing: 8) {
                Text("Sharpness")
                    .frame(width: 76, alignment: .leading)
                Slider(value: $store.settings.sharpness, in: 0...1)
                Text("\(Int(store.settings.sharpness * 100))%")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(width: 34, alignment: .trailing)
            }

            Toggle("Frame generation", isOn: $store.settings.frameGeneration)
                .font(.system(size: 11))
            Toggle("Interpolation instead of extrapolation (adds one source frame of latency)",
                   isOn: $store.settings.interpolation)
                .font(.system(size: 11))
            Toggle("Force full-display overlay", isOn: $store.settings.forceFullScreen)
                .font(.system(size: 11))
            Toggle("Show HUD", isOn: $store.settings.showHUD)
                .font(.system(size: 11))

            HStack(spacing: 8) {
                Text("Flow")
                    .frame(width: 76, alignment: .leading)
                Picker("", selection: $store.settings.flowQuality) {
                    ForEach(FlowQuality.allCases) { quality in
                        Text(quality.label).tag(quality)
                    }
                }
                .labelsHidden()
                .pickerStyle(.segmented)
                .frame(width: 110)

                Text("Inference")
                    .font(.system(size: 11))
                Picker("", selection: $store.settings.inferenceRoute) {
                    ForEach(InferenceRoute.allCases) { route in
                        Text(route.shortLabel).tag(route)
                    }
                }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(width: 70)
            }
            .font(.system(size: 11))

            HStack(spacing: 8) {
                Button("Recalibrate (~10 s)") {
                    model.recalibrate()
                }
                .controlSize(.small)
                .disabled(!model.isRunning || model.isCalibrating || model.videoTestActive)

                Spacer()

                Text(model.metalMLStatus)
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.secondary)
                Button("Test MTL4 ML") {
                    model.runMetalMLSelfTest()
                }
                .controlSize(.small)
            }
        }
    }

    private var liveMetrics: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("LIVE")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundStyle(.secondary)
                Spacer()
                Text("\(store.settings.mode.label) · \(store.settings.upscale.label) · \(store.settings.inferenceRoute.shortLabel)")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 3) {
                GridRow {
                    metric("Source", String(format: "%.0f", metrics.sourceFPS))
                    metric("Real", String(format: "%.0f", metrics.realFPS))
                    metric("Gen", String(format: "%.0f", metrics.generatedFPS))
                    metric("Shown", String(format: "%.0f", metrics.presentedFPS))
                }
                GridRow {
                    metric("Lat p50", String(format: "%.1f ms", metrics.latencyP50))
                    metric("Lat p95", String(format: "%.1f ms", metrics.latencyP95))
                    metric("GPU", String(format: "%.1f ms", metrics.gpuMs))
                    metric("Flow", String(format: "%.1f ms", metrics.flowMs))
                }
                GridRow {
                    metric("ANE", String(format: "%.1f ms", metrics.aneMs))
                    metric("CPU", String(format: "%.0f%%", metrics.cpuPercent))
                    metric("GPU", String(format: "%.0f%%", metrics.gpuPercent))
                    metric("RAM", String(format: "%.0f MB", metrics.memoryMB))
                }
            }
            Text("Thermal \(metrics.thermal) · hotkey ⌃⌥⌘G toggles overlay")
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
        }
    }

    private func metric(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(label.uppercased())
                .font(.system(size: 8, weight: .semibold))
                .foregroundStyle(.secondary)
            Text(value)
                .font(.system(size: 11, weight: .medium, design: .monospaced))
        }
        .frame(minWidth: 64, alignment: .leading)
    }

    private var calibrationSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("CALIBRATION")
                    .font(.system(size: 9, weight: .bold))
                    .foregroundStyle(.secondary)
                Spacer()
                if model.isCalibrating {
                    ProgressView(value: metrics.calibrationProgress)
                        .progressViewStyle(.linear)
                        .frame(width: 110)
                }
            }
            Text(model.calibrationSummary.isEmpty ? metrics.calibrationText : model.calibrationSummary)
                .font(.system(size: 10, design: .monospaced))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var diagnostics: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Button("Run 20 s benchmark") {
                    model.runBenchmark(seconds: 20)
                }
                .controlSize(.small)
                .disabled(!model.isRunning || model.videoTestActive || model.benchmarkRunning)

                if model.benchmarkRunning {
                    Button("Cancel") { model.cancelBenchmark() }
                        .controlSize(.small)
                }
                Spacer()
                Text("MTL4 ML: \(model.metalMLStatus)")
                    .font(.system(size: 9))
                    .foregroundStyle(.secondary)
            }
            if model.benchmarkRunning {
                ProgressView(value: model.benchmarkProgress)
                    .progressViewStyle(.linear)
            }
            if !model.benchmarkText.isEmpty {
                Text(model.benchmarkText)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack(spacing: 8) {
                Button("Export log") { model.exportLog() }
                    .controlSize(.small)
                Button("Open profiles") { model.openProfilesFolder() }
                    .controlSize(.small)
                Spacer()
            }
            Text("Inference: GPU flow is default and model-free. ANE uses IFRNetLite.mlmodelc; the bundled model is a plumbing stub. MTL4 ML is experimental and gated by a child-process self-test. Interpolation is opt-in; extrapolation is the low-latency default.")
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var footer: some View {
        HStack {
            Text("v1.0 · on-device, no uploads")
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
            Spacer()
            Button("Quit") { NSApp.terminate(nil) }
                .controlSize(.small)
        }
        .padding(.top, 2)
    }

    private func row<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        HStack(spacing: 8) {
            Text(title)
                .font(.system(size: 11))
                .frame(width: 76, alignment: .leading)
            content()
            Spacer(minLength: 0)
        }
    }
}
