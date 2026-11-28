import AppKit
import Metal
import QuartzCore
import SwiftUI

/// A layer-backed, click-through window that owns the CAMetalLayer. It is
/// excluded from screen sharing where the OS honors `sharingType`.
private final class OverlayMetalView: NSView {
    let metalLayer = CAMetalLayer()

    override func makeBackingLayer() -> CALayer {
        metalLayer
    }

    override var isOpaque: Bool { true }

    override func hitTest(_ point: NSPoint) -> NSView? { nil }
}

private final class OverlayWindow: NSWindow {
    override var canBecomeKey: Bool { false }
    override var canBecomeMain: Bool { false }

    override func constrainFrameRect(_ frameRect: NSRect, to screen: NSScreen?) -> NSRect {
        frameRect
    }
}

final class OverlayController: NSObject, CAMetalDisplayLinkDelegate {
    private let window: OverlayWindow
    private let metalView: OverlayMetalView
    private let renderer: Renderer
    private let device: MTLDevice
    private var displayLink: CAMetalDisplayLink?
    private var isScheduled = false
    private(set) var isVisible = false

    init(layout: CaptureLayout, renderer: Renderer, device: MTLDevice) {
        self.renderer = renderer
        self.device = device
        self.metalView = OverlayMetalView(frame: NSRect(origin: .zero, size: layout.overlayRect.size))
        self.window = OverlayWindow(
            contentRect: layout.overlayRect,
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        super.init()

        window.isOpaque = true
        window.backgroundColor = .black
        window.hasShadow = false
        window.level = .screenSaver
        window.collectionBehavior = [.canJoinAllSpaces, .stationary, .ignoresCycle, .fullScreenAuxiliary]
        window.ignoresMouseEvents = true
        window.isMovable = false
        window.isReleasedWhenClosed = false
        window.animationBehavior = .none
        window.sharingType = DebugTrace.allowsCapture ? .readOnly : .none
        window.contentView = metalView
        window.setFrame(layout.overlayRect, display: false)
        metalView.frame = NSRect(origin: .zero, size: layout.overlayRect.size)
        metalView.wantsLayer = true

        let scale = window.backingScaleFactor
        metalView.metalLayer.device = device
        metalView.metalLayer.pixelFormat = .bgra8Unorm
        metalView.metalLayer.framebufferOnly = true
        metalView.metalLayer.maximumDrawableCount = 2
        metalView.metalLayer.displaySyncEnabled = false
        metalView.metalLayer.isOpaque = true
        metalView.metalLayer.contentsScale = scale
        metalView.metalLayer.drawableSize = CGSize(
            width: layout.overlayRect.width * scale,
            height: layout.overlayRect.height * scale
        )
        metalView.metalLayer.colorspace = CGColorSpace(name: CGColorSpace.sRGB)

        let link = CAMetalDisplayLink(metalLayer: metalView.metalLayer)
        link.delegate = self
        link.preferredFrameLatency = 1
        link.preferredFrameRateRange = CAFrameRateRange(
            minimum: 30,
            maximum: 120,
            preferred: 60
        )
        displayLink = link
    }

    func updatePacing(targetFPS: Int, displayRefresh: Int) {
        guard let link = displayLink else { return }
        // Caracol targets at most 60 unique frames per second. Running the
        // display link faster only produces duplicate presents and wastes GPU.
        let maximum = Float(clamp(min(displayRefresh, 60), 30, 60))
        link.preferredFrameRateRange = CAFrameRateRange(
            minimum: 30,
            maximum: maximum,
            preferred: maximum
        )
    }

    func show() {
        guard !isVisible else { return }
        isVisible = true
        window.orderFrontRegardless()
        if let link = displayLink {
            if !isScheduled {
                link.add(to: RunLoop.main, forMode: .common)
                isScheduled = true
            }
            link.isPaused = false
        }
    }

    func hide() {
        guard isVisible else { return }
        isVisible = false
        displayLink?.isPaused = true
        window.orderOut(nil)
    }

    func setVisible(_ visible: Bool) {
        visible ? show() : hide()
    }

    func teardown() {
        displayLink?.invalidate()
        displayLink = nil
        window.orderOut(nil)
        window.contentView = nil
        isVisible = false
    }

    func metalDisplayLink(_ link: CAMetalDisplayLink, needsUpdate update: CAMetalDisplayLink.Update) {
        autoreleasepool {
            renderer.draw(update: update)
        }
    }
}

// MARK: - HUD

struct HUDView: View {
    @ObservedObject var metrics: MetricsStore
    let settings: CaracolSettings

    private var calibrationActive: Bool {
        !metrics.calibrationText.isEmpty && metrics.calibrationProgress < 1
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 10) {
                Text("REAL \(metrics.realFPS, specifier: "%.0f")")
                Text("GEN \(metrics.generatedFPS, specifier: "%.0f")")
                Text("SHOW \(metrics.presentedFPS, specifier: "%.0f")")
                Spacer(minLength: 0)
                Text("\(settings.mode.label) · \(settings.upscale.label) · \(settings.targetFPS)")
                    .foregroundStyle(.white.opacity(0.72))
            }
            HStack(spacing: 10) {
                Text("LAT \(metrics.latencyP50, specifier: "%.1f")/\(metrics.latencyP95, specifier: "%.1f") ms")
                Text("GPU \(metrics.gpuMs, specifier: "%.1f")")
                Text("FLOW \(metrics.flowMs, specifier: "%.1f")")
                Text("ANE \(metrics.aneMs, specifier: "%.1f")")
            }
            HStack(spacing: 10) {
                Text("CPU \(metrics.cpuPercent, specifier: "%.0f")%")
                Text("GPU \(metrics.gpuPercent, specifier: "%.0f")%")
                Text("RAM \(metrics.memoryMB, specifier: "%.0f") MB")
                Text(metrics.thermal)
            }
            if calibrationActive {
                HStack(spacing: 8) {
                    ProgressView(value: metrics.calibrationProgress)
                        .progressViewStyle(.linear)
                        .tint(.green)
                        .frame(width: 110)
                    Text(metrics.calibrationText)
                        .foregroundStyle(.green)
                }
            }
            if !metrics.sourceTitle.isEmpty {
                Text(metrics.sourceTitle)
                    .foregroundStyle(.white.opacity(0.55))
                    .lineLimit(1)
            }
        }
        .font(.system(size: 11, weight: .medium, design: .monospaced))
        .foregroundStyle(.white.opacity(0.92))
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .frame(width: 370, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .fill(Color.black.opacity(0.68))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .stroke(Color.white.opacity(0.12), lineWidth: 1)
        )
    }
}

final class HUDController {
    private let window: NSWindow
    private let hostingView: NSHostingView<HUDView>
    private var isVisible = true

    init(metrics: MetricsStore, settings: CaracolSettings, overlayRect: NSRect) {
        hostingView = NSHostingView(rootView: HUDView(metrics: metrics, settings: settings))
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 370, height: 110),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = false
        window.level = .screenSaver + 1
        window.collectionBehavior = [.canJoinAllSpaces, .stationary, .ignoresCycle, .fullScreenAuxiliary]
        window.ignoresMouseEvents = true
        window.isMovable = false
        window.isReleasedWhenClosed = false
        window.animationBehavior = .none
        window.sharingType = DebugTrace.allowsCapture ? .readOnly : .none
        hostingView.wantsLayer = true
        hostingView.layer?.backgroundColor = NSColor.clear.cgColor
        window.contentView = hostingView
        window.setContentSize(hostingView.fittingSize)
        window.setFrameTopLeftPoint(NSPoint(x: overlayRect.minX + 12, y: overlayRect.maxY - 12))
        update(settings: settings, overlayRect: overlayRect)
    }

    func update(settings: CaracolSettings, overlayRect: NSRect) {
        hostingView.rootView = HUDView(
            metrics: hostingView.rootView.metrics,
            settings: settings
        )
        let size = hostingView.fittingSize
        window.setContentSize(size)
        window.setFrameTopLeftPoint(NSPoint(x: overlayRect.minX + 12, y: overlayRect.maxY - 12))
    }

    func setVisible(_ visible: Bool) {
        isVisible = visible
        if visible {
            window.orderFrontRegardless()
        } else {
            window.orderOut(nil)
        }
    }

    func show() {
        window.orderFrontRegardless()
    }

    func teardown() {
        window.orderOut(nil)
        window.contentView = nil
    }
}
