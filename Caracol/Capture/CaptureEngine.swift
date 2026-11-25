import Foundation
import ScreenCaptureKit
import CoreMedia
import CoreVideo
import AppKit

struct CaptureLayout {
    var contentRect: CGRect          // points, global top-left
    var overlayRect: NSRect          // points, AppKit coordinates
    var inputSize: CGSize            // pixels sent to the renderer
    var outputSize: CGSize           // pixels of the overlay drawable
    var title: String
    var bundleID: String?
    var windowID: UInt32?
    var isDisplay: Bool
    var fullScreenOverlay: Bool

    var aspect: CGFloat {
        guard outputSize.height > 0 else { return 1 }
        return outputSize.width / outputSize.height
    }
}

protocol CaptureEngineDelegate: AnyObject {
    func captureEngine(_ engine: CaptureEngine, didProduce layout: CaptureLayout)
    func captureEngine(_ engine: CaptureEngine, didFail error: Error)
    func captureEngineDidStop(_ engine: CaptureEngine)
    func captureEngine(_ engine: CaptureEngine, pickerCancelled: Bool)
}

/// ScreenCaptureKit capture with the system content-sharing picker. The picker
/// path is deliberately used because it grants one-shot access without the
/// Screen Recording TCC prompt. Frames are zero-copy IOSurfaces; this class
/// only wraps them in CapturedFrame values and publishes the latest one.
final class CaptureEngine: NSObject {
    weak var delegate: CaptureEngineDelegate?

    let mailbox = FrameMailbox()
    private(set) var layout: CaptureLayout?
    private(set) var isCapturing = false
    private(set) var isPaused = false

    private let picker = SCContentSharingPicker.shared
    private let captureQueue = DispatchQueue(label: "com.caracol.capture", qos: .userInteractive)
    private var stream: SCStream?
    private var filter: SCContentFilter?
    private var settings: CaracolSettings
    private var displayRefresh: Int

    // Cached source metadata for the current filter.
    private var sourceTitle = "Game"
    private var sourceBundleID: String?
    private var sourceWindowID: UInt32?
    private var sourceIsDisplay = false

    init(settings: CaracolSettings, displayRefresh: Int) {
        self.settings = settings
        self.displayRefresh = displayRefresh
        super.init()
        picker.addObserver(self)
        let config = SCContentSharingPickerConfiguration()
        config.allowedPickerModes = [.singleWindow, .singleDisplay]
        config.allowsChangingSelectedContent = true
        picker.defaultConfiguration = config
    }

    deinit {
        picker.removeObserver(self)
    }

    func update(settings: CaracolSettings, displayRefresh: Int) {
        let restartNeeded = settings.upscale != self.settings.upscale ||
            settings.forceFullScreen != self.settings.forceFullScreen
        self.settings = settings
        self.displayRefresh = displayRefresh
        if restartNeeded, let filter, isCapturing, !isPaused {
            start(filter: filter, resetLayout: true)
        }
    }

    func presentPicker() {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.picker.isActive = true
            self.picker.present()
        }
    }

    func restart() {
        guard let filter else { return }
        start(filter: filter, resetLayout: true)
    }

    func stop() {
        isCapturing = false
        isPaused = false
        stream?.stopCapture { _ in }
        stream = nil
        filter = nil
        layout = nil
        mailbox.clear()
        delegate?.captureEngineDidStop(self)
        LogStore.info("capture", "Capture stopped")
    }

    func pause() {
        guard isCapturing, !isPaused else { return }
        isPaused = true
        stream?.stopCapture { error in
            if let error { LogStore.warn("capture", "Pause stop error: \(error)") }
        }
        LogStore.info("capture", "Capture paused")
    }

    func resume() {
        guard isPaused else { return }
        isPaused = false
        if let filter {
            start(filter: filter, resetLayout: false)
        }
    }

    // MARK: - Stream lifecycle

    private func start(filter: SCContentFilter, resetLayout: Bool) {
        stream?.stopCapture { _ in }
        stream = nil
        self.filter = filter

        let contentRect = filter.contentRect
        let pointScale = CGFloat(max(filter.pointPixelScale, 1))
        sourceIsDisplay = filter.style == .display
        if let window = filter.includedWindows.first {
            sourceTitle = window.title?.isEmpty == false ? window.title! : "Game"
            sourceBundleID = window.owningApplication?.bundleIdentifier
            sourceWindowID = window.windowID
        } else if let display = filter.includedDisplays.first {
            sourceTitle = "Display \(display.displayID)"
            sourceBundleID = nil
            sourceWindowID = nil
        } else {
            sourceTitle = sourceIsDisplay ? "Display" : "Game"
            sourceBundleID = nil
            sourceWindowID = nil
        }

        // The overlay always covers the game window. If the window is effectively
        // fullscreen, use the whole display so the menu bar / dock area is invisible.
        let screen = ScreenGeometry.screen(containingCaptureRect: contentRect) ?? NSScreen.main
        let fullScreenOverlay = settings.forceFullScreen || ScreenGeometry.coversDisplay(contentRect)
        let overlayRect: NSRect
        let screenScale: CGFloat
        if fullScreenOverlay, let screen {
            overlayRect = screen.frame
            screenScale = screen.backingScaleFactor
        } else {
            overlayRect = ScreenGeometry.appKitRect(fromCaptureRect: contentRect)
            screenScale = screen?.backingScaleFactor ?? 2
        }

        let outputSize = CGSize(
            width: max(64, (overlayRect.width * screenScale).rounded()),
            height: max(64, (overlayRect.height * screenScale).rounded())
        )

        let divisor = max(1.0, settings.upscale.factor)
        var inputSize = CGSize(
            width: (outputSize.width / divisor).rounded(),
            height: (outputSize.height / divisor).rounded()
        )
        // Keep both dimensions even; MetalFX and most video paths prefer it.
        inputSize.width = max(2, (inputSize.width / 2).rounded() * 2)
        inputSize.height = max(2, (inputSize.height / 2).rounded() * 2)

        let newLayout = CaptureLayout(
            contentRect: contentRect,
            overlayRect: overlayRect,
            inputSize: inputSize,
            outputSize: outputSize,
            title: sourceTitle,
            bundleID: sourceBundleID,
            windowID: sourceWindowID,
            isDisplay: sourceIsDisplay,
            fullScreenOverlay: fullScreenOverlay
        )
        layout = newLayout
        if resetLayout || layout == nil {
            delegate?.captureEngine(self, didProduce: newLayout)
        }

        let config = SCStreamConfiguration()
        config.width = Int(inputSize.width)
        config.height = Int(inputSize.height)
        config.minimumFrameInterval = CMTime(value: 1, timescale: CMTimeScale(max(30, min(120, displayRefresh))))
        config.pixelFormat = kCVPixelFormatType_32BGRA
        config.queueDepth = 1
        config.showsCursor = false
        config.showMouseClicks = false
        config.capturesAudio = false
        config.scalesToFit = true
        config.preservesAspectRatio = true
        config.captureResolution = .best
        config.shouldBeOpaque = true
        config.ignoreShadowsSingleWindow = true
        config.ignoreShadowsDisplay = true
        config.backgroundColor = CGColor(gray: 0, alpha: 1)

        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        do {
            try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: captureQueue)
        } catch {
            LogStore.error("capture", "addStreamOutput failed: \(error)")
            delegate?.captureEngine(self, didFail: error)
            return
        }
        self.stream = stream
        stream.startCapture { [weak self] error in
            guard let self else { return }
            if let error {
                LogStore.error("capture", "startCapture failed: \(error)")
                DispatchQueue.main.async { self.delegate?.captureEngine(self, didFail: error) }
            } else {
                self.isCapturing = true
                LogStore.info("capture", "Capture started: \(self.sourceTitle) \(Int(inputSize.width))x\(Int(inputSize.height))")
                DispatchQueue.main.async {
                    self.delegate?.captureEngine(self, didProduce: newLayout)
                }
            }
        }
    }

    private func makeFrame(from sampleBuffer: CMSampleBuffer) -> CapturedFrame? {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }

        // Drop idle / blank frames. ScreenCaptureKit emits idle frames when the
        // content has not changed; presenting them wastes energy and inflates fps.
        if let attachments = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
           let raw = attachments.first?[.status] as? Int,
           let status = SCFrameStatus(rawValue: raw),
           status != .complete {
            return nil
        }

        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let frameLayout = layout
        let rect = frameLayout?.contentRect ?? .zero
        return CapturedFrame(
            sequence: 0,
            pixelBuffer: pixelBuffer,
            hostTime: hostNow(),
            contentRect: rect,
            contentScale: frameLayout != nil ? CGFloat(max(1, (frameLayout!.outputSize.width / max(1, rect.width)))) : 2,
            sourceSize: CGSize(width: width, height: height),
            title: sourceTitle,
            bundleID: sourceBundleID,
            windowID: sourceWindowID,
            isDisplay: sourceIsDisplay
        )
    }
}

extension CaptureEngine: SCContentSharingPickerObserver {
    func contentSharingPicker(_ picker: SCContentSharingPicker, didCancelFor stream: SCStream?) {
        LogStore.info("capture", "Picker cancelled")
        delegate?.captureEngine(self, pickerCancelled: true)
    }

    func contentSharingPicker(_ picker: SCContentSharingPicker, didUpdateWith filter: SCContentFilter, for stream: SCStream?) {
        LogStore.info("capture", "Picker selected: \(filter.style.rawValue) \(filter.contentRect)")
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.start(filter: filter, resetLayout: true)
        }
    }

    func contentSharingPickerStartDidFailWithError(_ error: Error) {
        LogStore.error("capture", "Picker failed: \(error)")
        delegate?.captureEngine(self, didFail: error)
    }
}

extension CaptureEngine: SCStreamOutput {
    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, !isPaused, isCapturing else { return }
        guard let frame = makeFrame(from: sampleBuffer) else { return }
        mailbox.put(frame)
    }
}

extension CaptureEngine: SCStreamDelegate {
    func stream(_ stream: SCStream, didStopWithError error: Error) {
        LogStore.error("capture", "Stream stopped with error: \(error)")
        isCapturing = false
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.delegate?.captureEngine(self, didFail: error)
        }
    }
}
