import Foundation
import ScreenCaptureKit
import CoreMedia
import CoreVideo
import AppKit

struct CaptureLayout: Equatable {
    var contentRect: CGRect          // points, global top-left
    var overlayRect: NSRect          // points, AppKit coordinates
    var inputSize: CGSize            // pixels captured (before upscaling)
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

    var profileKey: String {
        GameProfile.key(bundleID: bundleID, title: title, windowID: windowID)
    }
}

protocol CaptureEngineDelegate: AnyObject {
    func captureEngine(_ engine: CaptureEngine, didProduce layout: CaptureLayout)
    func captureEngine(_ engine: CaptureEngine, didFail error: Error)
    func captureEngineDidStop(_ engine: CaptureEngine)
    func captureEngine(_ engine: CaptureEngine, pickerCancelled: Bool)
}

/// ScreenCaptureKit capture with the system content-sharing picker. The picker
/// path is used because it grants one-shot access without the Screen Recording
/// TCC prompt. Frames are zero-copy IOSurfaces; this class only wraps them in
/// CapturedFrame values and publishes the latest one.
final class CaptureEngine: NSObject {
    weak var delegate: CaptureEngineDelegate?

    let mailbox = FrameMailbox()
    private(set) var layout: CaptureLayout?

    private let picker = SCContentSharingPicker.shared
    private let captureQueue = DispatchQueue(label: "com.caracol.capture", qos: .userInteractive)
    private let stateLock = NSLock()
    private var _isCapturing = false
    private var _isPaused = false

    private var stream: SCStream?
    private var filter: SCContentFilter?
    private var settings: CaracolSettings
    private var displayRefresh: Int

    // Cached source metadata for the current filter.
    private var sourceTitle = "Game"
    private var sourceBundleID: String?
    private var sourceWindowID: UInt32?
    private var sourceIsDisplay = false
    private let pickerConfiguration: SCContentSharingPickerConfiguration = {
        var config = SCContentSharingPickerConfiguration()
        config.allowedPickerModes = [.singleWindow, .singleDisplay]
        config.allowsChangingSelectedContent = true
        return config
    }()
    private let opaqueBlack = CGColor(gray: 0, alpha: 1)

    var isCapturing: Bool {
        stateLock.lock(); defer { stateLock.unlock() }
        return _isCapturing
    }

    var isPaused: Bool {
        stateLock.lock(); defer { stateLock.unlock() }
        return _isPaused
    }

    init(settings: CaracolSettings, displayRefresh: Int) {
        self.settings = settings
        self.displayRefresh = displayRefresh
        super.init()
        picker.add(self)
        picker.defaultConfiguration = pickerConfiguration
    }

    deinit {
        picker.remove(self)
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
            NSApp.activate(ignoringOtherApps: true)
            self.picker.isActive = true
            self.picker.present()
        }
    }

    func restart() {
        guard let filter else { return }
        start(filter: filter, resetLayout: true)
    }

    func stop() {
        setState(capturing: false, paused: false)
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
        setState(capturing: true, paused: true)
        stream?.stopCapture { error in
            if let error { LogStore.warn("capture", "Pause stop error: \(error)") }
        }
        LogStore.info("capture", "Capture paused")
    }

    func resume() {
        guard isPaused else { return }
        setState(capturing: true, paused: false)
        if let filter {
            start(filter: filter, resetLayout: false)
        }
    }

    // MARK: - Stream lifecycle

    private func setState(capturing: Bool, paused: Bool) {
        stateLock.lock()
        _isCapturing = capturing
        _isPaused = paused
        stateLock.unlock()
    }

    private func start(filter: SCContentFilter, resetLayout: Bool) {
        let oldStream = stream
        stream = nil
        oldStream?.stopCapture { _ in }
        self.filter = filter

        let previousLayout = layout
        let contentRect = filter.contentRect
        sourceIsDisplay = filter.style == .display
        if let window = filter.includedWindows.first {
            sourceTitle = window.title.flatMap { $0.isEmpty ? nil : $0 } ?? "Game"
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

        // The overlay always covers the game content. If the content is
        // effectively fullscreen, use the whole display so menu bar / dock
        // remnants are not visible around it.
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

        let sourceAspect = contentRect.height > 0 ? contentRect.width / contentRect.height : 16.0 / 9.0
        let target = CGSize(
            width: max(2, outputSize.width / max(1, settings.upscale.factor)),
            height: max(2, outputSize.height / max(1, settings.upscale.factor))
        )
        var inputSize = Self.fit(aspect: sourceAspect, into: target)
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
        if resetLayout || previousLayout != newLayout {
            delegate?.captureEngine(self, didProduce: newLayout)
        }

        let configuration = SCStreamConfiguration()
        configuration.width = Int(inputSize.width)
        configuration.height = Int(inputSize.height)
        configuration.minimumFrameInterval = CMTime(
            value: 1,
            timescale: CMTimeScale(max(30, min(120, displayRefresh)))
        )
        configuration.pixelFormat = kCVPixelFormatType_32BGRA
        configuration.queueDepth = 1
        configuration.showsCursor = false
        configuration.showMouseClicks = false
        configuration.capturesAudio = false
        configuration.scalesToFit = true
        configuration.preservesAspectRatio = true
        configuration.captureResolution = .best
        configuration.shouldBeOpaque = true
        configuration.ignoreShadowsSingleWindow = true
        configuration.ignoreShadowsDisplay = true
        configuration.backgroundColor = opaqueBlack

        let stream = SCStream(filter: filter, configuration: configuration, delegate: self)
        do {
            try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: captureQueue)
        } catch {
            LogStore.error("capture", "addStreamOutput failed: \(error)")
            delegate?.captureEngine(self, didFail: error)
            return
        }
        self.stream = stream
        stream.startCapture { [weak self, weak stream] error in
            guard let self, let stream, self.stream === stream else { return }
            if let error {
                LogStore.error("capture", "startCapture failed: \(error)")
                DispatchQueue.main.async { self.delegate?.captureEngine(self, didFail: error) }
            } else {
                self.setState(capturing: true, paused: false)
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
           let raw = attachments.first?[.status] as? NSNumber,
           let status = SCFrameStatus(rawValue: raw.intValue),
           status != .complete {
            return nil
        }

        let frameLayout = layout
        let rect = frameLayout?.contentRect ?? .zero
        return CapturedFrame(
            sequence: 0,
            pixelBuffer: pixelBuffer,
            hostTime: hostNow(),
            contentRect: rect,
            contentScale: CGFloat(max(1, (filter?.pointPixelScale ?? 2))),
            sourceSize: CGSize(width: CVPixelBufferGetWidth(pixelBuffer),
                               height: CVPixelBufferGetHeight(pixelBuffer)),
            title: sourceTitle,
            bundleID: sourceBundleID,
            windowID: sourceWindowID,
            isDisplay: sourceIsDisplay
        )
    }

    private static func fit(aspect: CGFloat, into size: CGSize) -> CGSize {
        guard aspect > 0, size.width > 0, size.height > 0 else { return size }
        let containerAspect = size.width / size.height
        if containerAspect > aspect {
            return CGSize(width: (size.height * aspect).rounded(), height: size.height)
        }
        return CGSize(width: size.width, height: (size.width / aspect).rounded())
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
        setState(capturing: false, paused: false)
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.delegate?.captureEngine(self, didFail: error)
        }
    }
}
