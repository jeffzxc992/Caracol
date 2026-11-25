import Foundation
import CoreVideo
import CoreGraphics
import QuartzCore

struct CapturedFrame {
    let sequence: UInt64
    let pixelBuffer: CVPixelBuffer
    let hostTime: CFTimeInterval
    let contentRect: CGRect
    let contentScale: CGFloat
    let sourceSize: CGSize
    let title: String
    let bundleID: String?
    let windowID: UInt32?
    let isDisplay: Bool

    var pixelWidth: Int { CVPixelBufferGetWidth(pixelBuffer) }
    var pixelHeight: Int { CVPixelBufferGetHeight(pixelBuffer) }
}

/// Latest-frame-wins mailbox. Capture never waits for the renderer and the
/// renderer never drains a queue: it simply asks for the newest frame. This is
/// the core low-latency policy of Caracol.
final class FrameMailbox {
    private let lock = NSLock()
    private var latestFrame: CapturedFrame?
    private var sequence: UInt64 = 0
    private var frameCount: Int = 0

    func put(_ frame: CapturedFrame, hostTime: CFTimeInterval = hostNow()) {
        lock.lock()
        sequence &+= 1
        frameCount += 1
        let stamped = CapturedFrame(
            sequence: sequence,
            pixelBuffer: frame.pixelBuffer,
            hostTime: hostTime,
            contentRect: frame.contentRect,
            contentScale: frame.contentScale,
            sourceSize: frame.sourceSize,
            title: frame.title,
            bundleID: frame.bundleID,
            windowID: frame.windowID,
            isDisplay: frame.isDisplay
        )
        latestFrame = stamped
        lock.unlock()
    }

    func latest() -> CapturedFrame? {
        lock.lock()
        defer { lock.unlock() }
        return latestFrame
    }

    func latest(after lastSequence: UInt64) -> CapturedFrame? {
        lock.lock()
        defer { lock.unlock() }
        guard let frame = latestFrame, frame.sequence > lastSequence else { return nil }
        return frame
    }

    func clear() {
        lock.lock()
        latestFrame = nil
        lock.unlock()
    }

    var totalFrames: Int {
        lock.lock()
        defer { lock.unlock() }
        return frameCount
    }
}

struct CaptureSourceInfo {
    var title: String = "No window selected"
    var bundleID: String?
    var windowID: UInt32?
    var contentRect: CGRect = .zero
    var nativeSize: CGSize = .zero
    var isDisplay: Bool = false
    var key: String {
        GameProfile.key(bundleID: bundleID, title: title, windowID: windowID)
    }
}
