import AppKit
import CoreGraphics

enum ScreenGeometry {
    /// ScreenCaptureKit / CGWindow coordinates are global, top-left origin, Y down.
    /// AppKit screen coordinates are global, bottom-left origin, Y up, relative to
    /// the main display. This conversion keeps multi-display windows correct as
    /// long as the main display remains the global reference (the normal case).
    static func appKitRect(fromCaptureRect rect: CGRect) -> NSRect {
        let referenceHeight = NSScreen.screens.first?.frame.height ?? 0
        return NSRect(
            x: rect.origin.x,
            y: referenceHeight - rect.origin.y - rect.height,
            width: rect.width,
            height: rect.height
        )
    }

    static func captureRect(fromAppKitRect rect: NSRect) -> CGRect {
        let referenceHeight = NSScreen.screens.first?.frame.height ?? 0
        return CGRect(
            x: rect.origin.x,
            y: referenceHeight - rect.origin.y - rect.height,
            width: rect.width,
            height: rect.height
        )
    }

    static func screen(containingCaptureRect rect: CGRect) -> NSScreen? {
        let ns = appKitRect(fromCaptureRect: rect)
        return NSScreen.screens.first { $0.frame.intersects(ns) } ?? NSScreen.main
    }

    static func displayFrame(for screen: NSScreen) -> CGRect {
        captureRect(fromAppKitRect: screen.frame)
    }

    /// True when the captured content covers (almost) the whole display. In that
    /// case, a full-display overlay is less distracting than a window-shaped one.
    static func coversDisplay(_ rect: CGRect, tolerance: CGFloat = 0.94) -> Bool {
        guard let screen = screen(containingCaptureRect: rect) else { return false }
        let display = screen.frame
        let nsRect = appKitRect(fromCaptureRect: rect)
        let widthRatio = nsRect.width / max(1, display.width)
        let heightRatio = nsRect.height / max(1, display.height)
        return widthRatio >= tolerance && heightRatio >= tolerance
    }
}
