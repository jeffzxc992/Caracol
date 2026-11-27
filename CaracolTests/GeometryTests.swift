import XCTest
import AppKit
@testable import Caracol

final class GeometryTests: XCTestCase {
    func testCaptureRoundTrip() {
        let rect = CGRect(x: 120, y: 80, width: 1280, height: 720)
        let appKit = ScreenGeometry.appKitRect(fromCaptureRect: rect)
        let roundTrip = ScreenGeometry.captureRect(fromAppKitRect: appKit)
        XCTAssertEqual(roundTrip.minX, rect.minX, accuracy: 0.001)
        XCTAssertEqual(roundTrip.minY, rect.minY, accuracy: 0.001)
        XCTAssertEqual(roundTrip.width, rect.width, accuracy: 0.001)
        XCTAssertEqual(roundTrip.height, rect.height, accuracy: 0.001)
    }

    func testCoversDisplay() {
        guard let screen = NSScreen.main else { return }
        let display = ScreenGeometry.captureRect(fromAppKitRect: screen.frame)
        XCTAssertTrue(ScreenGeometry.coversDisplay(display))
        let small = CGRect(x: display.minX + 100, y: display.minY + 100,
                           width: display.width * 0.5, height: display.height * 0.5)
        XCTAssertFalse(ScreenGeometry.coversDisplay(small))
    }
}
