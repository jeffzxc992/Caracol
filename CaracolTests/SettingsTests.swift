import XCTest
@testable import Caracol

final class SettingsTests: XCTestCase {
    func testSanitizeClampsFPSAndSharpness() {
        var settings = CaracolSettings()
        settings.targetFPS = 120
        settings.sharpness = 4
        let sanitized = settings.sanitized(displayRefresh: 60)
        XCTAssertEqual(sanitized.targetFPS, 60)
        XCTAssertEqual(sanitized.sharpness, 1)

        settings.targetFPS = 10
        settings.sharpness = -1
        let low = settings.sanitized(displayRefresh: 60)
        XCTAssertEqual(low.targetFPS, 30)
        XCTAssertEqual(low.sharpness, 0)
    }

    func testSanitizeRespectsLowRefreshDisplay() {
        var settings = CaracolSettings()
        settings.targetFPS = 60
        let sanitized = settings.sanitized(displayRefresh: 40)
        XCTAssertEqual(sanitized.targetFPS, 40)
    }

    func testProfileKeys() {
        XCTAssertEqual(GameProfile.key(bundleID: "com.example.game", title: "Game", windowID: 12),
                       "bundle:com.example.game")
        XCTAssertEqual(GameProfile.key(bundleID: nil, title: "CrossOver Game", windowID: 12),
                       "title:crossover game")
        XCTAssertEqual(GameProfile.key(bundleID: nil, title: "Game", windowID: 12),
                       "window:12")
    }

    func testTargetChoicesAreOrdered() {
        XCTAssertEqual(CaracolSettings.targetFPSChoices, [30, 40, 50, 60])
    }
}
