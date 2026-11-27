import XCTest
@testable import Caracol

final class MathTests: XCTestCase {
    func testPercentile() {
        let values = (1...100).map(Double.init)
        XCTAssertEqual(Stats.percentile(values, 0.5), 51, accuracy: 0.001)
        XCTAssertEqual(Stats.percentile(values, 0.95), 95, accuracy: 0.001)
        XCTAssertEqual(Stats.percentile([], 0.5), 0, accuracy: 0.001)
    }

    func testEMAConverges() {
        var ema = EMA(alpha: 0.5)
        XCTAssertNil(ema.value)
        XCTAssertEqual(ema.update(10), 10, accuracy: 0.001)
        XCTAssertEqual(ema.update(20), 15, accuracy: 0.001)
        ema.reset()
        XCTAssertNil(ema.value)
    }

    func testRingBufferDropsOldest() {
        var ring = RingBuffer<Int>(capacity: 3)
        for value in 1...5 { ring.append(value) }
        XCTAssertEqual(ring.elements, [3, 4, 5])
    }

    func testFPSMeter() {
        var meter = FPSMeter(window: 1.0)
        let start = hostNow()
        for i in 0..<60 {
            meter.record(start + Double(i) / 60.0)
        }
        let value = meter.value(at: start + 0.99)
        XCTAssertGreaterThan(value, 30)
        XCTAssertLessThanOrEqual(value, 61)
    }
}
