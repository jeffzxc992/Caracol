import Foundation
import QuartzCore

@inline(__always) func clamp<T: Comparable>(_ value: T, _ lower: T, _ upper: T) -> T {
    min(max(value, lower), upper)
}

@inline(__always) func lerp(_ a: Double, _ b: Double, _ t: Double) -> Double {
    a + (b - a) * t
}

@inline(__always) func lerp(_ a: Float, _ b: Float, _ t: Float) -> Float {
    a + (b - a) * t
}

/// Exponential moving average with a dimensionless smoothing factor in (0, 1].
struct EMA {
    private(set) var value: Double?
    let alpha: Double

    init(alpha: Double) {
        self.alpha = clamp(alpha, 0.01, 1.0)
    }

    mutating func update(_ sample: Double) -> Double {
        if let current = value {
            value = current + alpha * (sample - current)
        } else {
            value = sample
        }
        return value!
    }

    mutating func reset() { value = nil }
}

struct RingBuffer<Element> {
    private var storage: [Element] = []
    let capacity: Int

    init(capacity: Int) {
        self.capacity = max(1, capacity)
    }

    mutating func append(_ element: Element) {
        storage.append(element)
        if storage.count > capacity {
            storage.removeFirst(storage.count - capacity)
        }
    }

    mutating func removeAll(keepingCapacity: Bool = true) {
        storage.removeAll(keepingCapacity: keepingCapacity)
    }

    var elements: [Element] { storage }
    var count: Int { storage.count }
    var isEmpty: Bool { storage.isEmpty }
    var last: Element? { storage.last }
}

enum Stats {
    /// p is in [0, 1]. Returns 0 for an empty input.
    static func percentile(_ values: [Double], _ p: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let clamped = clamp(p, 0.0, 1.0)
        let index = Int((Double(sorted.count - 1) * clamped).rounded())
        return sorted[clamp(index, 0, sorted.count - 1)]
    }

    static func mean(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    static func median(_ values: [Double]) -> Double {
        percentile(values, 0.5)
    }
}

@inline(__always) func hostNow() -> CFTimeInterval { CACurrentMediaTime() }
