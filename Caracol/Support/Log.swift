import Foundation
import os

enum Log {
    static let subsystem = "com.caracol.app"
    static let app = Logger(subsystem: subsystem, category: "app")
    static let capture = Logger(subsystem: subsystem, category: "capture")
    static let render = Logger(subsystem: subsystem, category: "render")
    static let flow = Logger(subsystem: subsystem, category: "flow")
    static let ml = Logger(subsystem: subsystem, category: "ml")
    static let metrics = Logger(subsystem: subsystem, category: "metrics")
    static let hotkey = Logger(subsystem: subsystem, category: "hotkey")
}

/// A small in-memory log ring that the user can export from the menu.
final class LogStore {
    static let shared = LogStore()

    private let lock = NSLock()
    private var lines: [String] = []
    private let limit = 4000
    private let formatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()

    private init() {}

    func write(_ category: String, _ message: String) {
        let line = "\(formatter.string(from: Date())) [\(category)] \(message)"
        lock.lock()
        lines.append(line)
        if lines.count > limit { lines.removeFirst(lines.count - limit) }
        lock.unlock()
    }

    func tail(_ count: Int = 200) -> [String] {
        lock.lock()
        defer { lock.unlock() }
        return Array(lines.suffix(count))
    }

    func export(to url: URL) throws {
        lock.lock()
        let snapshot = lines
        lock.unlock()
        try snapshot.joined(separator: "\n").appending("\n").write(to: url, atomically: true, encoding: .utf8)
    }

    static func info(_ category: String, _ message: String) {
        LogStore.shared.write(category, message)
        Log.app.info("\(message, privacy: .public)")
    }

    static func warn(_ category: String, _ message: String) {
        LogStore.shared.write(category, "WARN " + message)
        Log.app.warning("\(message, privacy: .public)")
    }

    static func error(_ category: String, _ message: String) {
        LogStore.shared.write(category, "ERROR " + message)
        Log.app.error("\(message, privacy: .public)")
    }
}
