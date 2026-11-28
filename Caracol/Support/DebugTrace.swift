import Foundation

/// Lightweight stdout tracing for the renderer. Enable with --debug-trace.
enum DebugTrace {
    static let enabled = CommandLine.arguments.contains("--debug-trace")
    static var allowsCapture = CommandLine.arguments.contains("--allow-capture")
    static var simplePresent = CommandLine.arguments.contains("--simple-present")
    static var clearRed = CommandLine.arguments.contains("--clear-red")
    static var solidGreen = CommandLine.arguments.contains("--solid-green")
    static var debugTexture = CommandLine.arguments.contains("--debug-texture")
    static var fixedUV = CommandLine.arguments.contains("--fixed-uv")
    static var noScaler = CommandLine.arguments.contains("--no-scaler")
    static var noFlow = CommandLine.arguments.contains("--no-flow")
    static var presentStage: Int = {
        let args = CommandLine.arguments
        if let index = args.firstIndex(of: "--present-stage"), index + 1 < args.count,
           let value = Int(args[index + 1]) {
            return value
        }
        return 3
    }()
    private static var counts: [String: Int] = [:]
    private static let lock = NSLock()

    static func log(_ tag: String, _ message: @autoclosure () -> String, limit: Int = 200) {
        guard enabled else { return }
        lock.lock()
        let count = counts[tag, default: 0]
        counts[tag] = count + 1
        lock.unlock()
        guard count < limit else { return }
        print("[\(tag)#\(count)] \(message())")
        fflush(stdout)
    }
}
