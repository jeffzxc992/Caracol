import Foundation

struct GameProfile: Codable, Identifiable, Equatable {
    var key: String
    var title: String
    var bundleID: String?
    var windowID: UInt32?
    var settings: CaracolSettings
    var measuredSourceFPS: Double
    var calibratedAt: Date
    var lastUsedAt: Date

    var id: String { key }

    static func key(bundleID: String?, title: String, windowID: UInt32?) -> String {
        if let bundleID, !bundleID.isEmpty, bundleID != "com.caracol.video-test" {
            return "bundle:\(bundleID)"
        }
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty, trimmed.lowercased() != "game", trimmed.lowercased() != "display" {
            return "title:\(trimmed.lowercased())"
        }
        if let windowID { return "window:\(windowID)" }
        return "title:\(title.lowercased())"
    }
}

final class ProfileStore {
    static let shared = ProfileStore()

    private let lock = NSLock()
    private var profiles: [String: GameProfile] = [:]
    private let fileURL: URL

    private init() {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent("Library/Application Support")
        let directory = base.appendingPathComponent("Caracol", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        fileURL = directory.appendingPathComponent("profiles.json")
        load()
    }

    var storageURL: URL { fileURL }

    func allProfiles() -> [GameProfile] {
        lock.lock()
        defer { lock.unlock() }
        return profiles.values.sorted { $0.lastUsedAt > $1.lastUsedAt }
    }

    func profile(for key: String) -> GameProfile? {
        lock.lock()
        defer { lock.unlock() }
        return profiles[key]
    }

    func upsert(_ profile: GameProfile) {
        lock.lock()
        profiles[profile.key] = profile
        let snapshot = Array(profiles.values)
        lock.unlock()
        save(snapshot)
    }

    func touch(key: String) {
        lock.lock()
        if var profile = profiles[key] {
            profile.lastUsedAt = Date()
            profiles[key] = profile
        }
        let snapshot = Array(profiles.values)
        lock.unlock()
        save(snapshot)
    }

    private func load() {
        guard let data = try? Data(contentsOf: fileURL),
              let decoded = try? JSONDecoder().decode([GameProfile].self, from: data) else { return }
        profiles = Dictionary(uniqueKeysWithValues: decoded.map { ($0.key, $0) })
    }

    private func save(_ snapshot: [GameProfile]) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        guard let data = try? encoder.encode(snapshot) else { return }
        try? data.write(to: fileURL, options: .atomic)
    }
}
