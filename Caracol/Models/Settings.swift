import Foundation
import SwiftUI

enum RenderMode: String, Codable, CaseIterable, Identifiable {
    case fast
    case quality

    var id: String { rawValue }
    var label: String {
        switch self {
        case .fast: return "Fast"
        case .quality: return "Quality"
        }
    }
    var detail: String {
        switch self {
        case .fast: return "GPU optical flow only. No model. Lowest latency."
        case .quality: return "Adds a neural flow refinement on the Neural Engine."
        }
    }
}

enum UpscaleMode: String, Codable, CaseIterable, Identifiable {
    case off
    case x15
    case x2

    var id: String { rawValue }
    var factor: CGFloat {
        switch self {
        case .off: return 1.0
        case .x15: return 1.5
        case .x2: return 2.0
        }
    }
    var label: String {
        switch self {
        case .off: return "Native"
        case .x15: return "1.5×"
        case .x2: return "2×"
        }
    }
}

enum FlowQuality: String, Codable, CaseIterable, Identifiable {
    case quarter
    case eighth

    var id: String { rawValue }
    var divisor: Int { self == .quarter ? 4 : 8 }
    var label: String { self == .quarter ? "1/4" : "1/8" }
}

enum InferenceRoute: String, Codable, CaseIterable, Identifiable {
    case gpuOnly
    case coreML
    case metalML

    var id: String { rawValue }
    var label: String {
        switch self {
        case .gpuOnly: return "GPU flow only"
        case .coreML: return "Core ML (ANE)"
        case .metalML: return "Metal ML (MTL4)"
        }
    }
    var shortLabel: String {
        switch self {
        case .gpuOnly: return "GPU"
        case .coreML: return "ANE"
        case .metalML: return "MTL4"
        }
    }
}

struct CaracolSettings: Codable, Equatable {
    var targetFPS: Int = 60
    var mode: RenderMode = .fast
    var upscale: UpscaleMode = .x2
    var sharpness: Double = 0.6
    var showHUD: Bool = true
    var flowQuality: FlowQuality = .quarter
    var forceFullScreen: Bool = false
    var frameGeneration: Bool = true
    var inferenceRoute: InferenceRoute = .gpuOnly
    var hotkeyEnabled: Bool = true
    var displayRefresh: Int = 60
    var calibrated: Bool = false

    static let targetFPSChoices = [30, 40, 50, 60]

    func sanitized(displayRefresh: Int) -> CaracolSettings {
        var copy = self
        let maximum = clamp(displayRefresh > 0 ? displayRefresh : 60, 30, 120)
        copy.targetFPS = clamp(copy.targetFPS, 30, maximum)
        copy.sharpness = clamp(copy.sharpness, 0, 1)
        return copy
    }
}

@MainActor
final class SettingsStore: ObservableObject {
    @Published var settings: CaracolSettings {
        didSet { save() }
    }

    var onChange: ((CaracolSettings) -> Void)?

    private let defaultsKey = "caracol.settings.v1"

    init() {
        if let data = UserDefaults.standard.data(forKey: defaultsKey),
           let decoded = try? JSONDecoder().decode(CaracolSettings.self, from: data) {
            settings = decoded
        } else {
            settings = CaracolSettings()
        }
        settings = settings.sanitized(displayRefresh: settings.displayRefresh)
    }

    func refreshDisplayRate(_ hz: Int) {
        guard hz > 0 else { return }
        settings.displayRefresh = hz
        settings = settings.sanitized(displayRefresh: hz)
    }

    private func save() {
        if let data = try? JSONEncoder().encode(settings) {
            UserDefaults.standard.set(data, forKey: defaultsKey)
        }
        onChange?(settings)
    }
}
