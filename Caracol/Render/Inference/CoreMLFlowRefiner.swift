import Foundation
import CoreML
import CoreVideo
import Metal
import QuartzCore

/// Core ML route: fixed-shape FP16-friendly model with three inputs
/// (`prev`, `cur`, `flow`) and one output (`flow_out`). The bundled stub is an
/// identity flow model used to validate the plumbing; replace it with trained
/// RIFE/IFRNet weights via Tools/convert_model.py for real quality gains.
final class CoreMLFlowRefiner {
    static let width = 256
    static let height = 144
    static let pixelCount = width * height

    private(set) var isAvailable = false
    private var model: MLModel?
    private let queue = DispatchQueue(label: "com.caracol.coreml", qos: .utility)
    private let lock = NSLock()
    private var inFlight = false
    private var warmed = false

    private var prevArray: MLMultiArray?
    private var curArray: MLMultiArray?
    private var flowArray: MLMultiArray?
    private var lastInferenceMs: Double = 0

    init() {
        guard let url = Bundle.main.url(forResource: "IFRNetLite", withExtension: "mlmodelc") else {
            LogStore.warn("ml", "Core ML model IFRNetLite.mlmodelc not found in bundle")
            return
        }
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .cpuAndNeuralEngine
            let loaded = try MLModel(contentsOf: url, configuration: config)
            let inputs = loaded.modelDescription.inputDescriptionsByName
            guard inputs["prev"] != nil, inputs["cur"] != nil, inputs["flow"] != nil else {
                LogStore.error("ml", "Core ML model is missing prev/cur/flow inputs")
                return
            }
            model = loaded
            prevArray = try MLMultiArray(shape: [1, 3, NSNumber(value: Self.height), NSNumber(value: Self.width)], dataType: .float32)
            curArray = try MLMultiArray(shape: [1, 3, NSNumber(value: Self.height), NSNumber(value: Self.width)], dataType: .float32)
            flowArray = try MLMultiArray(shape: [1, 2, NSNumber(value: Self.height), NSNumber(value: Self.width)], dataType: .float32)
            isAvailable = true
            warmUp()
            LogStore.info("ml", "Core ML model loaded (ANE preferred)")
        } catch {
            LogStore.error("ml", "Core ML model load failed: \(error)")
        }
    }

    var inferenceMilliseconds: Double {
        lock.lock(); defer { lock.unlock() }
        return lastInferenceMs
    }

    func refine(pairID: UInt64,
                previous: CVPixelBuffer,
                current: CVPixelBuffer,
                flow: MTLTexture,
                output: MTLTexture,
                completion: @escaping (UInt64) -> Void) {
        guard isAvailable, let model, let prevArray, let curArray, let flowArray else { return }
        lock.lock()
        if inFlight {
            lock.unlock()
            return
        }
        inFlight = true
        lock.unlock()

        queue.async { [weak self] in
            guard let self else { return }
            let start = hostNow()
            defer {
                self.lock.lock()
                self.inFlight = false
                self.lock.unlock()
            }

            Self.fillImage(previous, into: prevArray)
            Self.fillImage(current, into: curArray)
            Self.fillFlow(flow, into: flowArray)

            do {
                let input = try MLDictionaryFeatureProvider(dictionary: [
                    "prev": MLFeatureValue(multiArray: prevArray),
                    "cur": MLFeatureValue(multiArray: curArray),
                    "flow": MLFeatureValue(multiArray: flowArray)
                ])
                let result = try model.prediction(from: input)
                guard let outputArray = result.featureValue(for: "flow_out")?.multiArrayValue else {
                    LogStore.warn("ml", "Core ML prediction returned no flow_out")
                    return
                }
                Self.writeFlow(outputArray, to: output)
                let elapsed = (hostNow() - start) * 1000.0
                self.lock.lock()
                self.lastInferenceMs = elapsed
                self.lock.unlock()
                DispatchQueue.main.async {
                    completion(pairID)
                }
            } catch {
                LogStore.error("ml", "Core ML prediction failed: \(error)")
            }
        }
    }

    private func warmUp() {
        guard let model, let prevArray, let curArray, let flowArray else { return }
        queue.async { [weak self] in
            Self.zero(prevArray)
            Self.zero(curArray)
            Self.zero(flowArray)
            do {
                let input = try MLDictionaryFeatureProvider(dictionary: [
                    "prev": MLFeatureValue(multiArray: prevArray),
                    "cur": MLFeatureValue(multiArray: curArray),
                    "flow": MLFeatureValue(multiArray: flowArray)
                ])
                _ = try model.prediction(from: input)
                self?.lock.lock()
                self?.warmed = true
                self?.lock.unlock()
                LogStore.info("ml", "Core ML warm-up complete")
            } catch {
                LogStore.warn("ml", "Core ML warm-up failed: \(error)")
            }
        }
    }

    private static func zero(_ array: MLMultiArray) {
        memset(array.dataPointer, 0, array.count * MemoryLayout<Float>.stride)
    }

    private static func fillImage(_ pixelBuffer: CVPixelBuffer, into array: MLMultiArray) {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }
        let srcW = CVPixelBufferGetWidth(pixelBuffer)
        let srcH = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        let plane = pixelCount

        for y in 0..<height {
            let sy = min(srcH - 1, y * srcH / height)
            let row = base.advanced(by: sy * bytesPerRow)
            for x in 0..<width {
                let sx = min(srcW - 1, x * srcW / width)
                let pixel = row.advanced(by: sx * 4).assumingMemoryBound(to: UInt8.self)
                let b = Float(pixel[0]) / 255.0
                let g = Float(pixel[1]) / 255.0
                let r = Float(pixel[2]) / 255.0
                let index = y * width + x
                ptr[index] = r
                ptr[plane + index] = g
                ptr[plane * 2 + index] = b
            }
        }
    }

    private static func fillFlow(_ texture: MTLTexture, into array: MLMultiArray) {
        let w = texture.width
        let h = texture.height
        guard w > 0, h > 0 else { return }
        var raw = [UInt16](repeating: 0, count: w * h * 4)
        raw.withUnsafeMutableBytes { bytes in
            texture.getBytes(
                bytes.baseAddress!,
                bytesPerRow: w * 8,
                from: MTLRegionMake2D(0, 0, w, h),
                mipmapLevel: 0
            )
        }
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        let plane = pixelCount
        for y in 0..<height {
            let sy = min(h - 1, y * h / height)
            for x in 0..<width {
                let sx = min(w - 1, x * w / width)
                let i = (sy * w + sx) * 4
                let fx = Float(Float16(bitPattern: raw[i]))
                let fy = Float(Float16(bitPattern: raw[i + 1]))
                let index = y * width + x
                ptr[index] = fx.isFinite ? fx : 0
                ptr[plane + index] = fy.isFinite ? fy : 0
            }
        }
    }

    private static func writeFlow(_ array: MLMultiArray, to texture: MTLTexture) {
        guard texture.width == width, texture.height == height else { return }
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        let plane = pixelCount
        var out = [UInt16](repeating: 0, count: pixelCount * 4)
        for i in 0..<plane {
            let fx = ptr[i].isFinite ? ptr[i] : 0
            let fy = ptr[plane + i].isFinite ? ptr[plane + i] : 0
            out[i * 4 + 0] = Float16(fx).bitPattern
            out[i * 4 + 1] = Float16(fy).bitPattern
            out[i * 4 + 2] = Float16(1.0).bitPattern
            out[i * 4 + 3] = 0
        }
        out.withUnsafeBytes { bytes in
            texture.replace(
                region: MTLRegionMake2D(0, 0, width, height),
                mipmapLevel: 0,
                withBytes: bytes.baseAddress!,
                bytesPerRow: width * 8
            )
        }
    }
}
