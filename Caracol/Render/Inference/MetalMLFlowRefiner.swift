import Foundation
import Metal
import CoreVideo
import QuartzCore

/// Experimental Metal 4 machine-learning route. The model is compiled to a
/// `.mtlpackage` with `metal-package-builder -ml` and dispatched through
/// `MTL4MachineLearningCommandEncoder`. Because driver maturity varies, the
/// route is only enabled after a separate-process self test succeeds.
final class MetalMLFlowRefiner {
    static let width = 256
    static let height = 144
    static let pixelCount = width * height
    private static let availabilityKey = "caracol.metalML.available.v1"

    private(set) var isAvailable = false
    private let device: MTLDevice
    private let compiler: MTL4Compiler
    private let queue = DispatchQueue(label: "com.caracol.metalml", qos: .utility)
    private let lock = NSLock()
    private var inFlight = false

    private var pipeline: MTL4MachineLearningPipelineState?
    private var mlQueue: MTL4CommandQueue?
    private var argumentTable: MTL4ArgumentTable?
    private var heap: MTLHeap?
    private var buffers: [MTLBuffer] = []
    private var lastInferenceMs: Double = 0

    init(device: MTLDevice, compiler: MTL4Compiler) {
        self.device = device
        self.compiler = compiler
        isAvailable = UserDefaults.standard.bool(forKey: Self.availabilityKey)
    }

    var inferenceMilliseconds: Double {
        lock.lock(); defer { lock.unlock() }
        return lastInferenceMs
    }

    static func cacheAvailability(_ available: Bool) {
        UserDefaults.standard.set(available, forKey: availabilityKey)
    }

    static var cachedAvailability: Bool {
        UserDefaults.standard.bool(forKey: availabilityKey)
    }

    func refreshAvailability() {
        isAvailable = Self.cachedAvailability
    }

    /// Runs the permission-free self test in a child process so a driver crash
    /// cannot take down Caracol. The child uses the same code path as inference.
    static func runSelfTestInSubprocess(completion: @escaping (Bool) -> Void) {
        guard let executable = Bundle.main.executableURL else {
            completion(false)
            return
        }
        DispatchQueue.global(qos: .utility).async {
            let process = Process()
            process.executableURL = executable
            process.arguments = ["--metal-ml-selftest"]
            let pipe = Pipe()
            process.standardOutput = pipe
            process.standardError = pipe
            do {
                try process.run()
                process.waitUntilExit()
                let ok = process.terminationStatus == 0
                cacheAvailability(ok)
                LogStore.info("ml", "Metal ML self test exited \(process.terminationStatus)")
                DispatchQueue.main.async { completion(ok) }
            } catch {
                LogStore.warn("ml", "Metal ML self test could not launch: \(error)")
                cacheAvailability(false)
                DispatchQueue.main.async { completion(false) }
            }
        }
    }

    func prepare() -> Bool {
        refreshAvailability()
        guard isAvailable, pipeline == nil else { return isAvailable }
        do {
            guard let url = Bundle.main.url(forResource: "IFRNetLite", withExtension: "mtlpackage") else {
                throw NSError(domain: "Caracol", code: 1, userInfo: [NSLocalizedDescriptionKey: "mtlpackage missing"])
            }
            let library = try device.makeLibrary(URL: url)
            guard library.functionNames.contains("main") else {
                throw NSError(domain: "Caracol", code: 2, userInfo: [NSLocalizedDescriptionKey: "ML function main missing"])
            }
            let function = MTL4LibraryFunctionDescriptor()
            function.library = library
            function.name = "main"
            let descriptor = MTL4MachineLearningPipelineDescriptor()
            descriptor.machineLearningFunctionDescriptor = function
            let d3 = MTLTensorExtents([Self.width, Self.height, 3, 1])!
            let d2 = MTLTensorExtents([Self.width, Self.height, 2, 1])!
            // The MPSGraph module orders inputs alphabetically: cur, flow, prev.
            descriptor.setInputDimensions(d3, bufferIndex: 0)
            descriptor.setInputDimensions(d2, bufferIndex: 1)
            descriptor.setInputDimensions(d3, bufferIndex: 2)
            let pipeline = try compiler.makeMachineLearningPipelineState(descriptor: descriptor)
            self.pipeline = pipeline

            let sizes = [Self.width * Self.height * 3 * 4,
                         Self.width * Self.height * 2 * 4,
                         Self.width * Self.height * 3 * 4,
                         Self.width * Self.height * 2 * 4]
            buffers = sizes.map { device.makeBuffer(length: $0, options: .storageModeShared)! }
            let tableDescriptor = MTL4ArgumentTableDescriptor()
            tableDescriptor.maxBufferBindCount = 8
            tableDescriptor.maxTextureBindCount = 2
            tableDescriptor.maxSamplerStateBindCount = 1
            tableDescriptor.initializeBindings = true
            let table = try device.makeArgumentTable(descriptor: tableDescriptor)
            for (index, buffer) in buffers.enumerated() {
                table.setAddress(buffer.gpuAddress, index: index)
            }
            argumentTable = table
            let heapDescriptor = MTLHeapDescriptor()
            heapDescriptor.type = .placement
            heapDescriptor.storageMode = .private
            heapDescriptor.size = max(pipeline.intermediatesHeapSize, 16 * 1024)
            heap = device.makeHeap(descriptor: heapDescriptor)
            mlQueue = device.makeMTL4CommandQueue()
            LogStore.info("ml", "Metal ML pipeline ready (heap \(pipeline.intermediatesHeapSize) bytes)")
            return true
        } catch {
            LogStore.error("ml", "Metal ML pipeline unavailable: \(error)")
            isAvailable = false
            return false
        }
    }

    func refine(pairID: UInt64,
                previous: CVPixelBuffer,
                current: CVPixelBuffer,
                flow: MTLTexture,
                output: MTLTexture,
                completion: @escaping (UInt64) -> Void) {
        guard isAvailable, prepare(), let pipeline, let mlQueue, let argumentTable, let heap else { return }
        guard buffers.count == 4 else { return }
        lock.lock()
        if inFlight {
            lock.unlock()
            return
        }
        inFlight = true
        lock.unlock()

        let start = hostNow()
        queue.async { [weak self] in
            guard let self else { return }
            // Input binding order established by the compiled package:
            // 0 = prev, 1 = flow, 2 = cur, 3 = flow_out.
            Self.fillImage(previous, buffer: self.buffers[0])
            Self.fillFlow(flow, buffer: self.buffers[1])
            Self.fillImage(current, buffer: self.buffers[2])
            memset(self.buffers[3].contents(), 0, self.buffers[3].length)

            guard let allocator = self.device.makeCommandAllocator(),
                  let commandBuffer = self.device.makeCommandBuffer() else {
                self.finish(pairID: pairID, start: start, output: nil, texture: output, completion: completion)
                return
            }
            commandBuffer.beginCommandBuffer(allocator: allocator)
            guard let encoder = commandBuffer.makeMachineLearningCommandEncoder() else {
                commandBuffer.endCommandBuffer()
                self.finish(pairID: pairID, start: start, output: nil, texture: output, completion: completion)
                return
            }
            encoder.setPipelineState(pipeline)
            encoder.setArgumentTable(argumentTable)
            encoder.dispatchNetwork(intermediatesHeap: heap)
            encoder.endEncoding()
            commandBuffer.endCommandBuffer()

            let options = MTL4CommitOptions()
            options.addFeedbackHandler { [weak self] feedback in
                guard let self else { return }
                if let error = feedback.error {
                    LogStore.error("ml", "Metal ML dispatch error: \(error)")
                    self.finish(pairID: pairID, start: start, output: nil, texture: output, completion: completion)
                    return
                }
                let outputBuffer = self.buffers[3]
                self.finish(pairID: pairID, start: start, output: outputBuffer, texture: output, completion: completion)
            }
            mlQueue.commit([commandBuffer], options: options)
        }
    }

    private func finish(pairID: UInt64,
                        start: CFTimeInterval,
                        output: MTLBuffer?,
                        texture: MTLTexture,
                        completion: @escaping (UInt64) -> Void) {
        if let output {
            Self.writeFlow(output, to: texture)
        }
        let elapsed = (hostNow() - start) * 1000.0
        lock.lock()
        lastInferenceMs = elapsed
        inFlight = false
        lock.unlock()
        DispatchQueue.main.async {
            completion(pairID)
        }
    }

    private static func fillImage(_ pixelBuffer: CVPixelBuffer, buffer: MTLBuffer) {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }
        let srcW = CVPixelBufferGetWidth(pixelBuffer)
        let srcH = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: width * height * 3)

        for y in 0..<height {
            let sy = min(srcH - 1, y * srcH / height)
            let row = base.advanced(by: sy * bytesPerRow)
            for x in 0..<width {
                let sx = min(srcW - 1, x * srcW / width)
                let pixel = row.advanced(by: sx * 4).assumingMemoryBound(to: UInt8.self)
                let index = (y * width + x) * 3
                ptr[index] = Float(pixel[2]) / 255.0
                ptr[index + 1] = Float(pixel[1]) / 255.0
                ptr[index + 2] = Float(pixel[0]) / 255.0
            }
        }
    }

    private static func fillFlow(_ texture: MTLTexture, buffer: MTLBuffer) {
        let w = texture.width
        let h = texture.height
        guard w > 0, h > 0 else { return }
        var raw = [UInt16](repeating: 0, count: w * h * 4)
        raw.withUnsafeMutableBytes { bytes in
            texture.getBytes(bytes.baseAddress!, bytesPerRow: w * 8, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        }
        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: width * height * 2)
        for y in 0..<height {
            let sy = min(h - 1, y * h / height)
            for x in 0..<width {
                let sx = min(w - 1, x * w / width)
                let i = (sy * w + sx) * 4
                let index = (y * width + x) * 2
                let fx = Float(Float16(bitPattern: raw[i]))
                let fy = Float(Float16(bitPattern: raw[i + 1]))
                ptr[index] = fx.isFinite ? fx : 0
                ptr[index + 1] = fy.isFinite ? fy : 0
            }
        }
    }

    private static func writeFlow(_ buffer: MTLBuffer, to texture: MTLTexture) {
        guard texture.width == width, texture.height == height else { return }
        let ptr = buffer.contents().bindMemory(to: Float.self, capacity: width * height * 2)
        var out = [UInt16](repeating: 0, count: width * height * 4)
        for i in 0..<(width * height) {
            out[i * 4 + 0] = Float16(ptr[i * 2]).bitPattern
            out[i * 4 + 1] = Float16(ptr[i * 2 + 1]).bitPattern
            out[i * 4 + 2] = Float16(1.0).bitPattern
        }
        out.withUnsafeBytes { bytes in
            texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0, withBytes: bytes.baseAddress!, bytesPerRow: width * 8)
        }
    }
}

/// Runs inside the child process. Never call this in the main app process: a
/// failed dispatch can crash.
enum MetalMLSelfTest {
    static func run() -> Int32 {
        guard let device = MTLCreateSystemDefaultDevice(),
              let url = Bundle.main.url(forResource: "IFRNetLite", withExtension: "mtlpackage") else {
            return 1
        }
        do {
            let library = try device.makeLibrary(URL: url)
            guard library.functionNames.contains("main") else { return 2 }
            let compiler = try device.makeCompiler(descriptor: MTL4CompilerDescriptor())
            let function = MTL4LibraryFunctionDescriptor()
            function.library = library
            function.name = "main"
            let descriptor = MTL4MachineLearningPipelineDescriptor()
            descriptor.machineLearningFunctionDescriptor = function
            let d3 = MTLTensorExtents([256, 144, 3, 1])!
            let d2 = MTLTensorExtents([256, 144, 2, 1])!
            descriptor.setInputDimensions(d3, bufferIndex: 0)
            descriptor.setInputDimensions(d2, bufferIndex: 1)
            descriptor.setInputDimensions(d3, bufferIndex: 2)
            let pipeline = try compiler.makeMachineLearningPipelineState(descriptor: descriptor)
            let sizes = [256 * 144 * 3 * 4, 256 * 144 * 2 * 4, 256 * 144 * 3 * 4, 256 * 144 * 2 * 4]
            let buffers = sizes.map { device.makeBuffer(length: $0, options: .storageModeShared)! }
            let tableDescriptor = MTL4ArgumentTableDescriptor()
            tableDescriptor.maxBufferBindCount = 8
            tableDescriptor.maxTextureBindCount = 2
            tableDescriptor.maxSamplerStateBindCount = 1
            tableDescriptor.initializeBindings = true
            let table = try device.makeArgumentTable(descriptor: tableDescriptor)
            for (index, buffer) in buffers.enumerated() {
                table.setAddress(buffer.gpuAddress, index: index)
            }
            let heapDescriptor = MTLHeapDescriptor()
            heapDescriptor.type = .placement
            heapDescriptor.storageMode = .private
            heapDescriptor.size = max(pipeline.intermediatesHeapSize, 16 * 1024)
            guard let heap = device.makeHeap(descriptor: heapDescriptor) else { return 3 }
            guard let queue = device.makeMTL4CommandQueue(),
                  let allocator = device.makeCommandAllocator(),
                  let commandBuffer = device.makeCommandBuffer() else { return 3 }
            commandBuffer.beginCommandBuffer(allocator: allocator)
            guard let encoder = commandBuffer.makeMachineLearningCommandEncoder() else { return 4 }
            encoder.setPipelineState(pipeline)
            encoder.setArgumentTable(table)
            encoder.dispatchNetwork(intermediatesHeap: heap)
            encoder.endEncoding()
            commandBuffer.endCommandBuffer()

            let semaphore = DispatchSemaphore(value: 0)
            var failed = true
            let options = MTL4CommitOptions()
            options.addFeedbackHandler { feedback in
                failed = feedback.error != nil
                semaphore.signal()
            }
            queue.commit([commandBuffer], options: options)
            if semaphore.wait(timeout: .now() + 5) == .timedOut { return 5 }
            return failed ? 6 : 0
        } catch {
            return 7
        }
    }
}
