import Foundation
import Metal
import MetalFX
import CoreVideo
import QuartzCore

enum RendererError: LocalizedError {
    case noCommandQueue
    case noLibrary
    case missingFunction(String)
    case pipeline(String)
    case textureCache
    case texture(String)
    case allocator

    var errorDescription: String? {
        switch self {
        case .noCommandQueue: return "Metal 4 command queue is unavailable."
        case .noLibrary: return "The compiled Metal shader library is missing."
        case .missingFunction(let name): return "Metal shader function \(name) is missing."
        case .pipeline(let name): return "Could not create pipeline \(name)."
        case .textureCache: return "Could not create the CVMetalTextureCache."
        case .texture(let name): return "Could not create texture \(name)."
        case .allocator: return "Could not create a Metal 4 command allocator."
        }
    }
}

struct RendererConfig {
    var mode: RenderMode = .fast
    var targetFPS: Int = 60
    var sharpness: Float = 0.6
    var inputSize: CGSize = .zero
    var outputSize: CGSize = .zero
    var flowBlock: Int = 8
    var frameGeneration: Bool = true
    var interpolation: Bool = false
    var inferenceRoute: InferenceRoute = .gpuOnly
    var displayRefresh: Int = 60
}

private struct RenderUniforms {
    var p0 = SIMD4<Float>(0, 0, 0, 0)
    var p1 = SIMD4<Float>(0, 0, 0, 0)
    var p2 = SIMD4<Float>(0, 0, 0, 0)
    var p3 = SIMD4<Float>(0, 0, 0, 0)
}

private struct FrameSlot {
    let pairID: UInt64
    let texture: MTLTexture
    let cvTexture: CVMetalTexture
    let pixelBuffer: CVPixelBuffer
    let hostTime: CFTimeInterval
}

private final class FlowSet {
    let lumaPrevious: MTLTexture
    let lumaCurrent: MTLTexture
    let coarse: MTLTexture
    let flowA: MTLTexture
    let flowB: MTLTexture
    var pairID: UInt64 = 0
    var ready = false

    init(lumaPrevious: MTLTexture, lumaCurrent: MTLTexture, coarse: MTLTexture,
         flowA: MTLTexture, flowB: MTLTexture) {
        self.lumaPrevious = lumaPrevious
        self.lumaCurrent = lumaCurrent
        self.coarse = coarse
        self.flowA = flowA
        self.flowB = flowB
    }
}

private struct FrameTargets {
    let warped: MTLTexture
    let scaled: MTLTexture
    let uniforms: MTLBuffer
}

/// Recycles MTL4CommandAllocator instances. Metal 4 allocators hold encoded
/// commands, so an allocator is only reset after the GPU reports completion.
private final class AllocatorPool {
    private let device: MTLDevice
    private let lock = NSLock()
    private var free: [MTL4CommandAllocator] = []

    init(device: MTLDevice) { self.device = device }

    func acquire() -> MTL4CommandAllocator? {
        lock.lock()
        let recycled = free.popLast()
        lock.unlock()
        if let recycled {
            recycled.reset()
            return recycled
        }
        return device.makeCommandAllocator()
    }

    func recycle(_ allocator: MTL4CommandAllocator) {
        lock.lock()
        if free.count < 3 { free.append(allocator) }
        lock.unlock()
    }
}

/// Metal 4 renderer. Real frames are presented immediately; generated frames
/// are only produced when the latest-frame-wins mailbox has no newer real
/// frame. Flow analysis runs on a second queue and never blocks presentation.
final class Renderer {
    let device: MTLDevice
    let mailbox: FrameMailbox
    let metrics: MetricsStore
    private(set) var config = RendererConfig()

    private let presentQueue: MTL4CommandQueue
    private let flowQueue: MTL4CommandQueue
    private let compiler: MTL4Compiler
    private let library: MTLLibrary
    private var textureCache: CVMetalTextureCache?

    private let lumaPipeline: MTLComputePipelineState
    private let blockPipeline: MTLComputePipelineState
    private let refinePipeline: MTLComputePipelineState
    private let warpPipeline: MTLComputePipelineState
    private let casPipeline: MTLRenderPipelineState
    private let residencySet: MTLResidencySet?

    private let presentAllocators: AllocatorPool
    private let flowAllocators: AllocatorPool

    private var flowSets: [FlowSet] = []
    private var targets: [FrameTargets] = []
    private var zeroFlow: MTLTexture?
    private var mlFlows: [MTLTexture] = []
    private var mlFlowIndex = 0
    private var nextFlowSet = 0
    private var targetIndex = 0
    private var flowUniformBuffers: [MTLBuffer] = []
    private var flowUniformIndex = 0
    private var flowUniformBuffer: MTLBuffer?
    private var flowUniformOffset = 0
    private var scaler: MTL4FXSpatialScaler?
    private var scalerInputSize = CGSize.zero
    private var scalerOutputSize = CGSize.zero
    private var coreML: CoreMLFlowRefiner?
    private var metalML: MetalMLFlowRefiner?

    private var previous: FrameSlot?
    private var current: FrameSlot?
    private var lastSequence: UInt64 = 0
    private var lastRealSlotHostTime: CFTimeInterval = 0
    private var lastPresentHost: CFTimeInterval = 0
    private var sourceInterval = EMA(alpha: 0.2)
    private var sourceFPS: Double = 0
    private var generationCredit: Double = 0
    private var flowInFlight = false
    private var flowPreviousSlot: FrameSlot?
    private var flowCurrentSlot: FrameSlot?
    private var mlReadyID: UInt64 = 0
    private var mlReadyTexture: MTLTexture?
    private var mlGeneration: UInt64 = 0
    private var isPaused = false

    var isMetalMLAvailable: Bool { metalML?.isAvailable ?? false }

    func refreshMetalMLAvailability() {
        metalML?.refreshAvailability()
    }

    init(device: MTLDevice, mailbox: FrameMailbox, metrics: MetricsStore) throws {
        self.device = device
        self.mailbox = mailbox
        self.metrics = metrics

        guard let presentQueue = device.makeMTL4CommandQueue(),
              let flowQueue = device.makeMTL4CommandQueue() else {
            throw RendererError.noCommandQueue
        }
        self.presentQueue = presentQueue
        self.flowQueue = flowQueue
        self.compiler = try device.makeCompiler(descriptor: MTL4CompilerDescriptor())
        self.presentAllocators = AllocatorPool(device: device)
        self.flowAllocators = AllocatorPool(device: device)

        guard let library = device.makeDefaultLibrary() else {
            throw RendererError.noLibrary
        }
        self.library = library

        func computePipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let function = library.makeFunction(name: name) else {
                throw RendererError.missingFunction(name)
            }
            do {
                return try device.makeComputePipelineState(function: function)
            } catch {
                throw RendererError.pipeline(name)
            }
        }

        self.lumaPipeline = try computePipeline("luma_downsample")
        self.blockPipeline = try computePipeline("block_match")
        self.refinePipeline = try computePipeline("flow_refine")
        self.warpPipeline = try computePipeline("warp_compute")

        guard let vertex = library.makeFunction(name: "fullscreen_vertex"),
              let fragment = library.makeFunction(name: "cas_fragment") else {
            throw RendererError.missingFunction("cas_fragment")
        }
        let renderDescriptor = MTLRenderPipelineDescriptor()
        renderDescriptor.label = "Caracol CAS"
        renderDescriptor.vertexFunction = vertex
        renderDescriptor.fragmentFunction = fragment
        renderDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        do {
            self.casPipeline = try device.makeRenderPipelineState(descriptor: renderDescriptor)
        } catch {
            throw RendererError.pipeline("cas_fragment")
        }

        var cache: CVMetalTextureCache?
        let status = CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, device, nil, &cache)
        guard status == kCVReturnSuccess, let cache else {
            throw RendererError.textureCache
        }
        self.textureCache = cache

        if let set = try? device.makeResidencySet(descriptor: MTLResidencySetDescriptor()) {
            residencySet = set
            presentQueue.addResidencySet(set)
            flowQueue.addResidencySet(set)
        } else {
            residencySet = nil
        }

        metalML = MetalMLFlowRefiner(device: device, compiler: compiler)
        LogStore.info("render", "Renderer initialized on \(device.name)")
    }

    func configure(settings: CaracolSettings, layout: CaptureLayout) {
        let oldInput = config.inputSize
        let oldOutput = config.outputSize
        config.mode = settings.mode
        config.targetFPS = settings.targetFPS
        config.sharpness = Float(settings.sharpness)
        config.flowBlock = settings.flowQuality == .quarter ? 8 : 16
        config.frameGeneration = settings.frameGeneration
        config.interpolation = settings.interpolation
        config.inferenceRoute = settings.inferenceRoute
        config.displayRefresh = settings.displayRefresh
        config.inputSize = layout.inputSize
        config.outputSize = layout.outputSize

        let changed = Int(oldInput.width) != Int(layout.inputSize.width) ||
            Int(oldInput.height) != Int(layout.inputSize.height) ||
            Int(oldOutput.width) != Int(layout.outputSize.width) ||
            Int(oldOutput.height) != Int(layout.outputSize.height)
        if changed || targets.isEmpty {
            rebuild()
        }
        if settings.mode == .quality, coreML == nil {
            coreML = CoreMLFlowRefiner()
        }
        if settings.inferenceRoute == .metalML {
            _ = metalML?.prepare()
        }
        metrics.setMode(settings)
    }

    func pause() {
        isPaused = true
    }

    func resume() {
        isPaused = false
        lastSequence = mailbox.latest()?.sequence ?? 0
        lastRealSlotHostTime = 0
        previous = nil
        current = nil
        generationCredit = 0
        sourceInterval.reset()
        sourceFPS = 0
    }

    func stop() {
        isPaused = true
        mailbox.clear()
        previous = nil
        current = nil
        lastSequence = 0
        generationCredit = 0
    }

    // MARK: - Textures

    private func rebuild() {
        let inputWidth = max(2, Int(config.inputSize.width.rounded()) & ~1)
        let inputHeight = max(2, Int(config.inputSize.height.rounded()) & ~1)
        let outputWidth = max(2, Int(config.outputSize.width.rounded()) & ~1)
        let outputHeight = max(2, Int(config.outputSize.height.rounded()) & ~1)
        guard inputWidth > 1, inputHeight > 1, outputWidth > 1, outputHeight > 1 else { return }

        let lumaWidth = max(2, (inputWidth + 3) / 4)
        let lumaHeight = max(2, (inputHeight + 3) / 4)
        let block = max(4, config.flowBlock)
        let coarseWidth = max(1, (lumaWidth + block - 1) / block)
        let coarseHeight = max(1, (lumaHeight + block - 1) / block)

        guard let zero = makeTexture(width: 1, height: 1, format: .rgba16Float,
                                     usage: [.shaderRead], storage: .shared) else {
            LogStore.error("render", "Could not allocate zero flow texture")
            return
        }
        let zeroBytes = [UInt16](repeating: 0, count: 4)
        zeroBytes.withUnsafeBytes { raw in
            if let base = raw.baseAddress {
                zero.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0,
                             withBytes: base, bytesPerRow: 8)
            }
        }
        zeroFlow = zero

        var newFlowSets: [FlowSet] = []
        for _ in 0..<2 {
            guard let lumaPrev = makeTexture(width: lumaWidth, height: lumaHeight, format: .r16Float,
                                             usage: [.shaderRead, .shaderWrite]),
                  let lumaCur = makeTexture(width: lumaWidth, height: lumaHeight, format: .r16Float,
                                            usage: [.shaderRead, .shaderWrite]),
                  let coarse = makeTexture(width: coarseWidth, height: coarseHeight, format: .rgba16Float,
                                           usage: [.shaderRead, .shaderWrite]),
                  let flowA = makeTexture(width: lumaWidth, height: lumaHeight, format: .rgba16Float,
                                          usage: [.shaderRead, .shaderWrite]),
                  let flowB = makeTexture(width: lumaWidth, height: lumaHeight, format: .rgba16Float,
                                          usage: [.shaderRead, .shaderWrite], storage: .shared) else {
                LogStore.error("render", "Could not allocate flow textures")
                return
            }
            newFlowSets.append(FlowSet(lumaPrevious: lumaPrev, lumaCurrent: lumaCur, coarse: coarse,
                                       flowA: flowA, flowB: flowB))
        }
        flowSets = newFlowSets

        var newTargets: [FrameTargets] = []
        for _ in 0..<2 {
            guard let warped = makeTexture(width: inputWidth, height: inputHeight, format: .bgra8Unorm,
                                           usage: [.shaderRead, .shaderWrite, .renderTarget]),
                  let scaled = makeTexture(width: outputWidth, height: outputHeight, format: .bgra8Unorm,
                                           usage: [.shaderRead, .renderTarget]),
                  let uniforms = device.makeBuffer(length: 4096, options: .storageModeShared) else {
                LogStore.error("render", "Could not allocate presentation textures")
                return
            }
            newTargets.append(FrameTargets(warped: warped, scaled: scaled, uniforms: uniforms))
        }
        targets = newTargets
        flowUniformBuffers = (0..<2).compactMap { _ in device.makeBuffer(length: 2048, options: .storageModeShared) }

        var newMLFlows: [MTLTexture] = []
        for _ in 0..<2 {
            if let texture = makeTexture(width: CoreMLFlowRefiner.width,
                                        height: CoreMLFlowRefiner.height,
                                        format: .rgba16Float,
                                        usage: [.shaderRead],
                                        storage: .shared) {
                newMLFlows.append(texture)
            }
        }
        mlFlows = newMLFlows
        mlFlowIndex = 0
        mlReadyTexture = nil
        mlReadyID = 0
        mlGeneration &+= 1

        let newScalerInput = CGSize(width: inputWidth, height: inputHeight)
        let newScalerOutput = CGSize(width: outputWidth, height: outputHeight)
        if newScalerInput != scalerInputSize || newScalerOutput != scalerOutputSize || scaler == nil {
            scaler = nil
            scalerInputSize = newScalerInput
            scalerOutputSize = newScalerOutput
            ensureScaler()
        }

        previous = nil
        current = nil
        lastSequence = 0
        lastRealSlotHostTime = 0
        lastPresentHost = 0
        generationCredit = 0
        sourceInterval.reset()
        sourceFPS = 0
        flowInFlight = false
        flowPreviousSlot = nil
        flowCurrentSlot = nil
        updateResidency()
        LogStore.info("render", "Textures: input \(inputWidth)x\(inputHeight), flow \(lumaWidth)x\(lumaHeight), output \(outputWidth)x\(outputHeight), scaler \(scaler != nil)")
    }

    private func makeTexture(width: Int,
                             height: Int,
                             format: MTLPixelFormat,
                             usage: MTLTextureUsage,
                             storage: MTLStorageMode = .private) -> MTLTexture? {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: format,
            width: max(1, width),
            height: max(1, height),
            mipmapped: false
        )
        descriptor.usage = usage
        descriptor.storageMode = storage
        return device.makeTexture(descriptor: descriptor)
    }

    private func ensureScaler() {
        let inputWidth = Int(scalerInputSize.width)
        let inputHeight = Int(scalerInputSize.height)
        let outputWidth = Int(scalerOutputSize.width)
        let outputHeight = Int(scalerOutputSize.height)
        guard inputWidth > 1, inputHeight > 1, outputWidth > 1, outputHeight > 1 else { return }
        if inputWidth == outputWidth && inputHeight == outputHeight {
            scaler = nil
            return
        }
        let descriptor = MTLFXSpatialScalerDescriptor()
        descriptor.inputWidth = inputWidth
        descriptor.inputHeight = inputHeight
        descriptor.outputWidth = outputWidth
        descriptor.outputHeight = outputHeight
        descriptor.colorTextureFormat = .bgra8Unorm
        descriptor.outputTextureFormat = .bgra8Unorm
        descriptor.colorProcessingMode = .perceptual
        scaler = descriptor.makeSpatialScaler(device: device, compiler: compiler)
        if scaler == nil {
            LogStore.warn("render", "MetalFX spatial scaler unavailable for \(inputWidth)x\(inputHeight) -> \(outputWidth)x\(outputHeight)")
        }
    }

    private func updateResidency() {
        guard let set = residencySet else { return }
        for target in targets {
            set.addAllocation(target.warped)
            set.addAllocation(target.scaled)
            set.addAllocation(target.uniforms)
        }
        for flow in flowSets {
            set.addAllocation(flow.lumaPrevious)
            set.addAllocation(flow.lumaCurrent)
            set.addAllocation(flow.coarse)
            set.addAllocation(flow.flowA)
            set.addAllocation(flow.flowB)
        }
        if let zeroFlow { set.addAllocation(zeroFlow) }
        for texture in mlFlows { set.addAllocation(texture) }
        for buffer in flowUniformBuffers { set.addAllocation(buffer) }
        set.commit()
    }

    // MARK: - Draw

    func draw(update: CAMetalDisplayLink.Update) {
        guard !isPaused else { return }
        let drawable = update.drawable
        let now = hostNow()
        let targetTime = update.targetPresentationTimestamp

        let drawableSize = CGSize(width: drawable.texture.width, height: drawable.texture.height)
        if Int(drawableSize.width) != Int(config.outputSize.width) ||
            Int(drawableSize.height) != Int(config.outputSize.height) {
            config.outputSize = drawableSize
            rebuild()
        }

        if let frame = mailbox.latest(after: lastSequence) {
            lastSequence = frame.sequence
            if let slot = makeSlot(frame) {
                if Int(slot.texture.width) != Int(config.inputSize.width) ||
                    Int(slot.texture.height) != Int(config.inputSize.height) {
                    config.inputSize = CGSize(width: slot.texture.width, height: slot.texture.height)
                    rebuild()
                }
                handleReal(slot: slot, drawable: drawable, targetTime: targetTime, now: now)
                metrics.tick()
                return
            }
        }

        if let current, shouldGenerate(now: now) {
            if let flow = readyFlow(for: current.pairID) {
                let t = generatedT(targetTime: targetTime, now: now)
                present(slot: current, flowTexture: flow, t: t, kind: .generated,
                        interpolation: config.interpolation, drawable: drawable,
                        targetTime: targetTime)
                generationCredit = max(0, generationCredit - 1)
                metrics.tick()
                return
            }
        }

        if let current, let zeroFlow {
            // Repeat the newest real frame if no flow/no generated frame is due.
            // This keeps one drawable presented per display-link callback, which
            // CAMetalDisplayLink requires, without ever queuing stale frames.
            present(slot: current, flowTexture: zeroFlow, t: 0, kind: .duplicate,
                    interpolation: false, drawable: drawable, targetTime: targetTime)
        } else {
            presentBlack(drawable: drawable, targetTime: targetTime)
        }
        metrics.tick()
    }

    private func handleReal(slot: FrameSlot,
                            drawable: CAMetalDrawable,
                            targetTime: CFTimeInterval,
                            now: CFTimeInterval) {
        if lastRealSlotHostTime > 0 {
            let delta = slot.hostTime - lastRealSlotHostTime
            if delta > 0.004 && delta < 0.5 {
                _ = sourceInterval.update(delta)
                if let interval = sourceInterval.value, interval > 0 {
                    sourceFPS = 1.0 / interval
                }
            }
        }
        lastRealSlotHostTime = slot.hostTime

        if let previousReal = current {
            previous = previousReal
            if !flowInFlight {
                startFlow(previous: previousReal, current: slot)
            }
        } else {
            previous = nil
        }
        current = slot

        if config.frameGeneration {
            let target = Double(config.targetFPS)
            let source = max(1.0, sourceFPS)
            if target > source + 0.5 {
                generationCredit = min(3.0, generationCredit + (target - source) / source)
            } else {
                generationCredit = 0
            }
        } else {
            generationCredit = 0
        }

        metrics.noteSourceFrame(at: slot.hostTime)
        present(slot: slot, flowTexture: zeroFlow ?? slot.texture, t: 0, kind: .real,
                interpolation: false, drawable: drawable, targetTime: targetTime)
    }

    private func makeSlot(_ frame: CapturedFrame) -> FrameSlot? {
        guard let cache = textureCache else { return nil }
        var cvTexture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            cache,
            frame.pixelBuffer,
            nil,
            .bgra8Unorm,
            frame.pixelWidth,
            frame.pixelHeight,
            0,
            &cvTexture
        )
        guard status == kCVReturnSuccess,
              let cvTexture,
              let texture = CVMetalTextureGetTexture(cvTexture) else {
            LogStore.warn("render", "CVMetalTexture creation failed (\(status))")
            return nil
        }
        return FrameSlot(pairID: frame.sequence, texture: texture, cvTexture: cvTexture,
                         pixelBuffer: frame.pixelBuffer, hostTime: frame.hostTime)
    }

    private func shouldGenerate(now: CFTimeInterval) -> Bool {
        guard config.frameGeneration, let current, generationCredit >= 1.0 else { return false }
        let interval = sourceInterval.value ?? (1.0 / max(1.0, sourceFPS))
        let age = now - current.hostTime
        return age < max(0.08, interval * 2.5)
    }

    private func generatedT(targetTime: CFTimeInterval, now: CFTimeInterval) -> Float {
        if config.interpolation { return 0.5 }
        let interval = sourceInterval.value ?? (1.0 / 30.0)
        let reference = targetTime > 0 ? targetTime : now
        let anchor = current?.hostTime ?? now
        let t = (reference - anchor) / max(interval, 0.004)
        return Float(clamp(t, 0.08, 1.0))
    }

    private func readyFlow(for pairID: UInt64) -> MTLTexture? {
        guard pairID != 0 else { return nil }
        if config.mode == .quality, mlReadyID == pairID, let mlReadyTexture {
            return mlReadyTexture
        }
        for set in flowSets where set.ready && set.pairID == pairID {
            return set.flowB
        }
        return nil
    }

    // MARK: - Flow

    private func startFlow(previous previousSlot: FrameSlot, current currentSlot: FrameSlot) {
        guard !flowSets.isEmpty, !flowUniformBuffers.isEmpty, !flowInFlight else { return }
        let set = flowSets[nextFlowSet]
        nextFlowSet = (nextFlowSet + 1) % flowSets.count
        set.pairID = currentSlot.pairID
        set.ready = false
        flowInFlight = true
        flowPreviousSlot = previousSlot
        flowCurrentSlot = currentSlot
        flowUniformBuffer = flowUniformBuffers[flowUniformIndex]
        flowUniformIndex = (flowUniformIndex + 1) % flowUniformBuffers.count
        flowUniformOffset = 0

        let lumaWidth = set.lumaCurrent.width
        let lumaHeight = set.lumaCurrent.height

        var lumaUniforms = RenderUniforms()
        lumaUniforms.p0.z = 4
        let lumaPrevAddress = appendUniforms(lumaUniforms, offset: &flowUniformOffset)
        let lumaCurAddress = appendUniforms(lumaUniforms, offset: &flowUniformOffset)

        var blockUniforms = RenderUniforms()
        blockUniforms.p3.z = Float(config.flowBlock)
        blockUniforms.p3.w = Float(config.flowBlock == 8 ? 6 : 10)
        let blockAddress = appendUniforms(blockUniforms, offset: &flowUniformOffset)

        let refineUniforms = RenderUniforms()
        let refineAAddress = appendUniforms(refineUniforms, offset: &flowUniformOffset)
        let refineBAddress = appendUniforms(refineUniforms, offset: &flowUniformOffset)

        guard let lumaPrevTable = makeTable(address: lumaPrevAddress,
                                            textures: [(previousSlot.texture, 0), (set.lumaPrevious, 1)]),
              let lumaCurTable = makeTable(address: lumaCurAddress,
                                           textures: [(currentSlot.texture, 0), (set.lumaCurrent, 1)]),
              let blockTable = makeTable(address: blockAddress,
                                         textures: [(set.lumaPrevious, 0), (set.lumaCurrent, 1), (set.coarse, 2)]),
              let refineATable = makeTable(address: refineAAddress,
                                           textures: [(set.lumaPrevious, 0), (set.lumaCurrent, 1), (set.coarse, 2), (set.flowA, 3)]),
              let refineBTable = makeTable(address: refineBAddress,
                                           textures: [(set.lumaPrevious, 0), (set.lumaCurrent, 1), (set.flowA, 2), (set.flowB, 3)]),
              let allocator = flowAllocators.acquire(),
              let commandBuffer = device.makeCommandBuffer() else {
            flowInFlight = false
            flowPreviousSlot = nil
            flowCurrentSlot = nil
            return
        }

        commandBuffer.beginCommandBuffer(allocator: allocator)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            commandBuffer.endCommandBuffer()
            flowAllocators.recycle(allocator)
            flowInFlight = false
            flowPreviousSlot = nil
            flowCurrentSlot = nil
            return
        }
        encoder.setComputePipelineState(lumaPipeline)
        encoder.setArgumentTable(lumaPrevTable)
        encoder.dispatchThreads(threadsPerGrid: MTLSize(width: lumaWidth, height: lumaHeight, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        encoder.setArgumentTable(lumaCurTable)
        encoder.dispatchThreads(threadsPerGrid: MTLSize(width: lumaWidth, height: lumaHeight, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))

        encoder.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch,
                        visibilityOptions: .device)

        encoder.setComputePipelineState(blockPipeline)
        encoder.setArgumentTable(blockTable)
        encoder.dispatchThreads(threadsPerGrid: MTLSize(width: set.coarse.width, height: set.coarse.height, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))

        encoder.barrier(afterEncoderStages: .dispatch, beforeEncoderStages: .dispatch,
                        visibilityOptions: .device)

        encoder.setComputePipelineState(refinePipeline)
        encoder.setArgumentTable(refineATable)
        encoder.dispatchThreads(threadsPerGrid: MTLSize(width: lumaWidth, height: lumaHeight, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        encoder.setArgumentTable(refineBTable)
        encoder.dispatchThreads(threadsPerGrid: MTLSize(width: lumaWidth, height: lumaHeight, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        encoder.endEncoding()
        commandBuffer.endCommandBuffer()

        let options = MTL4CommitOptions()
        options.addFeedbackHandler { [weak self] feedback in
            guard let self else { return }
            self.flowAllocators.recycle(allocator)
            let gpuMs = max(0, (feedback.gpuEndTime - feedback.gpuStartTime) * 1000.0)
            if let error = feedback.error {
                LogStore.error("flow", "Flow command buffer error: \(error)")
            }
            DispatchQueue.main.async {
                self.flowInFlight = false
                set.ready = true
                self.metrics.noteFlowTime(ms: gpuMs)
                if self.config.mode == .quality,
                   let previous = self.flowPreviousSlot,
                   let current = self.flowCurrentSlot,
                   current.pairID == set.pairID {
                    self.runInference(pairID: set.pairID, previous: previous, current: current, flow: set.flowB)
                } else {
                    self.flowPreviousSlot = nil
                    self.flowCurrentSlot = nil
                }
            }
        }
        flowQueue.commit([commandBuffer], options: options)
    }

    private func runInference(pairID: UInt64,
                              previous: FrameSlot,
                              current: FrameSlot,
                              flow: MTLTexture) {
        guard !mlFlows.isEmpty, config.inferenceRoute != .gpuOnly else {
            flowPreviousSlot = nil
            flowCurrentSlot = nil
            return
        }
        let output = mlFlows[mlFlowIndex]
        mlFlowIndex = (mlFlowIndex + 1) % mlFlows.count
        let generation = mlGeneration

        let completion: (UInt64) -> Void = { [weak self] id in
            guard let self else { return }
            guard self.mlGeneration == generation else {
                self.flowPreviousSlot = nil
                self.flowCurrentSlot = nil
                return
            }
            self.mlReadyID = id
            self.mlReadyTexture = output
            self.metrics.noteInferenceTime(ms: self.activeRefinerMilliseconds)
            self.flowPreviousSlot = nil
            self.flowCurrentSlot = nil
        }

        switch config.inferenceRoute {
        case .gpuOnly:
            flowPreviousSlot = nil
            flowCurrentSlot = nil
        case .coreML:
            if coreML == nil { coreML = CoreMLFlowRefiner() }
            guard let coreML else {
                flowPreviousSlot = nil
                flowCurrentSlot = nil
                return
            }
            coreML.refine(pairID: pairID, previous: previous.pixelBuffer, current: current.pixelBuffer,
                          flow: flow, output: output, completion: completion)
        case .metalML:
            if let metalML, metalML.prepare() {
                metalML.refine(pairID: pairID, previous: previous.pixelBuffer, current: current.pixelBuffer,
                               flow: flow, output: output, completion: completion)
            } else {
                if coreML == nil { coreML = CoreMLFlowRefiner() }
                guard let coreML else {
                    flowPreviousSlot = nil
                    flowCurrentSlot = nil
                    return
                }
                coreML.refine(pairID: pairID, previous: previous.pixelBuffer, current: current.pixelBuffer,
                              flow: flow, output: output, completion: completion)
            }
        }
    }

    private var activeRefinerMilliseconds: Double {
        switch config.inferenceRoute {
        case .coreML: return coreML?.inferenceMilliseconds ?? 0
        case .metalML: return metalML?.inferenceMilliseconds ?? 0
        case .gpuOnly: return 0
        }
    }

    // MARK: - Presentation

    private func present(slot: FrameSlot,
                         flowTexture: MTLTexture,
                         t: Float,
                         kind: PresentedKind,
                         interpolation: Bool,
                         drawable: CAMetalDrawable,
                         targetTime: CFTimeInterval) {
        guard !targets.isEmpty else { return }
        let target = targets[targetIndex]
        targetIndex = (targetIndex + 1) % targets.count

        var uniforms = RenderUniforms()
        uniforms.p0 = SIMD4<Float>(t, 1, 0, config.sharpness)
        uniforms.p1 = SIMD4<Float>(Float(slot.texture.width), Float(slot.texture.height),
                                   Float(drawable.texture.width), Float(drawable.texture.height))
        uniforms.p2 = SIMD4<Float>(Float(slot.texture.width) / Float(max(1, slot.texture.height)), 1, 0, 0)
        uniforms.p3 = SIMD4<Float>(interpolation ? 1 : 0, 0.2, 0, 0)
        var offset = 0
        let warpAddress = appendUniforms(uniforms, buffer: target.uniforms, offset: &offset)

        let needsWarp = t > 0.0001 || interpolation
        var warpTable: MTL4ArgumentTable?
        if needsWarp {
            let previousTexture = interpolation ? (previous?.texture ?? slot.texture) : slot.texture
            warpTable = makeTable(address: warpAddress,
                                  textures: [(slot.texture, 0), (previousTexture, 1),
                                             (flowTexture, 2), (target.warped, 3)])
            if warpTable == nil { return }
        }

        var scalerInput = needsWarp ? target.warped : slot.texture
        if scaler != nil {
            scalerInput = target.scaled
        }

        var casUniforms = RenderUniforms()
        casUniforms.p0 = SIMD4<Float>(0, 0, 0, config.sharpness)
        casUniforms.p1 = SIMD4<Float>(Float(scalerInput.width), Float(scalerInput.height),
                                      Float(drawable.texture.width), Float(drawable.texture.height))
        casUniforms.p2 = SIMD4<Float>(Float(slot.texture.width) / Float(max(1, slot.texture.height)), 1, 0, 0)
        let casAddress = appendUniforms(casUniforms, buffer: target.uniforms, offset: &offset)
        guard let casTable = makeTable(address: casAddress, textures: [(scalerInput, 0)]),
              let allocator = presentAllocators.acquire(),
              let commandBuffer = device.makeCommandBuffer() else {
            return
        }

        commandBuffer.beginCommandBuffer(allocator: allocator)

        if needsWarp, let warpTable {
            guard let warpEncoder = commandBuffer.makeComputeCommandEncoder() else {
                commandBuffer.endCommandBuffer()
                presentAllocators.recycle(allocator)
                return
            }
            warpEncoder.setComputePipelineState(warpPipeline)
            warpEncoder.setArgumentTable(warpTable)
            warpEncoder.dispatchThreads(threadsPerGrid: MTLSize(width: target.warped.width, height: target.warped.height, depth: 1),
                                        threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            warpEncoder.barrier(afterStages: .dispatch, beforeQueueStages: .all,
                                visibilityOptions: .device)
            warpEncoder.endEncoding()
        }

        if let scaler, scalerInputSize != scalerOutputSize {
            scaler.colorTexture = needsWarp ? target.warped : slot.texture
            scaler.outputTexture = target.scaled
            scaler.inputContentWidth = needsWarp ? target.warped.width : slot.texture.width
            scaler.inputContentHeight = needsWarp ? target.warped.height : slot.texture.height
            scaler.encode(commandBuffer: commandBuffer)
        }

        guard let renderEncoder = makeCASEncoder(commandBuffer, table: casTable, drawable: drawable) else {
            commandBuffer.endCommandBuffer()
            presentAllocators.recycle(allocator)
            return
        }
        renderEncoder.endEncoding()
        commandBuffer.endCommandBuffer()

        let options = MTL4CommitOptions()
        options.addFeedbackHandler { [weak self] feedback in
            self?.presentAllocators.recycle(allocator)
            let ms = max(0, (feedback.gpuEndTime - feedback.gpuStartTime) * 1000.0)
            if let error = feedback.error {
                LogStore.error("render", "Present command buffer error: \(error)")
            }
            self?.metrics.noteGPUTime(ms: ms)
        }

        let hostTime = slot.hostTime
        drawable.addPresentedHandler { [weak self] presentedDrawable in
            let latency = presentedDrawable.presentedTime - hostTime
            self?.metrics.notePresented(kind: kind, latency: latency)
        }

        presentQueue.commit([commandBuffer], options: options)
        presentQueue.signalDrawable(drawable)
        let now = hostNow()
        if targetTime > now {
            drawable.present(at: targetTime)
        } else {
            drawable.present()
        }
        if lastPresentHost > 0 {
            metrics.noteFrameTime(ms: (now - lastPresentHost) * 1000.0)
        }
        lastPresentHost = now
    }

    private func presentBlack(drawable: CAMetalDrawable, targetTime: CFTimeInterval) {
        guard let allocator = presentAllocators.acquire(),
              let commandBuffer = device.makeCommandBuffer() else { return }
        commandBuffer.beginCommandBuffer(allocator: allocator)
        let pass = MTL4RenderPassDescriptor()
        pass.colorAttachments[0]?.texture = drawable.texture
        pass.colorAttachments[0]?.loadAction = .clear
        pass.colorAttachments[0]?.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        pass.colorAttachments[0]?.storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
            commandBuffer.endCommandBuffer()
            presentAllocators.recycle(allocator)
            return
        }
        encoder.endEncoding()
        commandBuffer.endCommandBuffer()

        let options = MTL4CommitOptions()
        options.addFeedbackHandler { [weak self] _ in
            self?.presentAllocators.recycle(allocator)
        }
        presentQueue.commit([commandBuffer], options: options)
        presentQueue.signalDrawable(drawable)
        if targetTime > hostNow() {
            drawable.present(at: targetTime)
        } else {
            drawable.present()
        }
    }

    private func makeCASEncoder(_ commandBuffer: MTL4CommandBuffer,
                                table: MTL4ArgumentTable,
                                drawable: CAMetalDrawable) -> MTL4RenderCommandEncoder? {
        let pass = MTL4RenderPassDescriptor()
        pass.colorAttachments[0]?.texture = drawable.texture
        pass.colorAttachments[0]?.loadAction = .clear
        pass.colorAttachments[0]?.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        pass.colorAttachments[0]?.storeAction = .store
        guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else {
            return nil
        }
        encoder.setRenderPipelineState(casPipeline)
        encoder.setArgumentTable(table, stages: .fragment)
        encoder.barrier(afterQueueStages: .all, beforeStages: .fragment, visibilityOptions: .device)
        encoder.drawPrimitives(primitiveType: .triangle, vertexStart: 0, vertexCount: 3)
        return encoder
    }

    // MARK: - Helpers

    private func appendUniforms(_ uniforms: RenderUniforms,
                                buffer: MTLBuffer,
                                offset: inout Int) -> UInt64 {
        var value = uniforms
        let stride = MemoryLayout<RenderUniforms>.stride
        let address = buffer.gpuAddress + UInt64(offset)
        withUnsafeBytes(of: &value) { raw in
            guard let base = raw.baseAddress else { return }
            buffer.contents().advanced(by: offset).copyMemory(from: base, byteCount: stride)
        }
        offset += 256
        return address
    }

    private func appendUniforms(_ uniforms: RenderUniforms, offset: inout Int) -> UInt64 {
        guard let buffer = flowUniformBuffer else { return 0 }
        return appendUniforms(uniforms, buffer: buffer, offset: &offset)
    }

    private func makeTable(address: UInt64, textures: [(MTLTexture, Int)]) -> MTL4ArgumentTable? {
        let descriptor = MTL4ArgumentTableDescriptor()
        descriptor.maxBufferBindCount = 4
        descriptor.maxTextureBindCount = 8
        descriptor.maxSamplerStateBindCount = 1
        descriptor.initializeBindings = true
        guard let table = try? device.makeArgumentTable(descriptor: descriptor) else { return nil }
        table.setAddress(address, index: 0)
        for (texture, index) in textures {
            table.setTexture(texture.gpuResourceID, index: index)
        }
        return table
    }
}
