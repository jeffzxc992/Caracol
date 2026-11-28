import Foundation
import AVFoundation
import CoreVideo
import QuartzCore

struct VideoTestInfo {
    var title: String
    var size: CGSize
    var fps: Double
    var duration: Double
}

/// Deterministic recorded-video test source. It decodes a local movie and
/// publishes frames into the same latest-wins mailbox as ScreenCaptureKit, so
/// the complete generation/upscale/present path is exercised without a game.
final class VideoTestSource {
    private let queue = DispatchQueue(label: "com.caracol.video-test", qos: .utility)
    private var reader: AVAssetReader?
    private var output: AVAssetReaderTrackOutput?
    private var timer: DispatchSourceTimer?
    private var track: AVAssetTrack?
    private var asset: AVURLAsset?
    private weak var mailbox: FrameMailbox?
    private var info: VideoTestInfo?
    private var isRunning = false

    var isActive: Bool {
        queue.sync { isRunning }
    }

    func start(url: URL,
               mailbox: FrameMailbox,
               completion: @escaping (Result<VideoTestInfo, Error>) -> Void) {
        stop()
        self.mailbox = mailbox
        let asset = AVURLAsset(url: url)
        self.asset = asset

        Task.detached(priority: .utility) {
            do {
                let tracks = try await asset.loadTracks(withMediaType: .video)
                guard let track = tracks.first else {
                    throw NSError(domain: "Caracol", code: 10,
                                  userInfo: [NSLocalizedDescriptionKey: "The movie has no video track."])
                }
                let natural = try await track.load(.naturalSize)
                let transform = try await track.load(.preferredTransform)
                let nominalFPS = try await track.load(.nominalFrameRate)
                let duration = try await asset.load(.duration)
                var size = natural.applying(transform)
                size = CGSize(width: abs(size.width), height: abs(size.height))
                if size.width < 2 || size.height < 2 { size = natural }
                let fps = nominalFPS > 1 ? Double(nominalFPS) : 30
                let info = VideoTestInfo(
                    title: url.lastPathComponent,
                    size: size,
                    fps: min(60, max(1, fps)),
                    duration: duration.seconds
                )
                await MainActor.run {
                    self.begin(info: info, track: track, completion: completion)
                }
            } catch {
                await MainActor.run {
                    completion(.failure(error))
                }
            }
        }
    }

    private func begin(info: VideoTestInfo,
                       track: AVAssetTrack,
                       completion: @escaping (Result<VideoTestInfo, Error>) -> Void) {
        self.track = track
        self.info = info
        queue.async { [weak self] in
            guard let self, let asset = self.asset else { return }
            do {
                try self.makeReader(asset: asset, track: track)
            } catch {
                DispatchQueue.main.async { completion(.failure(error)) }
                return
            }
            self.isRunning = true
            let interval = 1.0 / info.fps
            let timer = DispatchSource.makeTimerSource(queue: self.queue)
            timer.schedule(deadline: .now(), repeating: interval, leeway: .milliseconds(1))
            timer.setEventHandler { [weak self] in
                self?.readNextFrame()
            }
            self.timer = timer
            timer.resume()
            DispatchQueue.main.async {
                LogStore.info("video", "Test source started: \(info.title) \(Int(info.size.width))x\(Int(info.size.height)) @ \(String(format: "%.1f", info.fps)) fps")
                completion(.success(info))
            }
        }
    }

    private func makeReader(asset: AVURLAsset, track: AVAssetTrack) throws {
        reader?.cancelReading()
        let reader = try AVAssetReader(asset: asset)
        let settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferMetalCompatibilityKey as String: true
        ]
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: settings)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            throw NSError(domain: "Caracol", code: 11,
                          userInfo: [NSLocalizedDescriptionKey: "Could not add the movie reader output."])
        }
        reader.add(output)
        guard reader.startReading() else {
            throw reader.error ?? NSError(domain: "Caracol", code: 12,
                                          userInfo: [NSLocalizedDescriptionKey: "Could not start the movie reader."])
        }
        self.reader = reader
        self.output = output
    }

    private func readNextFrame() {
        guard isRunning, let reader, let output, let mailbox, let info else { return }
        if reader.status == .failed || reader.status == .cancelled {
            restartReader()
            return
        }
        guard let sample = output.copyNextSampleBuffer() else {
            restartReader()
            return
        }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sample) else { return }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        if DebugTrace.enabled {
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            let base = CVPixelBufferGetBaseAddress(pixelBuffer)?.assumingMemoryBound(to: UInt8.self)
            let row = CVPixelBufferGetBytesPerRow(pixelBuffer)
            let probe = base.map { $0.advanced(by: row * (height / 2) + (width / 2) * 4) }
            DebugTrace.log("video-probe", "\(width)x\(height) bpr \(row) center \(probe.map { "\($0[0]),\($0[1]),\($0[2]),\($0[3])" } ?? "no-base")")
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        }
        let frame = CapturedFrame(
            sequence: 0,
            pixelBuffer: pixelBuffer,
            hostTime: hostNow(),
            contentRect: CGRect(origin: .zero, size: info.size),
            contentScale: 1,
            sourceSize: CGSize(width: width, height: height),
            title: info.title,
            bundleID: "com.caracol.video-test",
            windowID: nil,
            isDisplay: false
        )
        mailbox.put(frame)
        withExtendedLifetime(sample) {}
    }

    private func restartReader() {
        guard isRunning, let asset, let track else { return }
        do {
            try makeReader(asset: asset, track: track)
        } catch {
            LogStore.error("video", "Movie loop failed: \(error)")
            stop()
        }
    }

    func stop() {
        let wasRunning = queue.sync { isRunning }
        queue.sync {
            isRunning = false
            timer?.cancel()
            timer = nil
            reader?.cancelReading()
            reader = nil
            output = nil
        }
        if wasRunning {
            LogStore.info("video", "Test source stopped")
        }
    }
}
