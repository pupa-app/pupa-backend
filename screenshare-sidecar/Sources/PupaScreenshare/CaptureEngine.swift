import CoreMedia
import CoreVideo
import Foundation
import ScreenCaptureKit

// SCContentFilter is an immutable ObjC reference type that ScreenCaptureKit
// hands us from the picker on the main thread; we then hand it to SCStream's
// init on the same thread. Swift 6 strict concurrency can't see that and
// flags the continuation resume as a data race. Retroactive Sendable
// conformance is the least intrusive fix — there are no mutating APIs on the
// type that could actually race.
extension SCContentFilter: @retroactive @unchecked Sendable {}

protocol CaptureEngineDelegate: AnyObject, Sendable {
    func captureEngine(_ engine: CaptureEngine, didProduce sampleBuffer: CMSampleBuffer, pixelSize: CGSize)
    func captureEngine(_ engine: CaptureEngine, didFailWith error: any Error)
}

// Captures a single user-picked window via ScreenCaptureKit and emits each
// `.complete` `CMSampleBuffer` to its delegate. The delegate is responsible
// for the destination — in Phase 2 that's WebRTCPublisher converting buffers
// into RTCVideoFrames, but the abstraction means we can plug in an
// AVAssetWriter sink later for debug capture without touching this file.
final class CaptureEngine: NSObject, @unchecked Sendable {
    weak var delegate: (any CaptureEngineDelegate)?

    private let processingQueue = DispatchQueue(label: "com.pupa.screenshare.processing")
    private let stateLock = NSLock()

    private var pickerContinuation: CheckedContinuation<SCContentFilter, Error>?
    private var stream: SCStream?
    private var pixelSize: CGSize = .zero

    func start() async throws {
        let filter = try await presentPickerAwaitingSelection()
        try await beginCapture(filter: filter)
    }

    /// Stop the current capture session and present the picker again so the
    /// user can choose a different window or application. Called when the
    /// viewer sends a `repick` message. On picker cancellation the previous
    /// stream is already stopped; the caller is responsible for error handling.
    func repick() async throws {
        stop()
        try await start()
    }

    func stop() {
        let streamToStop: SCStream? = stateLock.withLock {
            let s = self.stream
            self.stream = nil
            return s
        }
        guard let streamToStop else { return }
        let group = DispatchGroup()
        group.enter()
        streamToStop.stopCapture { _ in group.leave() }
        _ = group.wait(timeout: .now() + .seconds(2))
    }
}

// MARK: - Picker

extension CaptureEngine: SCContentSharingPickerObserver {
    private func presentPickerAwaitingSelection() async throws -> SCContentFilter {
        try await withCheckedThrowingContinuation { continuation in
            self.pickerContinuation = continuation
            let picker = SCContentSharingPicker.shared
            picker.add(self)
            picker.isActive = true
            var config = SCContentSharingPickerConfiguration()
            config.allowedPickerModes = [.singleApplication, .singleWindow]
            picker.defaultConfiguration = config
            picker.present()
        }
    }

    func contentSharingPicker(_ picker: SCContentSharingPicker, didCancelFor stream: SCStream?) {
        cleanupPicker(picker)
        resumePicker(.failure(PickerError.cancelled))
    }

    func contentSharingPicker(_ picker: SCContentSharingPicker, didUpdateWith filter: SCContentFilter, for stream: SCStream?) {
        cleanupPicker(picker)
        resumePicker(.success(filter))
    }

    func contentSharingPickerStartDidFailWithError(_ error: any Error) {
        let picker = SCContentSharingPicker.shared
        cleanupPicker(picker)
        resumePicker(.failure(error))
    }

    private func cleanupPicker(_ picker: SCContentSharingPicker) {
        picker.isActive = false
        picker.remove(self)
    }

    private func resumePicker(_ result: Result<SCContentFilter, Error>) {
        let continuation = stateLock.withLock { () -> CheckedContinuation<SCContentFilter, Error>? in
            let c = self.pickerContinuation
            self.pickerContinuation = nil
            return c
        }
        guard let continuation else { return }
        continuation.resume(with: result)
    }

    enum PickerError: Error, CustomStringConvertible {
        case cancelled
        var description: String {
            switch self {
            case .cancelled: "user cancelled the picker"
            }
        }
    }
}

// MARK: - Capture

extension CaptureEngine {
    private func beginCapture(filter: SCContentFilter) async throws {
        let pixelWidth = max(2, Int(filter.contentRect.width * CGFloat(filter.pointPixelScale)))
        let pixelHeight = max(2, Int(filter.contentRect.height * CGFloat(filter.pointPixelScale)))

        let config = SCStreamConfiguration()
        config.width = pixelWidth
        config.height = pixelHeight
        config.minimumFrameInterval = CMTime(value: 1, timescale: 30)
        config.queueDepth = 5
        config.showsCursor = true
        config.pixelFormat = kCVPixelFormatType_32BGRA

        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: processingQueue)

        stateLock.withLock {
            self.stream = stream
            self.pixelSize = CGSize(width: pixelWidth, height: pixelHeight)
        }

        try await stream.startCapture()
        FileHandle.standardError.log("recording \(pixelWidth)x\(pixelHeight) @ 30 fps")
    }
}

// MARK: - SCStream callbacks

extension CaptureEngine: SCStreamDelegate {
    func stream(_ stream: SCStream, didStopWithError error: Error) {
        FileHandle.standardError.log("stream stopped with error: \(error)")
        delegate?.captureEngine(self, didFailWith: error)
    }
}

extension CaptureEngine: SCStreamOutput {
    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of outputType: SCStreamOutputType) {
        guard outputType == .screen else { return }
        guard sampleBuffer.isValid else { return }
        guard hasCompleteAttachment(sampleBuffer) else { return }

        let size = stateLock.withLock { self.pixelSize }
        delegate?.captureEngine(self, didProduce: sampleBuffer, pixelSize: size)
    }

    private func hasCompleteAttachment(_ sampleBuffer: CMSampleBuffer) -> Bool {
        // SCStream emits both `.complete` frames (which carry pixel data) and
        // status-only frames (e.g. `.idle` when the source hasn't moved). Only
        // the former should be forwarded.
        guard let raw = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
              let info = raw.first,
              let statusRaw = info[.status] as? Int,
              let status = SCFrameStatus(rawValue: statusRaw)
        else { return false }
        return status == .complete
    }
}
