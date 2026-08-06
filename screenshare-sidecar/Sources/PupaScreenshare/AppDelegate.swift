import AppKit
import CoreMedia
import Darwin
import Foundation
@preconcurrency import WebRTC

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private let args: SidecarArgs
    private let capture: CaptureEngine
    private let publisher: WebRTCPublisher
    private let signaling: SignalingClient
    private var sigintSource: DispatchSourceSignal?

    init(args: SidecarArgs) {
        self.args = args
        self.capture = CaptureEngine()
        self.publisher = WebRTCPublisher()
        self.signaling = SignalingClient()
        super.init()
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        FileHandle.standardError.log("pupa-screenshare \(PupaScreenshareVersion)")
        FileHandle.standardError.log("broker   : \(args.brokerURL.absoluteString)")
        FileHandle.standardError.log("share_id : \(args.shareID)")
        FileHandle.standardError.log("Pick a window in the system picker. Press Ctrl+C to stop.")

        capture.delegate = self
        publisher.delegate = self
        signaling.delegate = self
        signaling.connect(brokerURL: args.brokerURL, shareID: args.shareID, apiKey: args.apiKey)

        installSigintHandler()
        Task { @MainActor in
            do {
                try await capture.start()
            } catch {
                FileHandle.standardError.log("capture failed: \(error)")
                let reason = captureFailureReason(error)
                self.signaling.send(.error(code: 5001, reason: reason))
                // Give the WS a beat to flush before NSApp tears it down.
                try? await Task.sleep(nanoseconds: 100_000_000)
                NSApp.terminate(nil)
            }
        }
    }

    /// Human-readable explanation of why capture didn't start. Distinguishes
    /// the most common causes so the viewer can show something more useful
    /// than "publisher exited".
    private func captureFailureReason(_ error: Error) -> String {
        if let pickerError = error as? CaptureEngine.PickerError {
            switch pickerError {
            case .cancelled:
                return "publisher dismissed the window picker before sharing started"
            }
        }
        // SCStream.startCapture failures bubble up here. TCC-denied capture
        // surfaces as an NSError with the Screen Capture domain; the .code
        // values aren't documented stably so we just include the description.
        let nsError = error as NSError
        if nsError.domain.contains("ScreenCaptureKit") || nsError.domain.contains("ReplayKit") {
            return "publisher couldn't start screen capture (likely Screen Recording permission denied): \(nsError.localizedDescription)"
        }
        return "publisher failed to start capture: \(error)"
    }

    func applicationWillTerminate(_ notification: Notification) {
        signaling.send(.bye)
        capture.stop()
        publisher.close()
        signaling.close()
    }

    private func installSigintHandler() {
        signal(SIGINT, SIG_IGN)
        let source = DispatchSource.makeSignalSource(signal: SIGINT, queue: .main)
        source.setEventHandler {
            NSApp.terminate(nil)
        }
        source.resume()
        sigintSource = source
    }
}

extension AppDelegate: CaptureEngineDelegate {
    nonisolated func captureEngine(_ engine: CaptureEngine, didProduce sampleBuffer: CMSampleBuffer, pixelSize: CGSize) {
        publisher.ingest(sampleBuffer: sampleBuffer)
    }

    nonisolated func captureEngine(_ engine: CaptureEngine, didFailWith error: any Error) {
        Task { @MainActor in
            FileHandle.standardError.log("capture engine failed: \(error)")
            self.signaling.send(.error(code: 5002, reason: self.captureFailureReason(error)))
            try? await Task.sleep(nanoseconds: 100_000_000)
            NSApp.terminate(nil)
        }
    }
}

extension AppDelegate: WebRTCPublisherDelegate {
    nonisolated func webRTCPublisher(_ publisher: WebRTCPublisher, didProduceOffer sdp: String) {
        signaling.send(.offer(sdp: sdp))
    }

    nonisolated func webRTCPublisher(_ publisher: WebRTCPublisher, didDiscoverLocalCandidate candidate: WebRTCCandidatePayload) {
        signaling.send(.ice(candidate))
    }

    nonisolated func webRTCPublisher(_ publisher: WebRTCPublisher, didChangeConnectionState state: RTCPeerConnectionState) {
        Task { @MainActor in
            FileHandle.standardError.log("peer connection state: \(connectionStateDescription(state))")
            if state == .failed || state == .closed {
                // Don't terminate — viewer may reconnect on the same share_id.
            }
        }
    }
}

extension AppDelegate: SignalingClientDelegate {
    nonisolated func signalingClient(_ client: SignalingClient, didReceive message: SignalingMessage) {
        switch message {
        case .viewerJoined:
            publisher.createOffer()
        case .answer(let sdp):
            publisher.acceptRemoteAnswer(sdp: sdp)
        case .ice(let payload):
            publisher.addRemoteCandidate(payload)
        case .bye:
            Task { @MainActor in
                FileHandle.standardError.log("viewer left; waiting for next connect")
            }
        case .error(let code, let reason):
            // Broker rejected us — likely 4409 (another publisher already
            // owns this share_id). No retry; exit so `pupa-backend screenshare`
            // surfaces the reason and the user can pick a fresh share id.
            Task { @MainActor in
                FileHandle.standardError.log("broker error \(code): \(reason)")
                NSApp.terminate(nil)
            }
        case .repick:
            Task { @MainActor in
                FileHandle.standardError.log("viewer requested re-pick; re-opening source picker")
                do {
                    try await self.capture.repick()
                } catch {
                    FileHandle.standardError.log("re-pick failed: \(error)")
                    self.signaling.send(.error(code: 5003, reason: self.captureFailureReason(error)))
                    // Keep running — viewer can try again or disconnect cleanly.
                }
            }
        case .offer, .unknown:
            // Publishers shouldn't receive offers, and unknowns are no-op.
            break
        }
    }

    nonisolated func signalingClient(_ client: SignalingClient, didCloseWithCode code: URLSessionWebSocketTask.CloseCode) {
        Task { @MainActor in
            FileHandle.standardError.log("signalling closed (code=\(code.rawValue)); exiting")
            NSApp.terminate(nil)
        }
    }
}

private func connectionStateDescription(_ state: RTCPeerConnectionState) -> String {
    switch state {
    case .new: return "new"
    case .connecting: return "connecting"
    case .connected: return "connected"
    case .disconnected: return "disconnected"
    case .failed: return "failed"
    case .closed: return "closed"
    @unknown default: return "unknown(\(state.rawValue))"
    }
}

extension FileHandle {
    func log(_ message: String) {
        if let data = (message + "\n").data(using: .utf8) {
            write(data)
        }
    }
}
