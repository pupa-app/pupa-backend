import CoreMedia
import CoreVideo
import Foundation
@preconcurrency import WebRTC

protocol WebRTCPublisherDelegate: AnyObject, Sendable {
    func webRTCPublisher(_ publisher: WebRTCPublisher, didProduceOffer sdp: String)
    func webRTCPublisher(_ publisher: WebRTCPublisher, didDiscoverLocalCandidate candidate: WebRTCCandidatePayload)
    func webRTCPublisher(_ publisher: WebRTCPublisher, didChangeConnectionState state: RTCPeerConnectionState)
}

// Wire format mirrors the browser RTCIceCandidate dict shape: `candidate`,
// `sdpMid`, `sdpMLineIndex`. Swift's RTCIceCandidate uses `.sdp` for the same
// field; the bridge happens here.
struct WebRTCCandidatePayload: Sendable, Codable, Equatable {
    let candidate: String
    let sdpMid: String?
    let sdpMLineIndex: Int32
}

// Publishes the captured window as a sendOnly H.264 video track on an
// RTCPeerConnection. Frame ingestion is the hot path (30 fps off the SCStream
// queue) — everything else (SDP negotiation, ICE relay) is one-shot signalling
// dispatched to the SignalingClient via the delegate.
final class WebRTCPublisher: NSObject, @unchecked Sendable {
    weak var delegate: (any WebRTCPublisherDelegate)?

    private let factory: RTCPeerConnectionFactory
    private let peerConnection: RTCPeerConnection
    private let videoSource: RTCVideoSource
    private let videoCapturer: RTCVideoCapturer
    private let videoTrack: RTCVideoTrack

    private let stateLock = NSLock()
    private var pendingRemoteCandidates: [WebRTCCandidatePayload] = []
    private var remoteDescriptionSet = false
    private var ingestCounter = 0

    override init() {
        RTCInitializeSSL()
        let encoderFactory = RTCDefaultVideoEncoderFactory()
        let decoderFactory = RTCDefaultVideoDecoderFactory()
        self.factory = RTCPeerConnectionFactory(encoderFactory: encoderFactory, decoderFactory: decoderFactory)

        let config = RTCConfiguration()
        config.sdpSemantics = .unifiedPlan
        config.iceServers = [RTCIceServer(urlStrings: ["stun:stun.l.google.com:19302"])]
        let constraints = RTCMediaConstraints(mandatoryConstraints: nil, optionalConstraints: nil)
        guard let pc = factory.peerConnection(with: config, constraints: constraints, delegate: nil) else {
            fatalError("RTCPeerConnectionFactory.peerConnection returned nil")
        }
        self.peerConnection = pc

        self.videoSource = factory.videoSource()
        self.videoCapturer = RTCVideoCapturer(delegate: videoSource)
        self.videoTrack = factory.videoTrack(with: videoSource, trackId: "screen0")

        super.init()
        peerConnection.delegate = self

        let transceiverInit = RTCRtpTransceiverInit()
        transceiverInit.direction = .sendOnly
        transceiverInit.streamIds = ["pupa-screen"]
        peerConnection.addTransceiver(with: videoTrack, init: transceiverInit)
    }

    deinit {
        peerConnection.close()
        RTCCleanupSSL()
    }

    // MARK: - Frame ingestion (called from SCStream output queue)

    func ingest(sampleBuffer: CMSampleBuffer) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let timeStampNs = Int64(CMTimeGetSeconds(pts) * 1_000_000_000)
        let rtcBuffer = RTCCVPixelBuffer(pixelBuffer: pixelBuffer)
        let frame = RTCVideoFrame(buffer: rtcBuffer, rotation: ._0, timeStampNs: timeStampNs)
        videoSource.capturer(videoCapturer, didCapture: frame)
        let count = stateLock.withLock {
            self.ingestCounter += 1
            return self.ingestCounter
        }
        // Heartbeat — confirms frames are actually reaching the encoder.
        if count == 1 || count % 150 == 0 {
            FileHandle.standardError.log("ingested \(count) frames")
        }
    }

    // MARK: - Negotiation

    func createOffer() {
        let constraints = RTCMediaConstraints(mandatoryConstraints: nil, optionalConstraints: nil)
        peerConnection.offer(for: constraints) { [weak self] sdp, error in
            guard let self else { return }
            if let error {
                FileHandle.standardError.log("createOffer failed: \(error)")
                return
            }
            guard let sdp else { return }
            self.peerConnection.setLocalDescription(sdp) { [weak self] error in
                guard let self else { return }
                if let error {
                    FileHandle.standardError.log("setLocalDescription failed: \(error)")
                    return
                }
                self.delegate?.webRTCPublisher(self, didProduceOffer: sdp.sdp)
            }
        }
    }

    func acceptRemoteAnswer(sdp: String) {
        let desc = RTCSessionDescription(type: .answer, sdp: sdp)
        peerConnection.setRemoteDescription(desc) { [weak self] error in
            guard let self else { return }
            if let error {
                FileHandle.standardError.log("setRemoteDescription failed: \(error)")
                return
            }
            self.flushPendingRemoteCandidates()
        }
    }

    func addRemoteCandidate(_ payload: WebRTCCandidatePayload) {
        let alreadySet: Bool = stateLock.withLock {
            if !self.remoteDescriptionSet {
                self.pendingRemoteCandidates.append(payload)
                return false
            }
            return true
        }
        guard alreadySet else { return }
        applyRemoteCandidate(payload)
    }

    private func flushPendingRemoteCandidates() {
        let queued: [WebRTCCandidatePayload] = stateLock.withLock {
            self.remoteDescriptionSet = true
            let q = self.pendingRemoteCandidates
            self.pendingRemoteCandidates.removeAll()
            return q
        }
        for payload in queued {
            applyRemoteCandidate(payload)
        }
    }

    private func applyRemoteCandidate(_ payload: WebRTCCandidatePayload) {
        let candidate = RTCIceCandidate(
            sdp: payload.candidate,
            sdpMLineIndex: payload.sdpMLineIndex,
            sdpMid: payload.sdpMid
        )
        peerConnection.add(candidate) { error in
            if let error {
                FileHandle.standardError.log("addIceCandidate failed: \(error)")
            }
        }
    }

    func close() {
        peerConnection.close()
    }
}

extension WebRTCPublisher: RTCPeerConnectionDelegate {
    func peerConnection(_ peerConnection: RTCPeerConnection, didChange stateChanged: RTCSignalingState) {}
    func peerConnection(_ peerConnection: RTCPeerConnection, didAdd stream: RTCMediaStream) {}
    func peerConnection(_ peerConnection: RTCPeerConnection, didRemove stream: RTCMediaStream) {}
    func peerConnectionShouldNegotiate(_ peerConnection: RTCPeerConnection) {}
    func peerConnection(_ peerConnection: RTCPeerConnection, didChange newState: RTCIceConnectionState) {}
    func peerConnection(_ peerConnection: RTCPeerConnection, didChange newState: RTCIceGatheringState) {}
    func peerConnection(_ peerConnection: RTCPeerConnection, didRemove candidates: [RTCIceCandidate]) {}
    func peerConnection(_ peerConnection: RTCPeerConnection, didOpen dataChannel: RTCDataChannel) {}

    func peerConnection(_ peerConnection: RTCPeerConnection, didGenerate candidate: RTCIceCandidate) {
        let payload = WebRTCCandidatePayload(
            candidate: candidate.sdp,
            sdpMid: candidate.sdpMid,
            sdpMLineIndex: candidate.sdpMLineIndex
        )
        delegate?.webRTCPublisher(self, didDiscoverLocalCandidate: payload)
    }

    func peerConnection(_ peerConnection: RTCPeerConnection, didChange newState: RTCPeerConnectionState) {
        delegate?.webRTCPublisher(self, didChangeConnectionState: newState)
    }
}
