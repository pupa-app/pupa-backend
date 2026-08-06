import Foundation

protocol SignalingClientDelegate: AnyObject, Sendable {
    func signalingClient(_ client: SignalingClient, didReceive message: SignalingMessage)
    func signalingClient(_ client: SignalingClient, didCloseWithCode code: URLSessionWebSocketTask.CloseCode)
}

// Wire shape mirrors the broker's relay format. Opaque to the broker — only
// the publisher and viewer interpret these.
enum SignalingMessage: Sendable {
    case viewerJoined
    case offer(sdp: String)
    case answer(sdp: String)
    case ice(WebRTCCandidatePayload)
    case bye
    /// Pre-close error from the broker (4404 / 4409) — diagnostic for the
    /// publisher so it can log + exit cleanly without retrying.
    case error(code: Int, reason: String)
    /// Viewer requests the publisher to re-open the source picker so the user
    /// can share a different window or application without reconnecting.
    case repick
    case unknown(String)

    init?(json: [String: Any]) {
        guard let type = json["type"] as? String else { return nil }
        switch type {
        case "viewer_joined": self = .viewerJoined
        case "bye": self = .bye
        case "repick": self = .repick
        case "offer":
            guard let sdp = json["sdp"] as? String else { return nil }
            self = .offer(sdp: sdp)
        case "answer":
            guard let sdp = json["sdp"] as? String else { return nil }
            self = .answer(sdp: sdp)
        case "ice":
            guard let inner = json["candidate"] as? [String: Any],
                  let candidate = inner["candidate"] as? String
            else { return nil }
            let sdpMid = inner["sdpMid"] as? String
            let sdpMLineIndex: Int32
            if let i = inner["sdpMLineIndex"] as? Int {
                sdpMLineIndex = Int32(i)
            } else if let n = inner["sdpMLineIndex"] as? NSNumber {
                sdpMLineIndex = n.int32Value
            } else {
                sdpMLineIndex = 0
            }
            self = .ice(WebRTCCandidatePayload(candidate: candidate, sdpMid: sdpMid, sdpMLineIndex: sdpMLineIndex))
        case "error":
            let code = (json["code"] as? Int) ?? (json["code"] as? NSNumber)?.intValue ?? 0
            let reason = (json["reason"] as? String) ?? "unspecified error"
            self = .error(code: code, reason: reason)
        default:
            self = .unknown(type)
        }
    }

    var json: [String: Any] {
        switch self {
        case .viewerJoined: return ["type": "viewer_joined"]
        case .bye: return ["type": "bye"]
        case .offer(let sdp): return ["type": "offer", "sdp": sdp]
        case .answer(let sdp): return ["type": "answer", "sdp": sdp]
        case .ice(let payload):
            var inner: [String: Any] = [
                "candidate": payload.candidate,
                "sdpMLineIndex": payload.sdpMLineIndex,
            ]
            if let mid = payload.sdpMid { inner["sdpMid"] = mid }
            return ["type": "ice", "candidate": inner]
        case .error(let code, let reason):
            return ["type": "error", "code": code, "reason": reason]
        case .repick: return ["type": "repick"]
        case .unknown(let type): return ["type": type]
        }
    }
}

final class SignalingClient: NSObject, @unchecked Sendable {
    weak var delegate: (any SignalingClientDelegate)?

    private var session: URLSession!
    private var task: URLSessionWebSocketTask?
    private let stateLock = NSLock()
    private var closed = false

    override init() {
        super.init()
        let config = URLSessionConfiguration.default
        self.session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }

    func connect(brokerURL: URL, shareID: String, apiKey: String?) {
        var components = URLComponents(url: brokerURL, resolvingAgainstBaseURL: false)
        components?.queryItems = [
            URLQueryItem(name: "role", value: "publisher"),
            URLQueryItem(name: "share_id", value: shareID),
        ]
        guard let finalURL = components?.url else {
            FileHandle.standardError.log("invalid broker URL: \(brokerURL)")
            return
        }
        var request = URLRequest(url: finalURL)
        if let apiKey, !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        let task = session.webSocketTask(with: request)
        stateLock.withLock { self.task = task }
        task.resume()
        receiveLoop(task)
    }

    func send(_ message: SignalingMessage) {
        guard let task = stateLock.withLock({ self.task }) else { return }
        guard let data = try? JSONSerialization.data(withJSONObject: message.json) else {
            FileHandle.standardError.log("failed to encode \(message.json)")
            return
        }
        guard let text = String(data: data, encoding: .utf8) else { return }
        task.send(.string(text)) { error in
            if let error {
                FileHandle.standardError.log("ws send failed: \(error)")
            }
        }
    }

    func close() {
        let task = stateLock.withLock { () -> URLSessionWebSocketTask? in
            guard !self.closed else { return nil }
            self.closed = true
            return self.task
        }
        task?.cancel(with: .goingAway, reason: nil)
    }

    private func receiveLoop(_ task: URLSessionWebSocketTask) {
        task.receive { [weak self, weak task] result in
            guard let self, let task else { return }
            switch result {
            case .failure(let error):
                FileHandle.standardError.log("ws receive failed: \(error)")
                self.delegate?.signalingClient(self, didCloseWithCode: .abnormalClosure)
            case .success(let message):
                self.handleIncoming(message)
                if !self.stateLock.withLock({ self.closed }) {
                    self.receiveLoop(task)
                }
            }
        }
    }

    private func handleIncoming(_ wsMessage: URLSessionWebSocketTask.Message) {
        let payload: Data?
        switch wsMessage {
        case .string(let text): payload = text.data(using: .utf8)
        case .data(let data): payload = data
        @unknown default: payload = nil
        }
        guard let payload,
              let json = (try? JSONSerialization.jsonObject(with: payload)) as? [String: Any],
              let message = SignalingMessage(json: json)
        else {
            FileHandle.standardError.log("dropping malformed signalling payload")
            return
        }
        delegate?.signalingClient(self, didReceive: message)
    }
}

extension SignalingClient: URLSessionWebSocketDelegate {
    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didOpenWithProtocol protocol: String?) {
        FileHandle.standardError.log("signalling connected")
    }

    func urlSession(_ session: URLSession, webSocketTask: URLSessionWebSocketTask, didCloseWith closeCode: URLSessionWebSocketTask.CloseCode, reason: Data?) {
        let reasonString = reason.flatMap { String(data: $0, encoding: .utf8) } ?? ""
        FileHandle.standardError.log("signalling closed code=\(closeCode.rawValue) reason=\(reasonString)")
        delegate?.signalingClient(self, didCloseWithCode: closeCode)
    }

    func urlSession(
        _ session: URLSession,
        didReceive challenge: URLAuthenticationChallenge,
        completionHandler: @escaping (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
    ) {
        // The sidecar connects only to localhost — accept the self-signed cert
        // generated by `make setup` without requiring a CA-signed certificate.
        guard challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust,
              challenge.protectionSpace.host == "localhost",
              let trust = challenge.protectionSpace.serverTrust
        else {
            completionHandler(.performDefaultHandling, nil)
            return
        }
        completionHandler(.useCredential, URLCredential(trust: trust))
    }
}
