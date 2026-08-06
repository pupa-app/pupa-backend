import AppKit
import Foundation

// CLI sidecar entry point. Runs an accessory NSApplication so SCContentSharingPicker
// has a host run loop; the picker is presented system-wide but still requires
// NSApp to be alive. On SIGINT, terminates cleanly so the WebRTC peer
// connection sends a `bye` and the broker tears down the viewer.

struct SidecarArgs {
    var brokerURL: URL
    var shareID: String
    var apiKey: String?

    static func parse(_ argv: [String]) -> SidecarArgs {
        func value(for flag: String) -> String? {
            guard let i = argv.firstIndex(of: flag), i + 1 < argv.count else { return nil }
            return argv[i + 1]
        }
        let brokerString = value(for: "--broker") ?? "ws://localhost:8004/screenshare/ws"
        let shareID = value(for: "--share-id") ?? UUID().uuidString
        let apiKey = value(for: "--api-key")
        guard let url = URL(string: brokerString) else {
            FileHandle.standardError.log("invalid --broker URL: \(brokerString)")
            exit(EXIT_FAILURE)
        }
        return SidecarArgs(brokerURL: url, shareID: shareID, apiKey: apiKey)
    }
}

let args = SidecarArgs.parse(CommandLine.arguments)
let app = NSApplication.shared
let delegate = AppDelegate(args: args)
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
