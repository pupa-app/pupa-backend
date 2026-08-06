// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "PupaScreenshare",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        // CLI sidecar: lets the user pick a window via SCContentSharingPicker,
        // captures via ScreenCaptureKit, and publishes the video as an
        // outgoing WebRTC track to whichever viewer the broker has paired with
        // this publisher's share_id.
        .executable(name: "pupa-screenshare", targets: ["PupaScreenshare"]),
    ],
    dependencies: [
        // stasel/WebRTC is a SwiftPM-packaged wrapper around the official
        // Google `WebRTC.xcframework`. Adds ~50 MB to the build but keeps the
        // Pupa iOS app on SPM-only (no CocoaPods).
        .package(url: "https://github.com/stasel/WebRTC.git", from: "137.0.0"),
    ],
    targets: [
        .executableTarget(
            name: "PupaScreenshare",
            dependencies: [
                .product(name: "WebRTC", package: "WebRTC"),
            ],
            path: "Sources/PupaScreenshare"
        ),
    ]
)
