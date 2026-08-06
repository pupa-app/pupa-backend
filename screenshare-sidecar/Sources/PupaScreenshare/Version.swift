// Version of the `pupa-screenshare` sidecar binary. Bump in lockstep
// with `screenshare-sidecar/CHANGELOG.md`. Reported to stderr on startup so
// it shows up in `pupa-backend screenshare` output and (Phase 2+) sent to the broker
// in PublisherHello for backend-side compatibility checks.
public let PupaScreenshareVersion = "0.0.6"
