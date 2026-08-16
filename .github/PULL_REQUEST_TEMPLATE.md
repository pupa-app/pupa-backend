## Summary

<!-- What does this PR change, and why? One or two sentences. -->

## Test plan

<!-- What you ran and what you eyeballed. -->

- [ ] `cd backend && uv sync && uv run pytest`
- [ ] `swift build --package-path screenshare-sidecar` (only if the sidecar was touched; macOS-only)

## Checklist

- [ ] Base branch is `dev` (not `main`).
- [ ] Version bumped if code in a sub-package changed (see [CONTRIBUTING.md](../CONTRIBUTING.md) → Releases).
- [ ] Docs updated if behaviour changed ([docs/architecture.md](../docs/architecture.md), [CHANGELOG.md](../CHANGELOG.md)).
