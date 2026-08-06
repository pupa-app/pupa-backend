# Shell startup script template for the agent's shell session.
#
# Copy to shell_startup.local.sh (gitignored) and customise.
# Set SHELL_STARTUP_SCRIPT=/path/to/your/script in .env to use a different path.
#
# Each non-empty, non-comment line runs once when the shell session starts.
# The subprocess inherits the full backend environment (HOME, PATH, etc.).

# --- gh CLI auth ---
# Wraps `gh` so GH_TOKEN is fetched from the macOS Keychain per-invocation
# and never appears in printenv. Adjust the binary path if gh is elsewhere.
#
# gh() { GH_TOKEN=$(/opt/homebrew/bin/gh auth token 2>/dev/null) /opt/homebrew/bin/gh "$@"; }

# --- extra PATH ---
# export PATH="$PATH:/your/custom/bin"
