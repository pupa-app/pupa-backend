.DEFAULT_GOAL := help
.PHONY: help install install-local install-backend install-playwright install-setup install-cli setup backend backend-shell backend-open backend-keyed backend-tunneled tunnel tunnel-named tunneled pair screenshare screenshare-viewer service-install service-uninstall service-status test test-backend clean

BACKEND := backend
SCREENSHARE := screenshare-sidecar
# Auto-detect wss:// when TLS is configured in ~/.pupa-backend/config.yml
_PUPA_TLS := $(shell python3 -c "import yaml,pathlib; d=yaml.safe_load(pathlib.Path.home()/'.pupa-backend/config.yml'); print(d.get('tls',{}).get('cert',''))" 2>/dev/null)
SCREENSHARE_BROKER ?= $(if $(_PUPA_TLS),wss,ws)://localhost:8004/screenshare/ws
SHARE_ID ?= $(shell uuidgen)
SCREENSHARE_VIEWER_PORT ?= 8005
SIDECAR_TOKEN_FILE ?= /tmp/pupa-sidecar.token

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: install-backend  ## Install backend deps (uv)

install-local: install-backend install-setup setup install-cli  ## Full install from inside the repo (deps + setup wizard + CLI)

install-backend:  ## Install backend deps via uv sync (respects uv.lock)
	cd $(BACKEND) && uv sync

install-setup:  ## Install setup extras (qrcode + cryptography + pyyaml for TLS cert gen and QR pairing)
	cd $(BACKEND) && uv sync --extra setup

install-cli:  ## Install the pupa-backend CLI to ~/.local/bin/
	@bash install.sh --local --cli-only

setup:  ## Interactive onboarding wizard — writes ~/.pupa-backend/.env, optionally generates TLS cert and installs the service
	@cd $(BACKEND) && .venv/bin/python -m pupa_backend.scripts.setup

service-install:  ## Install backend as a launchd (macOS) or systemd (Linux) user service with auto-restart
	@cd $(BACKEND) && .venv/bin/python -m pupa_backend.scripts.service install

service-uninstall:  ## Remove the background service
	@cd $(BACKEND) && .venv/bin/python -m pupa_backend.scripts.service uninstall

service-status:  ## Show background service status
	@cd $(BACKEND) && .venv/bin/python -m pupa_backend.scripts.service status

install-mcp:  ## MCP client deps are core now — alias for a plain sync
	cd $(BACKEND) && uv sync

install-playwright:  ## Sync deps + download chromium for Playwright (requires Node/npx)
	@command -v npx >/dev/null 2>&1 || { echo "npx not found — install Node.js: brew install node" >&2; exit 1; }
	cd $(BACKEND) && uv sync
	NODE_TLS_REJECT_UNAUTHORIZED=0 npx --yes playwright install chromium

backend:  ## Run the LangGraph A2UI backend on :8004 (LLM provider from shell env, else ~/.pupa-backend/config.yml)
	@cd $(BACKEND) && .venv/bin/python -m pupa_backend.app

backend-shell:  ## Backend with shell tool enabled; pass env to subprocess, exclude vars from ~/.zshrc (override via SHELL_PASS_ENV / SHELL_ENV_EXCLUDE / SHELL_ENV_EXCLUDE_FROM)
	@export SHELL_TOOL_ENABLED=1 \
	&& export SHELL_PASS_ENV=$${SHELL_PASS_ENV:-1} \
	&& export SHELL_ENV_EXCLUDE=$${SHELL_ENV_EXCLUDE:-GH_TOKEN} \
	&& export SHELL_ENV_EXCLUDE_FROM=$${SHELL_ENV_EXCLUDE_FROM:-$$HOME/.zshrc} \
	&& cd $(BACKEND) && .venv/bin/python -m pupa_backend.app

backend-open:  ## Backend with auth disabled (PUPA_AUTH_DISABLED=1). Local dev only — never expose this to a reachable backend.
	@export PUPA_AUTH_DISABLED=1 && cd $(BACKEND) && .venv/bin/python -m pupa_backend.app

backend-keyed:  ## Backend with a freshly generated PUPA_API_KEY (local, no tunnel). Run `make pair` after to get a pairing code.
	@KEY=$$(openssl rand -hex 16) && \
	  echo "$$KEY" > .api-key && \
	  echo "" && echo "  PUPA_API_KEY=$$KEY" && \
	  echo "  (run 'make pair' in another shell to pair a device)" && echo "" && \
	  export PUPA_API_KEY=$$KEY && cd $(BACKEND) && .venv/bin/python -m pupa_backend.app

backend-tunneled:  ## Backend with PUPA_API_KEY set (auto-generated if unset). Pair with `make tunnel` to test from a phone.
	@KEY=$${PUPA_API_KEY:-$$(openssl rand -hex 16)} && \
	  echo "" && echo "  PUPA_API_KEY=$$KEY" && \
	  echo "  (paste this + the cloudflared URL into iOS Settings → Backend)" && echo "" && \
	  export PUPA_API_KEY=$$KEY && cd $(BACKEND) && .venv/bin/python -m pupa_backend.app

tunnel:  ## Expose localhost:8004 via Cloudflare quick tunnel (HTTPS, no signup). Pair with `make backend-tunneled`.
	@command -v cloudflared >/dev/null 2>&1 || { echo "cloudflared not found — install via: brew install cloudflared" >&2; exit 1; }
	cloudflared tunnel --url http://localhost:8004

tunnel-named:  ## Run the configured Cloudflare *named* tunnel (stable URL on your own domain). Set up via `make setup` (cloudflared + your domain). Override the tunnel name with NAME=...
	@command -v cloudflared >/dev/null 2>&1 || { echo "cloudflared not found — install via: brew install cloudflared" >&2; exit 1; }
	@NAME=$${NAME:-$$(python3 -c "import yaml,pathlib; d=yaml.safe_load((pathlib.Path.home()/'.pupa-backend/config.yml').read_text()) or {}; print((d.get('cloudflared') or {}).get('tunnel',''))" 2>/dev/null)}; \
	  if [ -z "$$NAME" ]; then echo "no named tunnel configured — run 'make setup' (cloudflared + your domain), or pass NAME=<tunnel>" >&2; exit 1; fi; \
	  echo "  cloudflared tunnel run $$NAME"; \
	  cloudflared tunnel run "$$NAME"

tunneled:  ## Run backend-tunneled + tunnel together (Ctrl+C stops both)
	$(MAKE) -j2 backend-tunneled tunnel

pair:  ## Mint a one-time pairing code + QR. LABEL="iPhone" pre-fills the label; URL=https://... targets a remote deploy; KEY=... is that backend's PUPA_API_KEY; CODE_TTL / DEVICE_TTL are lifetimes in seconds.
	@cd $(BACKEND) && $(if $(KEY),PUPA_API_KEY="$(KEY)" ,).venv/bin/python -m pupa_backend.scripts.pair \
		$(if $(LABEL),--label "$(LABEL)",) \
		$(if $(URL),--public-url "$(URL)",) \
		$(if $(CODE_TTL),--code-ttl $(CODE_TTL),) \
		$(if $(DEVICE_TTL),--device-ttl $(DEVICE_TTL),)

mcp:  ## Manage MCP servers in ~/.pupa-backend/config.yml. e.g. make mcp ARGS="add", make mcp ARGS="list", make mcp ARGS="remove atlassian"
	@cd $(BACKEND) && .venv/bin/python -m pupa_backend.scripts.mcp $(ARGS)

smoke:  ## Hit the running backend with the auth + scope check matrix. Pass URL="https://..." and KEY="..." to target a remote (e.g. Railway); defaults to localhost:8004 + auth.api_key from ~/.pupa-backend/config.yml.
	@cd $(BACKEND) && .venv/bin/python -m pupa_backend.scripts.smoke \
		$(if $(URL),--base-url "$(URL)",) \
		$(if $(KEY),--api-key  "$(KEY)",)

screenshare:  ## Use `pupa-backend screenshare` instead. This is for development. Run the screen-share sidecar — pick a window, publish via WebRTC to the broker. Pair with `make screenshare-viewer` in another shell, or use the iOS app (Phase 3+).
	@token=$$(cat $(SIDECAR_TOKEN_FILE) 2>/dev/null); \
	if [ -z "$$token" ]; then \
	  echo "error: sidecar token not found. Start the backend first: pupa-backend run"; \
	  exit 1; \
	fi; \
	echo ""; \
	echo "  share id: $(SHARE_ID)"; \
	echo "  (paste this into the browser viewer at http://localhost:$(SCREENSHARE_VIEWER_PORT)/viewer.html)"; \
	echo ""; \
	swift run --package-path $(SCREENSHARE) pupa-screenshare --broker $(SCREENSHARE_BROKER) --share-id $(SHARE_ID) --api-key "$$token"

screenshare-viewer:  ## Serve the debug browser viewer on http://localhost:8005/viewer.html (override port with SCREENSHARE_VIEWER_PORT). Binds 127.0.0.1 so the browser treats the page as a secure context.
	cd $(SCREENSHARE)/viewer && python3 -m http.server --bind 127.0.0.1 $(SCREENSHARE_VIEWER_PORT)

test: test-backend  ## Run backend pytest suite

test-backend:  ## Run backend pytest suite (use FILTER=foo to scope via -k)
	cd $(BACKEND) && .venv/bin/pytest $(if $(FILTER),-k $(FILTER),)

clean:  ## Remove backend venv
	rm -rf $(BACKEND)/.venv
