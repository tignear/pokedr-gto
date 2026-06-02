#!/usr/bin/env bash
set -euo pipefail

mkdir -p "$HOME/.codex" "$HOME/.local/bin"

if command -v sudo >/dev/null 2>&1; then
	sudo chown -R "$(id -u):$(id -g)" "$HOME/.codex" "$HOME/.local"
else
	chown -R "$(id -u):$(id -g)" "$HOME/.codex" "$HOME/.local"
fi

if command -v codex >/dev/null 2>&1 || [ -x "$HOME/.local/bin/codex" ]; then
	exit 0
fi

curl -fsSL https://chatgpt.com/codex/install.sh | CODEX_NON_INTERACTIVE=1 sh
