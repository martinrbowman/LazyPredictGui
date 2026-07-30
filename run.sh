#!/usr/bin/env bash
# Creates .venv on first run (installing requirements.txt into it), reuses
# it on every later run, then launches the GUI.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -d .venv ]; then
	echo "Creating venv in .venv ..."
	python3 -m venv .venv
	./.venv/bin/pip install --upgrade pip
	./.venv/bin/pip install -r requirements.txt
fi

exec ./.venv/bin/python main.py
