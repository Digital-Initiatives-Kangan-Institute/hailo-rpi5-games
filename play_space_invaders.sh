#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/setup_env.sh"
python "$SCRIPT_DIR/community_projects/space_invaders/Space_Invaders.py" --input usb "$@"
