#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/setup_env.sh"
python "$SCRIPT_DIR/community_projects/fruit_ninja/Fruit_Ninja_Hailo.py" --input usb "$@"
