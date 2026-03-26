#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: Hailo environment is not active."
    echo ""
    echo "Please source the setup script before running this game:"
    echo ""
    echo "    source $SCRIPT_DIR/setup_env.sh"
    echo ""
    echo "Then run this script again."
    exit 1
fi

python "$SCRIPT_DIR/community_projects/dance_dance_revolution/DDR_Hailo.py" --input usb "$@"
