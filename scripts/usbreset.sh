#!/bin/bash
# USB-reset the three Orbbec Gemini 305 cameras (VID:PID 2bc5:0840).
#
# Run on the *host*, not inside the Noetic container. Needs sudo.
# After this, restart the robot stack so the driver re-opens the devices:
#
#   ./scripts/usbreset.sh
#   ./scripts/run_collection.sh -i
#
set -euo pipefail

VIDPID=2bc5:0840

if ! command -v usbreset >/dev/null; then
    echo "usbreset not found; install with: sudo apt install usbutils" >&2
    exit 1
fi

mapfile -t devices < <(lsusb -d "$VIDPID" | awk '{gsub(":","",$4); print $2"/"$4}')
if [[ ${#devices[@]} -eq 0 ]]; then
    echo "no Orbbec Gemini 305 ($VIDPID) on USB" >&2
    lsusb | grep -i orbbec || true
    exit 1
fi

echo "found ${#devices[@]} camera(s): ${devices[*]}"
for dev in "${devices[@]}"; do
    sudo usbreset "$dev"
done
