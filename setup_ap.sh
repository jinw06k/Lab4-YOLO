#!/usr/bin/env bash
# =============================================================================
# setup_ap.sh -- turn this Pi Zero 2W into its own 2.4 GHz wifi access point.
#
# Why: the Pi Zero 2W radio is 2.4 GHz only and MWireless is 5 GHz only, so the
# Pi can't join campus wifi. Instead of putting a router on the campus network
# (needs DCO approval), the Pi hosts its OWN network. A laptop joins it directly
# and views the YOLO video stream from partF.py -- no campus wifi, no router.
#
# After setup, on your laptop:
#     1. join the wifi SSID printed below
#     2. open   http://10.42.0.1:8000/      (the video stream)
#     3. SSH if needed:  ssh <user>@10.42.0.1
#
# Run ONCE per Pi (re-run to change SSID/password):
#     sudo bash setup_ap.sh                       # SSID defaults to Lab4-<hostname>
#     sudo bash setup_ap.sh MySSID MyPassword     # or set your own
#
# Requires Raspberry Pi OS Bookworm (uses NetworkManager / nmcli).
# NOTE: bringing the AP up DROPS any current wifi connection to the Pi. Run this
# from a monitor+keyboard, or over USB/ethernet, or be ready to reconnect to the
# new SSID afterward.
# =============================================================================
set -euo pipefail

# Each Pi in the room MUST have a unique SSID, or laptops can't tell them apart.
# Defaults to the Pi's hostname; override via the two positional args.
SSID="${1:-Lab4-$(hostname)}"
PASS="${2:-eecs473lab4}"        # WPA2 password, must be 8-63 characters
CON="lab4-ap"                    # NetworkManager connection profile name

if [[ $EUID -ne 0 ]]; then
    echo "Please run with sudo:  sudo bash setup_ap.sh"
    exit 1
fi

if (( ${#PASS} < 8 || ${#PASS} > 63 )); then
    echo "Password must be 8-63 characters (got ${#PASS})."
    exit 1
fi

# The wifi radio stays blocked until a regulatory country is set.
raspi-config nonint do_wifi_country US || true

# Recreate the profile from scratch so re-runs are clean.
nmcli connection delete "$CON" 2>/dev/null || true
nmcli connection add type wifi ifname wlan0 con-name "$CON" autoconnect yes ssid "$SSID"
nmcli connection modify "$CON" \
    802-11-wireless.mode ap \
    802-11-wireless.band bg \
    ipv4.method shared \
    wifi-sec.key-mgmt wpa-psk \
    wifi-sec.psk "$PASS"
nmcli connection up "$CON"

cat <<EOF

===================================================================
 Access point is UP and will auto-start on every boot.
   SSID:     $SSID
   Password: $PASS

 On your laptop: join that wifi, then open
       http://10.42.0.1:8000/        (YOLO video stream)
 SSH to the Pi:  ssh \$USER@10.42.0.1

 To undo (go back to normal wifi):
       sudo nmcli connection delete $CON
===================================================================
EOF
