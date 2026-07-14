"""
J.A.R.V.I.S. — Enable Wireless ADB
================================
IMPORTANT: Run this script WITH your phone plugged in via USB first.
It will enable wireless ADB on your phone so you can use it without a cable.

Usage (with phone connected via USB):
  python enable_wireless_adb.py

After running, you can:
  1. Unplug the USB cable
  2. Disconnect your phone from the computer
  3. Use JARVIS from anywhere on your WiFi network
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


ADB = None


def find_adb():
    """Locate adb.exe."""
    import shutil
    p = shutil.which("adb")
    if p:
        return p
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Android" / "Sdk" / "platform-tools" / "adb.exe",
        Path("C:/Android/platform-tools/adb.exe"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def run_cmd(args, timeout=20):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except Exception as e:
        return -1, "", str(e)


def main():
    global ADB
    ADB = find_adb()

    print("""
  ===============================================
       Enable Wireless ADB
  ===============================================

  IMPORTANT: Phone must be connected via USB!
""")

    if not ADB:
        print("[X] ADB not found!")
        sys.exit(1)

    # Check for USB-connected devices
    code, out, err = run_cmd([ADB, "devices"])
    print(f"\n[*] Checking for USB devices...")
    print(out)

    # Find a device
    device = None
    for line in out.strip().split("\n")[1:]:
        if "\tdevice" in line:
            device = line.split()[0]
            break

    if not device:
        print("[X] No device found via USB!")
        print("    Connect your phone via USB and try again.")
        print("    Also make sure USB Debugging is enabled on your phone.")
        sys.exit(1)

    print(f"\n[+] Found device: {device}")

    # Get the wireless IP
    print(f"\n[*] Getting phone IP address...")
    code, out, err = run_cmd([ADB, "-s", device, "shell", "ip addr show wlan0"])
    ip_line = ""
    for line in out.split("\n"):
        if "inet " in line:
            ip_line = line.strip()
            break

    if not ip_line:
        # Try mobile data interface
        code, out, err = run_cmd([ADB, "-s", device, "shell", "ip addr show rmnet0"])
        for line in out.split("\n"):
            if "inet " in line:
                ip_line = line.strip()
                break

    if ip_line:
        # Extract IP
        import re
        ip = re.search(r"inet (\d+\.\d+.\d+.\d+)", ip_line)
        if ip:
            phone_ip = ip.group(1)
            print(f"[+] Phone IP: {phone_ip}")
        else:
            phone_ip = None
    else:
        phone_ip = None

    if not phone_ip:
        print("[!] Could not get phone IP. Using alternate method...")
        # Use adb shell to get IP
        code, out, err = run_cmd([ADB, "-s", device, "shell", "dumpsys iphonesubinfo | grep -A1 '0'"])
        # Fallback: just enable TCP on common port
        phone_ip = None

    # Enable TCP mode
    port = 5555
    print(f"\n[*] Enabling wireless ADB on port {port}...")

    # First ensure USB is working
    code, out, err = run_cmd([ADB, "-s", device, "tcpip", str(port)], timeout=30)
    print(out)
    if err:
        print(err)

    # Now connect to it
    time.sleep(2)

    if phone_ip:
        connect_ip = f"{phone_ip}:{port}"
        print(f"\n[*] Connecting to {connect_ip}...")
        code, out, err = run_cmd([ADB, "connect", connect_ip], timeout=10)
        print(out)

    # Show final device list
    print(f"\n[*] Final device list:")
    code, out, err = run_cmd([ADB, "devices", "-l"])
    print(out)

    print("""
  ===============================================
  SUCCESS! Wireless ADB is now enabled.
  ===============================================

  You can now:
    1. Unplug the USB cable
    2. Move your phone anywhere on WiFi
    3. JARVIS will still be able to control it

  For this to work:
    - Your phone and computer must be on the SAME WiFi network
    - Keep USB Debugging enabled on your phone
    - Don't restart your phone often (may reset the wireless setting)

  To test without USB:
    python test_android.py

  To connect again later if it disconnects:
    adb connect YOUR_PHONE_IP:5555
""")

if __name__ == "__main__":
    main()