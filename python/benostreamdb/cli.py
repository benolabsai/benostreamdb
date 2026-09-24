import argparse
import sys
import os
import platform
import subprocess
from pathlib import Path

def run_command(cmd, shell=False):
    """Run a shell command and return the return code."""
    print(f"Running: {' '.join(cmd) if not shell else cmd}")
    try:
        subprocess.run(cmd, shell=shell, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return False

def check_sudo():
    if os.geteuid() != 0:
        print("Error: This command requires root privileges. Please run with 'sudo'.", file=sys.stderr)
        sys.exit(1)

def get_os():
    os_name = platform.system()
    if os_name not in ["Linux", "Darwin"]:
        print(f"Error: Unsupported operating system: {os_name}", file=sys.stderr)
        sys.exit(1)
    return os_name

def write_file(path: Path, content: str, mode=0o644):
    print(f"Writing {path}...")
    path.write_text(content)
    path.chmod(mode)

CONF_FILE_CONTENT = """# ---------------------------------------------------------
# BenoStreamDB Search Node Configuration
# ---------------------------------------------------------
# This file is loaded by the systemd service (Linux) or 
# the launchd wrapper script (macOS). Uncomment and modify
# any of these values to configure the background service.

# --- Core Configuration ---
RUST_LOG=info
# BENOSEARCH_BIND=127.0.0.1
# BENOSEARCH_PORT=8080

# --- Qdrant API Configuration ---
QDRANT_BIND=0.0.0.0
QDRANT_PORT=6333

# --- Hardware / Performance ---
# BENOSEARCH_DEVICE=auto          # Options: auto, cpu, cuda, wgpu
# RAYON_NUM_THREADS=8
# BENOSEARCH_INDEX_CACHE_GB=4
# BENOSEARCH_AUTO_REFRESH_SECS=5
# BENOSEARCH_RRF_K=60

# --- Storage & WAL ---
# BENOSEARCH_STORAGE_URI=file:///var/lib/benostreamdb/data
# BENOSEARCH_WAL_DURABILITY=relaxed  # Options: relaxed, strict

# --- Catalog Configuration (Iceberg, etc.) ---
# BENOSEARCH_CATALOG_TYPE=rest    # Options: rest, sql, glue, nessie
# BENOSEARCH_CATALOG_URL=http://localhost:8181
# BENOSEARCH_CATALOG_TOKEN=my-token
# BENOSEARCH_CATALOG_CREDENTIAL=user:pass
# BENOSEARCH_CATALOG_NAMESPACE=default
# BENOSEARCH_CATALOG_PREFIX=
# BENOSEARCH_CATALOG_ID=
"""

SYSTEMD_SERVICE_CONTENT = """[Unit]
Description=BenoStreamDB Search Node
After=network.target

[Service]
Type=simple
# Load the configuration file
EnvironmentFile=/etc/benostreamdb/benostream-search.conf

# Execute the binary
ExecStart={binary_path}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
"""

MACOS_RUNNER_CONTENT = """#!/bin/bash
# Wrapper script for macOS launchd to load environment variables from the config file

CONF_FILE="/usr/local/etc/benostreamdb/benostream-search.conf"

if [ -f "$CONF_FILE" ]; then
    # Read the file line by line to export variables, ignoring comments and empty lines
    set -a
    source "$CONF_FILE"
    set +a
fi

exec {binary_path}
"""

LAUNCHD_PLIST_CONTENT = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.benostreamdb.search</string>
    <key>ProgramArguments</key>
    <array>
        <!-- Use the wrapper script to load the config file first -->
        <string>/usr/local/bin/benostream-search-runner.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/benostream-search.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/benostream-search.err</string>
</dict>
</plist>
"""

def install_service(args):
    check_sudo()
    os_name = get_os()
    
    binary_path = args.binary_path
    if not os.path.isfile(binary_path):
        print(f"Warning: Binary not found at {binary_path}")
        print("Please ensure you have built or installed benostream-search before starting the service.")
    
    if os_name == "Linux":
        conf_dir = Path("/etc/benostreamdb")
    else:
        conf_dir = Path("/usr/local/etc/benostreamdb")
        
    conf_dir.mkdir(parents=True, exist_ok=True)
    conf_path = conf_dir / "benostream-search.conf"
    if not conf_path.exists():
        write_file(conf_path, CONF_FILE_CONTENT)
    else:
        print(f"Configuration file {conf_path} already exists, skipping...")
        
    if os_name == "Linux":
        print("Installing systemd service...")
        service_path = Path("/etc/systemd/system/benostream-search.service")
        write_file(service_path, SYSTEMD_SERVICE_CONTENT.format(binary_path=binary_path))
        
        run_command(["systemctl", "daemon-reload"])
        run_command(["systemctl", "enable", "benostream-search.service"])
        run_command(["systemctl", "start", "benostream-search.service"])
        print("Service installed and started! Check logs with: sudo journalctl -u benostream-search.service -f")
        
    elif os_name == "Darwin":
        print("Installing launchd service (macOS)...")
        runner_path = Path("/usr/local/bin/benostream-search-runner.sh")
        write_file(runner_path, MACOS_RUNNER_CONTENT.format(binary_path=binary_path), mode=0o755)
        
        plist_path = Path("/Library/LaunchDaemons/com.benostreamdb.search.plist")
        write_file(plist_path, LAUNCHD_PLIST_CONTENT)
        
        run_command(["chown", "root:wheel", str(plist_path)])
        
        run_command(["launchctl", "unload", "-w", str(plist_path)])
        run_command(["launchctl", "load", "-w", str(plist_path)])
        print("Service installed and started! Check logs with: tail -f /tmp/benostream-search.log")

def uninstall_service(args):
    check_sudo()
    os_name = get_os()
    
    if os_name == "Linux":
        print("Stopping and disabling systemd service...")
        run_command(["systemctl", "stop", "benostream-search.service"])
        run_command(["systemctl", "disable", "benostream-search.service"])
        
        service_path = Path("/etc/systemd/system/benostream-search.service")
        if service_path.exists():
            service_path.unlink()
        run_command(["systemctl", "daemon-reload"])
        
        conf_dir = Path("/etc/benostreamdb")
    elif os_name == "Darwin":
        print("Unloading and removing launchd service...")
        plist_path = Path("/Library/LaunchDaemons/com.benostreamdb.search.plist")
        run_command(["launchctl", "unload", "-w", str(plist_path)])
        if plist_path.exists():
            plist_path.unlink()
            
        runner_path = Path("/usr/local/bin/benostream-search-runner.sh")
        if runner_path.exists():
            runner_path.unlink()
            
        conf_dir = Path("/usr/local/etc/benostreamdb")

    if conf_dir.exists():
        print(f"Removing configuration files at {conf_dir}...")
        for item in conf_dir.iterdir():
            item.unlink()
        conf_dir.rmdir()
        
    print("Uninstallation Complete!")

def main():
    parser = argparse.ArgumentParser(description="BenoStreamDB CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    install_parser = subparsers.add_parser("install-service", help="Install the native background service (requires sudo)")
    install_parser.add_argument("--binary-path", type=str, default="/usr/local/bin/benostream-search",
                                help="Path to the benostream-search binary (default: /usr/local/bin/benostream-search)")
    install_parser.set_defaults(func=install_service)
    
    uninstall_parser = subparsers.add_parser("uninstall-service", help="Uninstall the native background service (requires sudo)")
    uninstall_parser.set_defaults(func=uninstall_service)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
