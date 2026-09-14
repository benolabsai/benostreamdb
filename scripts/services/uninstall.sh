#!/bin/bash
set -e

echo "=================================================="
echo "  HyperStreamDB Search Background Service Uninstaller"
echo "=================================================="

# Detect OS
OS="$(uname -s)"
if [ "$OS" != "Linux" ] && [ "$OS" != "Darwin" ]; then
    echo "Error: Unsupported operating system: $OS"
    exit 1
fi

echo "Requesting administrative privileges for uninstallation..."
sudo -v

if [ "$OS" = "Linux" ]; then
    echo "Stopping and disabling systemd service..."
    sudo systemctl stop hyperstream-search.service || true
    sudo systemctl disable hyperstream-search.service || true
    
    echo "Removing systemd service file..."
    sudo rm -f /etc/systemd/system/hyperstream-search.service
    sudo systemctl daemon-reload
    
    CONF_DIR="/etc/hyperstreamdb"
    
elif [ "$OS" = "Darwin" ]; then
    echo "Unloading and removing launchd service..."
    PLIST_DEST="/Library/LaunchDaemons/com.hyperstreamdb.search.plist"
    sudo launchctl unload -w "$PLIST_DEST" 2>/dev/null || true
    sudo rm -f "$PLIST_DEST"
    
    echo "Removing macOS wrapper script..."
    sudo rm -f /usr/local/bin/hyperstream-search-runner.sh
    
    CONF_DIR="/usr/local/etc/hyperstreamdb"
fi

echo "Removing binary..."
sudo rm -f /usr/local/bin/hyperstream-search

echo "Removing configuration files at $CONF_DIR..."
sudo rm -rf "$CONF_DIR"

echo "=================================================="
echo "  Uninstallation Complete!"
echo "=================================================="
