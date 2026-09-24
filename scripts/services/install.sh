#!/bin/bash
set -e

echo "=================================================="
echo "  BenoStreamDB Search Background Service Installer"
echo "=================================================="

# Detect OS
OS="$(uname -s)"
if [ "$OS" != "Linux" ] && [ "$OS" != "Darwin" ]; then
    echo "Error: Unsupported operating system: $OS"
    exit 1
fi

# Ensure benostream-search is built or available in target/release
BINARY_SRC="../../target/release/benostream-search"
if [ ! -f "$BINARY_SRC" ]; then
    echo "Warning: $BINARY_SRC not found."
    echo "Make sure you build the project first with: cargo build --release -p benostreamdb-search"
    # We won't exit here, just warn them, in case they already have it installed
fi

# Need sudo for installation
echo "Requesting administrative privileges for installation..."
sudo -v

echo "Installing benostream-search binary to /usr/local/bin..."
if [ -f "$BINARY_SRC" ]; then
    sudo cp "$BINARY_SRC" /usr/local/bin/benostream-search
fi
sudo chmod +x /usr/local/bin/benostream-search

# Install configuration file
if [ "$OS" = "Linux" ]; then
    CONF_DIR="/etc/benostreamdb"
else
    # macOS
    CONF_DIR="/usr/local/etc/benostreamdb"
fi

echo "Creating configuration directory at $CONF_DIR..."
sudo mkdir -p "$CONF_DIR"

echo "Installing benostream-search.conf..."
sudo cp benostream-search.conf "$CONF_DIR/"
echo "You can configure your settings by editing: $CONF_DIR/benostream-search.conf"

# Install Services
if [ "$OS" = "Linux" ]; then
    echo "Installing systemd service (Linux)..."
    sudo cp benostream-search.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable benostream-search.service
    sudo systemctl start benostream-search.service
    echo "Service installed and started! Check logs with: sudo journalctl -u benostream-search.service -f"
    
elif [ "$OS" = "Darwin" ]; then
    echo "Installing launchd service (macOS)..."
    sudo cp benostream-search-runner.sh /usr/local/bin/
    sudo chmod +x /usr/local/bin/benostream-search-runner.sh
    
    PLIST_DEST="/Library/LaunchDaemons/com.benostreamdb.search.plist"
    sudo cp com.benostreamdb.search.plist "$PLIST_DEST"
    sudo chown root:wheel "$PLIST_DEST"
    
    # Reload if it was already loaded
    sudo launchctl unload -w "$PLIST_DEST" 2>/dev/null || true
    sudo launchctl load -w "$PLIST_DEST"
    
    echo "Service installed and started! Check logs with: tail -f /tmp/benostream-search.log"
fi

echo "=================================================="
echo "  Installation Complete!"
echo "=================================================="
