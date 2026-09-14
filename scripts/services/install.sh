#!/bin/bash
set -e

echo "=================================================="
echo "  HyperStreamDB Search Background Service Installer"
echo "=================================================="

# Detect OS
OS="$(uname -s)"
if [ "$OS" != "Linux" ] && [ "$OS" != "Darwin" ]; then
    echo "Error: Unsupported operating system: $OS"
    exit 1
fi

# Ensure hyperstream-search is built or available in target/release
BINARY_SRC="../../target/release/hyperstream-search"
if [ ! -f "$BINARY_SRC" ]; then
    echo "Warning: $BINARY_SRC not found."
    echo "Make sure you build the project first with: cargo build --release -p hyperstreamdb-search"
    # We won't exit here, just warn them, in case they already have it installed
fi

# Need sudo for installation
echo "Requesting administrative privileges for installation..."
sudo -v

echo "Installing hyperstream-search binary to /usr/local/bin..."
if [ -f "$BINARY_SRC" ]; then
    sudo cp "$BINARY_SRC" /usr/local/bin/hyperstream-search
fi
sudo chmod +x /usr/local/bin/hyperstream-search

# Install configuration file
if [ "$OS" = "Linux" ]; then
    CONF_DIR="/etc/hyperstreamdb"
else
    # macOS
    CONF_DIR="/usr/local/etc/hyperstreamdb"
fi

echo "Creating configuration directory at $CONF_DIR..."
sudo mkdir -p "$CONF_DIR"

echo "Installing hyperstream-search.conf..."
sudo cp hyperstream-search.conf "$CONF_DIR/"
echo "You can configure your settings by editing: $CONF_DIR/hyperstream-search.conf"

# Install Services
if [ "$OS" = "Linux" ]; then
    echo "Installing systemd service (Linux)..."
    sudo cp hyperstream-search.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable hyperstream-search.service
    sudo systemctl start hyperstream-search.service
    echo "Service installed and started! Check logs with: sudo journalctl -u hyperstream-search.service -f"
    
elif [ "$OS" = "Darwin" ]; then
    echo "Installing launchd service (macOS)..."
    sudo cp hyperstream-search-runner.sh /usr/local/bin/
    sudo chmod +x /usr/local/bin/hyperstream-search-runner.sh
    
    PLIST_DEST="/Library/LaunchDaemons/com.hyperstreamdb.search.plist"
    sudo cp com.hyperstreamdb.search.plist "$PLIST_DEST"
    sudo chown root:wheel "$PLIST_DEST"
    
    # Reload if it was already loaded
    sudo launchctl unload -w "$PLIST_DEST" 2>/dev/null || true
    sudo launchctl load -w "$PLIST_DEST"
    
    echo "Service installed and started! Check logs with: tail -f /tmp/hyperstream-search.log"
fi

echo "=================================================="
echo "  Installation Complete!"
echo "=================================================="
