# HyperStreamDB Background Services

This directory contains configuration files to run `hyperstream-search` natively in the background, allowing it to start automatically on system boot.

Running natively provides the best performance and enables direct GPU (CUDA/wgpu) access without needing to configure complex Docker GPU passthrough like the NVIDIA Container Toolkit.

By default, the Qdrant-compatible REST API is exposed on `http://localhost:6333`.

## Prerequisites
Ensure `hyperstream-search` is built and installed to a known location, such as `/usr/local/bin/hyperstream-search`. If it's installed somewhere else, please edit the path in the respective service file before installation.

## Linux (systemd)

1. Copy the `.service` file to the systemd directory:
   ```bash
   sudo cp hyperstream-search.service /etc/systemd/system/
   ```
2. Reload the systemd daemon:
   ```bash
   sudo systemctl daemon-reload
   ```
3. Enable the service to start automatically on boot:
   ```bash
   sudo systemctl enable hyperstream-search.service
   ```
4. Start the service immediately:
   ```bash
   sudo systemctl start hyperstream-search.service
   ```
5. Check logs:
   ```bash
   sudo journalctl -u hyperstream-search.service -f
   ```

## macOS (launchd)

1. Copy the `.plist` file to the LaunchDaemons directory (requires admin) or LaunchAgents (for user only). For system-wide:
   ```bash
   sudo cp com.hyperstreamdb.search.plist /Library/LaunchDaemons/
   ```
2. Set correct permissions:
   ```bash
   sudo chown root:wheel /Library/LaunchDaemons/com.hyperstreamdb.search.plist
   ```
3. Load and start the service:
   ```bash
   sudo launchctl load -w /Library/LaunchDaemons/com.hyperstreamdb.search.plist
   ```
4. Check logs:
   ```bash
   tail -f /tmp/hyperstream-search.log
   ```
