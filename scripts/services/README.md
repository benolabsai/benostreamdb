# HyperStreamDB Search Background Services

This directory contains configuration files and an installer script to run `hyperstream-search` natively in the background, allowing it to start automatically on system boot.

Running natively provides the best performance and enables direct GPU (CUDA/wgpu) access without needing to configure complex Docker GPU passthrough.

By default, the Qdrant-compatible REST API is exposed on `http://localhost:6333`.

## Automated Installation (Linux & macOS)

The easiest way to install the background service is to run the provided installer script.

Ensure you have built the `hyperstream-search` binary first:
```bash
cargo build --release -p hyperstreamdb-search
```

Then run the installer:
```bash
./install.sh
```

The script will automatically detect your OS, install the binary to `/usr/local/bin`, and configure the background service (`systemd` for Linux, `launchd` for macOS).

## Configuration

The background service uses a centralized configuration file where you can adjust environment variables (such as enabling GPU, changing ports, or configuring storage).

- **Linux**: Edit `/etc/hyperstreamdb/hyperstream-search.conf`
- **macOS**: Edit `/usr/local/etc/hyperstreamdb/hyperstream-search.conf`

After changing the configuration file, you must restart the service:

- **Linux**:
  ```bash
  sudo systemctl restart hyperstream-search.service
  ```
- **macOS**:
  ```bash
  sudo launchctl unload -w /Library/LaunchDaemons/com.hyperstreamdb.search.plist
  sudo launchctl load -w /Library/LaunchDaemons/com.hyperstreamdb.search.plist
  ```

## Viewing Logs

- **Linux**:
  ```bash
  sudo journalctl -u hyperstream-search.service -f
  ```
- **macOS**:
  ```bash
  tail -f /tmp/hyperstream-search.log
  ```
