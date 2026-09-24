# BenoStreamDB Search Background Services

This directory contains configuration files and an installer script to run `benostream-search` natively in the background, allowing it to start automatically on system boot.

Running natively provides the best performance and enables direct GPU (CUDA/wgpu) access without needing to configure complex Docker GPU passthrough.

By default, the Qdrant-compatible REST API is exposed on `http://localhost:6333`.

## Automated Installation (Linux & macOS)

### Method 1: Using the Python CLI (Recommended)

If you have installed the `benostreamdb` Python package via PyPI, you can install and configure the background service directly from the terminal:

```bash
sudo benostreamdb install-service
```
This automatically detects your OS and installs the required `systemd` or `launchd` configuration files. (Note: you must have the `benostream-search` binary built or installed at `/usr/local/bin/benostream-search`, or pass `--binary-path`).

### Method 2: Using the Shell Scripts

If you are building from source and prefer not to use the Python CLI:

Ensure you have built the `benostream-search` binary first:
```bash
cargo build --release -p benostreamdb-search
```

Then run the installer:
```bash
./install.sh
```

The script will automatically detect your OS, install the binary to `/usr/local/bin`, and configure the background service (`systemd` for Linux, `launchd` for macOS).

## Configuration

The background service uses a centralized configuration file where you can adjust environment variables (such as enabling GPU, changing ports, or configuring storage).

- **Linux**: Edit `/etc/benostreamdb/benostream-search.conf`
- **macOS**: Edit `/usr/local/etc/benostreamdb/benostream-search.conf`

After changing the configuration file, you must restart the service:

- **Linux**:
  ```bash
  sudo systemctl restart benostream-search.service
  ```
- **macOS**:
  ```bash
  sudo launchctl unload -w /Library/LaunchDaemons/com.benostreamdb.search.plist
  sudo launchctl load -w /Library/LaunchDaemons/com.benostreamdb.search.plist
  ```

## Viewing Logs

- **Linux**:
  ```bash
  sudo journalctl -u benostream-search.service -f
  ```
- **macOS**:
  ```bash
  tail -f /tmp/benostream-search.log
  ```

## Uninstallation

To completely remove the service, configuration, and binary, you can use the Python CLI:
```bash
sudo benostreamdb uninstall-service
```

Or you can use the included script:
```bash
./uninstall.sh
```
