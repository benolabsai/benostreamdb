#!/bin/bash
# Wrapper script for macOS launchd to load environment variables from the config file

CONF_FILE="/usr/local/etc/hyperstreamdb/hyperstream-search.conf"

if [ -f "$CONF_FILE" ]; then
    # Read the file line by line to export variables, ignoring comments and empty lines
    set -a
    source "$CONF_FILE"
    set +a
fi

exec /usr/local/bin/hyperstream-search
