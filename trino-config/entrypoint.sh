#!/bin/bash
set -e

# Parse PG_CONN_RW if provided and generate postgres.properties
if [ -n "$PG_CONN_RW" ] && [ "$PG_CONN_RW" != "" ]; then
  # Parse postgresql://user:password@host:port/database
  # Remove postgresql:// prefix
  CONN_STR="${PG_CONN_RW#postgresql://}"
  CONN_STR="${CONN_STR#postgres://}"
  CONN_STR="${CONN_STR#pgsql://}"
  
  # Extract components: user:password@host:port/database
  if [[ $CONN_STR =~ ^([^:]+):([^@]+)@([^:]+):([0-9]+)/(.+)$ ]]; then
    PG_USER="${BASH_REMATCH[1]}"
    PG_PASSWORD="${BASH_REMATCH[2]}"
    PG_HOST="${BASH_REMATCH[3]}"
    PG_PORT="${BASH_REMATCH[4]}"
    PG_DATABASE="${BASH_REMATCH[5]}"
    
    # If host is localhost, use host.docker.internal for Docker networking
    if [ "$PG_HOST" = "localhost" ] || [ "$PG_HOST" = "127.0.0.1" ]; then
      PG_HOST="host.docker.internal"
    fi
    
    # Generate postgres.properties
    cat > /etc/trino/catalog/postgres.properties <<EOF
connector.name=postgresql
connection-url=jdbc:postgresql://${PG_HOST}:${PG_PORT}/${PG_DATABASE}
connection-user=${PG_USER}
connection-password=${PG_PASSWORD}
EOF
    
    echo "Generated postgres.properties from PG_CONN_RW"
    echo "  Host: ${PG_HOST}:${PG_PORT}"
    echo "  Database: ${PG_DATABASE}"
    echo "  User: ${PG_USER}"
  else
    echo "Warning: Could not parse PG_CONN_RW: $PG_CONN_RW"
    echo "Expected format: postgresql://user:password@host:port/database"
  fi
fi

# Execute the original Trino entrypoint
exec /usr/lib/trino/bin/run-trino

