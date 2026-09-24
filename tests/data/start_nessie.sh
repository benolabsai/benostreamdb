#!/bin/bash
# Start Nessie in Docker
echo "Starting Nessie..."
docker run -d -p 19120:19120 --name nessie projectnessie/nessie
echo "Waiting for Nessie to be ready..."
sleep 5
echo "Nessie started on http://localhost:19120"
