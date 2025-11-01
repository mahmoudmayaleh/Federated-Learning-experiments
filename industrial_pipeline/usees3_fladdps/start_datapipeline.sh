#!/bin/bash
set -euo pipefail

# Configurations:
COMPOSE_DIR="."
# Define networks to be removed
NETWORKS=("server_client_network" "dps_network")

# Function to show usage help
usage() {
  echo "Usage: $0 <docker-compose-file.yml>"
  echo "Example: $0 docker-compose-task2.yml"
  exit 1
}

# Check for docker compose
if [[ $# -ne 1 ]]; then
  echo "Error: Missing Docker Compose file argument."
  usage
fi

COMPOSE_FILE="$1"

# Ensure the Compose file exists
if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Error: File '$COMPOSE_FILE' does not exist."
  exit 1
fi

# Function to remove a network if it exists
remove_network_if_exists() {
  local network="$1"
  if docker network ls --format '{{.Name}}' | grep -q "^${network}$"; then
    echo "Removing network: $network"
    docker network rm "$network"
  else
    echo "Network '$network' does not exist, skipping..."
  fi
}


# Stop and clean up containers
echo "🧹 Stopping and removing containers..."
docker compose -f "$COMPOSE_FILE" down --remove-orphans

# Remove stopeed containers
docker container prune -f

# Prune unused networks
echo "🧼 Pruning unused networks..."
docker network prune -f

# Remove dps networks
for net in "${NETWORKS[@]}"; do
  remove_network_if_exists "$net"
done

# Rebuild and start containers
echo "Starting up containers with build..."
docker compose -f "$COMPOSE_FILE" up --build
