#!/bin/bash

# Generate SSH key pair if not already present
if [ ! -f /root/.ssh/id_rsa ]; then
    ssh-keygen -t rsa -N "" -f /root/.ssh/id_rsa
fi

# List of client container names (passed as arguments, e.g., flclient-1 flclient-2)
CLIENTS="$@"
if [ -z "$CLIENTS" ]; then
    echo "Error: No client container names provided"
    exit 1
fi

# Copy public key to each client
for CLIENT in $CLIENTS; do
    echo "Pinging $CLIENT to ensure it's reachable..."
    timeout 100s bash -c "until ping -c 1 $CLIENT >/dev/null 2>&1; do sleep 1; done"
    if [ $? -ne 0 ]; then
        echo "Warning: Client $CLIENT not reachable"
        continue
    fi
    echo "Setting up SSH for $CLIENT"
    sshpass -p "rootpass" ssh-copy-id -o StrictHostKeyChecking=no root@$CLIENT
    if [ $? -eq 0 ]; then
        echo "Successfully set up SSH for $CLIENT"
    else
        echo "Warning: Failed to set up SSH for $CLIENT"
        continue
    fi
done

# Test SCP to each client
for CLIENT in $CLIENTS; do
    echo "Testing SCP to $CLIENT"
    touch /tmp/test.txt
    scp -o StrictHostKeyChecking=no /tmp/test.txt root@$CLIENT:/tmp/
    if [ $? -eq 0 ]; then
        echo "SCP test to $CLIENT succeeded"
    else
        echo "Warning: SCP test to $CLIENT failed"
    fi
done

echo "SSH setup process completed"
exit 0