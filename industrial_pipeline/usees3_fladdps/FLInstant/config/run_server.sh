#!/bin/bash
sleep 30
echo "This is the server"

# Run SSH setup in background
echo "Starting SSH setup for clients..."
chmod +x /app/confs/setup_ssh.sh
/app/confs/setup_ssh.sh flclient_1 flclient_2 flclient_3 flinference_1 flinference_2 flinference_3 &
SSH_PID=$!

echo "Updating server config file ..."
envsubst < /app/confs/config.json.template > /in_network_federaed_learning_for_anomaly_detection/FLConfig/config.json

echo "starting server container ..."

cd /in_network_federaed_learning_for_anomaly_detection/FLServer
python3 server_drive_current.py
