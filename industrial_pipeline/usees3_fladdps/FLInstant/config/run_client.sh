#!/bin/bash
sleep 45
echo "This is the Client-$1"

CLIENT_INDEX=$1
export CLIENT_INDEX


KAFKA_IP=10.10.1.10
KAFKAPORT=${KAFKAPORT:-9092}

echo "Watiting for others to be setup ..."

echo "Updating server config file ..."
envsubst < /app/confs/config.json.template > /in_network_federaed_learning_for_anomaly_detection/FLConfig/config.json
envsubst < /app/confs/client_driver_current.py.template > /in_network_federaed_learning_for_anomaly_detection/FLClients/client_driver_current.py

echo "starting Client-$1 container ..."
echo "Going to connect to eNDBF: $KAFKA_IP, on topic: preprocessed_cpu_topic_$1"
cd /in_network_federaed_learning_for_anomaly_detection/FLClients
python3 client_driver_current.py ${KAFKA_IP}:${KAFKAPORT} preprocessed_cpu_topic_$1
