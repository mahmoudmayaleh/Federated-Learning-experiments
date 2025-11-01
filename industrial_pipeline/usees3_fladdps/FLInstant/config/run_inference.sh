#!/bin/bash
sleep 55
echo "This is the Inference-$1"

KAFKA_IP_VAR=10.10.1.10
KAFKA_IP=10.10.1.10
echo "Watiting for others to be setup ..."
# sleep 15

echo "Updating server config file ..."
envsubst < /app/confs/config.json.template > /in_network_federaed_learning_for_anomaly_detection/FLConfig/config.json

echo "starting Inference-$1 container ..."
echo "Going to connect to eNDBF: $KAFKA_IP, on topic: preprocessed_cpu_topic_i_$1"
cd /in_network_federaed_learning_for_anomaly_detection/FLInference
python3 anomaly_detection.py ${KAFKA_IP}:${KAFKAPORT} preprocessed_cpu_topic_i_$1
