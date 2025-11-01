#!/bin/bash

echo "This is Borker ${2}NDBF_$1"

echo "updating the config file"


KAFKAIP_VAR="KAFKA_MAIN_${2}_IP"
KAFKAIP=${!KAFKAIP_VAR}


echo KAFKAIP: $KAFKAIP, KAFKAPORT: $KAFKAPORT
export KAFKAIP
envsubst < /app/confs/server.properties.template > /kafka_broker_instance/kafka_2.13-3.9.0/config/server.properties
echo "starting ${2}NDBF_$1 broker..."
cd /kafka_broker_instance/
python3 ${2}NDBF.py /kafka_broker_instance/kafka_2.13-3.9.0/
# sleep infinity
