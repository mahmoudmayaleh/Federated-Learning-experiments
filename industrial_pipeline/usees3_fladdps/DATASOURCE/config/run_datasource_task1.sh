#!/bin/bash

sleep 70
echo "This is Datasoruce instance$1 from DPS$2"
# sleep 15
echo "updating the config file"

INSTANCE_NUMBER=$1
DPS_NUMBER=$2

INDBF_IP_VAR="KAFKA_i${2}_IP"
INDBF_IP=${!INDBF_IP_VAR}
RAW_DATA_TOPIC_NAME="collected_cpu_topic_$2"

echo INDBF_IP: $INDBF_IP, KAFKAPORT: $KAFKAPORT
export INDBF_IP RAW_DATA_TOPIC_NAME INSTANCE_NUMBER DPS_NUMBER

envsubst < /app/confs/collect_produce_normal_task1.py > /data_sources/collect_produce_normal.py

echo "starting Datasource $1 from DPS$2"
echo "Going to be connected to iNDBF: $INDBF_IP, topic name: $RAW_DATA_TOPIC_NAME"
cd /data_sources/

echo ">>>> [$(date)] Sending Normal data for training..."
python3 collect_produce_normal.py
