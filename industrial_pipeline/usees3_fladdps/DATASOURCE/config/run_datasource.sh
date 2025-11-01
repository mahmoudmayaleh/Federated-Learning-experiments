#!/bin/bash

sleep 70
echo "This is Datasource instance$1 from DPS"
# sleep 15
echo "updating the config file"

INSTANCE_NUMBER=$1
DPS_NUMBER=1



INDBF_IP=10.10.1.12

KAFKAPORT=9092

SAMP_NUMBER=$(($FED_CONFIG_R * $FED_CONFIG_E * $SAMP_CONSTANT))
RAW_DATA_TOPIC_NAME="collected_cpu_topic_$1"

echo INDBF_IP: $INDBF_IP, KAFKAPORT: $KAFKAPORT
export INDBF_IP RAW_DATA_TOPIC_NAME SAMP_NUMBER INSTANCE_NUMBER DPS_NUMBER
echo "Number of samples: ${SAMP_NUMBER}"

envsubst < /app/confs/collect_produce_normal_template.py > /data_sources/collect_produce_normal.py
envsubst < /app/confs/collect_produce_anomaly_template.py > /data_sources/collect_produce_anomaly.py
cp /app/confs/stress_background.py /data_sources/stress_background.py

echo "starting Datasource $1 from DPS"
echo "Going to be connected to iNDBF: $INDBF_IP, topic name: $RAW_DATA_TOPIC_NAME"
cd /data_sources/

echo ">>>> [$(date)] Sending Normal data for training..."
python3 collect_produce_normal.py

echo "***********************************************"
echo ">>>> [$(date)] Starting alternating random/fixed  *****"
echo "***********************************************"
sleep 10

for i in {1..10}; do

    echo "***********************************************"
    echo ">>>> [$(date)] Starting round ${i}..."
    echo "***********************************************"


    echo "***********************************************"
    echo ">>>> [$(date)] Sending ABNORMAL data for a Maximum of 3 minutes..."
    echo "***********************************************"
    timeout 180 python3 collect_produce_anomaly.py
    
    echo "[CLEANUP] Killing all remaining Python processes..."
    pkill -f python
done
