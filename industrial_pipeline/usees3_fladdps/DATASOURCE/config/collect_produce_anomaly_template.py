import psutil
from kafka import KafkaProducer
import time
import json
import uuid

import multiprocessing
import pandas
from scipy.stats import linregress
import random
from joblib import Parallel, delayed
import ntplib
from stress_background import start_stress_in_background

# Generate a unique identifier for the source
source_id = str(uuid.uuid4())[:8]

bootstrap_servers = "${INDBF_IP}:${KAFKAPORT}"


topic_name = "${RAW_DATA_TOPIC_NAME}"

producer = KafkaProducer(bootstrap_servers=bootstrap_servers, api_version=(0, 10))

sample_counting = 1


# Publish metrics to Kafka
def publish_to_topic(producer_instance, topic_name, metrics_json):
    
    try:
        value_bytes = bytes(metrics_json, encoding='utf-8')
        producer_instance.send(topic_name, value=value_bytes)
        producer_instance.flush()

    except Exception as ex:
        print('Exception in publishing metrics to topic')
        print(str(ex))


def get_ntp_server_time():
    
    wait_for_response = True
    retry = 0
    while wait_for_response:
        
        try:
            ntp_client = ntplib.NTPClient()
            response = ntp_client.request('ntp.cnam.fr')
            wait_for_response = False
        except (ntplib.NTPException) as e:
            print('NTP client request error:', str(e))
            retry = retry + 1
            time.sleep(1)
        
        if retry == 3:
            wait_for_response = False
            
    return response.tx_timestamp
    
def compute_regression_in_parallel_to_simulate_load():
    
    start = True
    sleep_times = [500, 750, 1000,  250, 100]
    while start:
        
        index = random.randint(0, len(sleep_times)-1)
        sleep_time = (sleep_times[index]/1000)
        time.sleep(sleep_time)
        print('start compute_regression_in_parallel')
        random_list = []
        max1 = random.randint(100, 500)
        max2 = random.randint(10, 20)
        for i in range(1,max1):
            for j in range(1,max2):
                random_list.append({"id":i,"date":j,"value":random.random()})
                
    
        df = pandas.DataFrame(random_list)
    
        df = applyParallel(df.groupby('id'), compute_regression)
        
        print('stop compute_regression_in_parallel')
        

    

def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
    return pandas.concat(retLst)

def compute_regression(df):
    result = {}

    (slope,intercept,rvalue,pvalue,stderr) = linregress(df.date,df.value)
    result["slope"] = [slope]
    result["intercept"] = [intercept]
    return pandas.DataFrame(result)

def collect_metrics(source_id):
    metrics = {
        "source_id": source_id,
        "timestamp_col": str(time.time()),
        "cpu_times_user": psutil.cpu_times().user,
        "cpu_times_nice": psutil.cpu_times().nice,
        "cpu_times_system": psutil.cpu_times().system,
        "cpu_times_idle": psutil.cpu_times().idle,
        "cpu_times_iowait": psutil.cpu_times().iowait,
        "cpu_times_irq": psutil.cpu_times().irq,
        "cpu_times_softirq": psutil.cpu_times().softirq,
        "cpu_times_steal": psutil.cpu_times().steal,
        "cpu_times_guest": psutil.cpu_times().guest,
        "cpu_times_guest_nice": psutil.cpu_times().guest_nice,        
        "cpu_utilization": psutil.cpu_percent(),
        "cpu_stats_ctx_switches": psutil.cpu_stats().ctx_switches,
        "cpu_stats_interrupts": psutil.cpu_stats().interrupts,
        "cpu_stats_soft_interrupts": psutil.cpu_stats().soft_interrupts,
        "cpu_stats_syscalls": psutil.cpu_stats().syscalls,
        "memory_usage_total": psutil.virtual_memory().total,
        "memory_usage_available": psutil.virtual_memory().available,
        "memory_usage_percent": psutil.virtual_memory().percent,
        "memory_usage_used": psutil.virtual_memory().used,
        "memory_usage_free": psutil.virtual_memory().free,
        "memory_swap_total": psutil.swap_memory().total,
        "memory_swap_used": psutil.swap_memory().used,
        "memory_swap_free": psutil.swap_memory().free,
        "memory_swap_percent": psutil.swap_memory().percent,
        "memory_swap_sin": psutil.swap_memory().sin,
        "memory_swap_sout": psutil.swap_memory().sout
    }
    return metrics
    
def collect_cpu_metrics(source_id):
    cpu_metrics = {
        "source_id": source_id,
        "timestamp_col": str(time.time()),
        "cpu_times_user": psutil.cpu_times().user,
        "cpu_times_nice": psutil.cpu_times().nice,
        "cpu_times_system": psutil.cpu_times().system,
        "cpu_times_idle": psutil.cpu_times().idle,
        "cpu_times_iowait": psutil.cpu_times().iowait,
        "cpu_times_irq": psutil.cpu_times().irq,
        "cpu_times_softirq": psutil.cpu_times().softirq,
        "cpu_times_steal": psutil.cpu_times().steal,
        "cpu_times_guest": psutil.cpu_times().guest,
        "cpu_times_guest_nice": psutil.cpu_times().guest_nice,        
        "cpu_utilization": psutil.cpu_percent(),
        "cpu_stats_ctx_switches": psutil.cpu_stats().ctx_switches,
        "cpu_stats_interrupts": psutil.cpu_stats().interrupts,
        "cpu_stats_soft_interrupts": psutil.cpu_stats().soft_interrupts,
        "cpu_stats_syscalls": psutil.cpu_stats().syscalls
    }
    return cpu_metrics

def collect_memory_metrics(source_id):
    memory_metrics = {
        "source_id": source_id,
        "timestamp": str(time.time()),
        "memory_usage_total": psutil.virtual_memory().total,
        "memory_usage_available": psutil.virtual_memory().available,
        "memory_usage_percent": psutil.virtual_memory().percent,
        "memory_usage_used": psutil.virtual_memory().used,
        "memory_usage_free": psutil.virtual_memory().free,
        "memory_swap_total": psutil.swap_memory().total,
        "memory_swap_used": psutil.swap_memory().used,
        "memory_swap_free": psutil.swap_memory().free,
        "memory_swap_percent": psutil.swap_memory().percent,
        "memory_swap_sin": psutil.swap_memory().sin,
        "memory_swap_sout": psutil.swap_memory().sout
    }
    return memory_metrics

def collect_publish(producer):
    
    number_of_samples = 1500 
    sampling_rate = 100/1000
    start = True
    while start:
        try:

            metrics_dict = collect_metrics(source_id)

            timestamp_pub = str(get_ntp_server_time())
            
            metrics_dict["timestamp_pub"] = timestamp_pub

            # Publish metrics to Kafka
            metrics_json = json.dumps(metrics_dict)
            publish_to_topic(producer, topic_name, metrics_json)
            time.sleep(sampling_rate)
            global sample_counting
            sample_counting = sample_counting + 1
          
                
            if sample_counting == number_of_samples:
                start = False

                
                
        except KeyboardInterrupt:
            # Handle keyboard interrupt 
            print("Keyboard Interrupt, exiting...")
            break


if __name__ == '__main__':
    
    # Start stressing in background
    stop_event, thread = start_stress_in_background()

    print("Starting kafka metric collection script...")
    # Collect and publish metrics
    print("Collecting and Publishing metrics...") 
    collect_publish(producer)
    print("Stop Collecting and Publishing metrics...")

    # Stop Background stress
    stop_event.set()
    thread.join()
