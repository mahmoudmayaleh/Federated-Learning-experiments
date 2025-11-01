import json
import os
import platform
import stat

import gc

import numpy as np

import tensor_pb2 as tr
import tensor_pb2_grpc as tr_rpc
import grpc
import time
import ntplib
import pickle
import threading
import math
import sys


from kafka import KafkaProducer
# import torch as th
# import torch.nn as nn


from datetime import datetime

# as there a problem for pickling tf.keras, this is a work around
# from tf_keras_pickel_solution import make_keras_picklable
from Data_operations.DTAT_LAKE import get_data_batch, load_database_file_split

from Colors.print_colors import (print_purple,
                                 print_blue,
                                 print_cyne,
                                 print_orange)

import Data_operations.DataOperations as do
import Data_operations.DataAnalysis as da
import keras


from keras.models import model_from_json

from ProducerConsumer import Producer
from ProducerConsumer import OnlineConsumer

ROOT_DIR = os.getcwd()
ROOT_DIR = ROOT_DIR.split("/")
ROOT_DIR = ROOT_DIR[:len(ROOT_DIR) - 1]
ROOT_DIR = "/".join(ROOT_DIR)
sys.path.insert(0, f'{ROOT_DIR}/Utils')
import utils

gRPC_MESSAGE_MAX_LENGTH = 4194304 - 4  # 4 bytes are removed

CHECK_IS_ALIVE_INTERVAL = 3  # a heart pulse every # seconds

# MAX_MESSAGE_LENGTH = 1024 * 1024 * 20
MAX_MESSAGE_LENGTH = 1024 * 1024 * 390

WAIT_FOR_DS_SPLITS = 2 * 60  # 6  minutes
WAIT_BEFORE_CHECK = 2 * 60  # 5 minutes

PATIENCE_NO_ITERATIONS = 100  # iterations to lookup for dataset in the file system

END_TO_END_TIME_THRESHOLD = 5.0


CPU = do.CPU
MEMORY = do.MEMORY
NETWORK = do.NETWORK
DISK = do.DISK
ENB = do.ENB

from enum import Enum



class KerasVerbosity(Enum):
    silent = 0
    progress_bar = 1
    one_line = 2


class Client:
    # Added to handle real time

    # pipeline arguments

    _options = [
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)
    ]

    def __init__(self,
                 client_id,
                 client_str,
                 training_data_dir_path,
                 analysis_dir_path,
                 target="localhost:50055",
                 running_mode={},
                 rounds=0,
                 epochs_before_aggregation=0,
                 data_splits_count=0,  # no of days to go through
                 options=_options,
                 rsc_target="",
                 layer="",
                 data_lake_param={},
                 quantile=0.999,
                 number_of_participants=0,
                 test_data_percent=0.01,
                 no_training_samples=0,
                 lr=0.001,
                 data_pipeline_parameter={}
                 ):

        self.producer_sleep_time_interval = 0.00
        # self.connect_to_data_pipeline = False
        self.data_pipeline_parameter = data_pipeline_parameter

        print(f"connection {target}")
        # make_keras_picklable()  # WORKROUND
        self.layer = layer
        self.client_crd = tr.ClientCredentials(
            clientId=client_str,
            clientName=platform.uname()[1],  # node
            # TODO: Replace [currently used to tell server the the calculated no of rounds]
            clientToken=str(rounds * data_splits_count),  # change the name in proto files later on NOT USED RIGHT NOW
            machineName='Salah Bin',  # TODO:    Replace
            machineHW=platform.uname()[4],  # machine
        )
        self.rsc_target = rsc_target  # TODO: just for the sake of the out put: modify to be general
        self.client_id = client_id
        self.analysis_data_dir_path = analysis_dir_path
        self.client_count = 2

        self.channel = grpc.insecure_channel(target=target, options=options)

        self.conn = tr_rpc.FederatedLearningStub(self.channel)

        self.transmit_funcs = self.conn.TheMessageCPU
        
        
        """
        {CPU: self.conn.TheMessageCPU,
                               MEMORY: self.conn.TheMessageMemory,
                               NETWORK: self.conn.TheMessageNW,
                               DISK: self.conn.TheMessageDisk}
        """

        # self.full_dataset = full_dataset
        self.test_dataset = {CPU: None,
                             MEMORY: None,
                             NETWORK: None,
                             DISK: None,
                             ENB: None
                             }

        self.global_model = {CPU: None,
                             MEMORY: None,
                             NETWORK: None,
                             DISK: None,
                             ENB: None}
        self.local_model = {CPU: None,
                            MEMORY: None,
                            NETWORK: None,
                            DISK: None,
                             ENB: None}
        # For SCAFFOLD algorithm
        self.local_control_variate = {
            CPU: None,
            MEMORY: None,
            NETWORK: None,
            DISK: None,
            ENB: None}
        self.global_control_variate = {
            CPU: None,
            MEMORY: None,
            NETWORK: None,
            DISK: None,
            ENB: None}


        self.model_type = ''
        self.current_round = 0
        self.current_epochs_before_aggregation = 1
        self.total_rounds = rounds * data_splits_count
        self.rounds = rounds
        self.epochs_before_aggregation = epochs_before_aggregation
        # self.dataset_split_count = data_splits_count
        self.epochs = -1
        self.batch_size = -1
        self.segment_size = -1
        self.chunk_count = 1
        self.learning_rate = lr

        self.nnModel = None
        self.training_func = None
        self.training_data = {CPU: None,
                              MEMORY: None,
                              NETWORK: None,
                              DISK: None,
                             ENB: None
                              }

        self.evaluation_data = {CPU: None,
                                MEMORY: None,
                                NETWORK: None,
                                DISK: None,
                             ENB: None
                                }

        self.training_finished = False

        self.counter = 0
        self.print = {CPU: print_orange,
                      MEMORY: print_purple,
                      NETWORK: print_blue,
                      DISK: print_cyne,
                      ENB: print_orange}

        self.round_analysis = []
        self.full_analysis = {str(client_id): {}}
        self.output_folder = "output_file"
        self.model_sent_time = time.time()
        self.model_exchange_time = time.time()
        self.model_size = 0.0

        self.start_time = time.localtime()
        self.start_timestamp = time.time()

        self.training_time = -1

        self.running_mode = running_mode
        self.aggregation_method = "scaffold"
        self.incremental_MSE = np.array([])  # to concatenate MSE of all rounds
        self.quantile = quantile
        self.number_of_participants = number_of_participants

        # TODO: check sned_data function
        self.training_data_dir_path = training_data_dir_path
        self.test_data_percent = test_data_percent
        self.no_training_samples = no_training_samples

        self.init()

        self.training_data_shape = {}

        self.training_data_shape[self.rsc_target] = 0
        self.training_data[self.rsc_target] = []
        self.file_part = 1  # used in a loop

        self.total_data_samples = 0
        self.local_model_size = 0
        self.global_model_size = 0

        # =======================  DANGEROUS ZONE ==============================
        # Old Python limits recursion depyh to 1000
        # We extend it to 3000 to solve the problem in
        # train_predicted = model.predict(train_data)
        sys.setrecursionlimit(3000)


    def prepare_model_to_send(self, resource_name, local_model):
        # print(' In def get_model_chunks(self)')
        resource_name = resource_name.lower()
        s_model = pickle.dumps(local_model)  # * random.randint(1, 10))
        self.local_model_size = sys.getsizeof(s_model) 
        print(f'model size = {self.local_model_size}')

        return tr.Model(resource_name=resource_name, parameters=s_model, client=self.client_crd, act_time=time.time())

    def transmit_local_model(self, resource_name, local_model):
        t_start = time.time()
        self.counter += 1
        resource_name = resource_name.lower()

        l_model = self.prepare_model_to_send(resource_name, local_model)
        ts = time.time()
        lcl_t = f'{time.strftime("%H:%M:%S", time.localtime())}'
        self.print[resource_name](f'At {lcl_t} {resource_name} local trained model transmit starting  ')
        g_model = self.transmit_funcs(l_model)
        Ucl_t = f'{time.strftime("%H:%M:%S", time.localtime())}'
        self.print[resource_name](f'At {Ucl_t} {resource_name} local trained model transmit finish  ')
        self.model_exchange_time = time.time() - ts
        return g_model
    
    def update_local_model(self, resource_name,current_local_model):
        resource_name = resource_name.lower()
        
        s_model = pickle.dumps(current_local_model)
        model = tr.Model(resource_name=resource_name, parameters=s_model, client=self.client_crd, act_time=time.time())
        local_model = pickle.loads(model.parameters)
        self.global_model[resource_name].set_weights(local_model)
        return self.global_model[resource_name]
        
    def get_fl_global_model(self, resource_name, model):
        resource_name = resource_name.lower()
        self.global_model_size = sys.getsizeof(pickle.dumps(model)) 
        print("Dumpted model")
        print(self.global_model_size)
        g_model = model.parameters
        global_model_weights = pickle.loads(g_model)
        self.global_model[resource_name].set_weights(global_model_weights)
        return self.global_model[resource_name]

    def select_training_frmwrk(self, training_data, resource_name, g_model):
        resource_name = resource_name.lower()
        if self.model_type.lower() == 'torch':
            return self.train_torch_model(resource_name, g_model)
        elif self.model_type.lower() == 'keras':
            return self.train_keras_model(training_data, resource_name, g_model)

    def send_heartbeat(self):
        try:
            while not self.training_finished:
                rsp = self.conn.Heartbeat(self.client_crd)
                time.sleep(CHECK_IS_ALIVE_INTERVAL)


        except Exception:
            print("Gone ")
            print(type(Exception))  # the exception instance
            print(Exception.args)  # arguments stored in .args

    # First contact with server

    def register_client(self):
        rsp = tr.RegistrationParams(clientCredentials=self.client_crd, verbose=True, resource_name=self.rsc_target,
                                    client_count=self.client_count)

        print_blue("Try to register client with FL server ")
        t = time.time()
        rsp = self.conn.ClientRegistration(rsp)
        if rsp.registered:
            print(f'client  registered successfully in [{time.time() - t}] sec')
            self.get_init_global_model_and_init_params()

    def get_init_global_model_and_init_params(self):

        # try:
        print_orange(f'Getting global model and init params ... ')
        t_start = time.time()
        tmf = self.conn.TransmitInitializationParams(self.client_crd)
        self.model_exchange_time = time.time() - t_start
        print_orange(f'Global model and init params received in {self.model_exchange_time}')

        self.epochs = tmf.epochs
        self.batch_size = tmf.batchSize

        # Scaffold
        if self.aggregation_method.lower() == "scaffold":
            import tensorflow as tf
            self.training_func = lambda X, y, epochs, model, batch_size, lr, opt, local_cv, global_cv: keras_scaffold_training(
                model=model,
                train_data=tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size),
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                global_cv=global_cv,
                local_cv=local_cv
            )

        # FedProx
        elif self.aggregation_method.lower() == "fedprox":
            self.training_func = lambda X, y, epochs, model, batch_size, lr, opt, local_cv, global_cv: keras_lstm_fedprox_training(
                X=X,
                y=y,
                epochs=epochs,
                model=model,
                global_weights=model.get_weights(),
                mu=0.01,
                batch_size=batch_size,
                verbose=1
            )
        else:
            raise ValueError(f"[ERROR] Unknown aggregation method: {self.aggregation_method}")

        # Deserialize bytes into actual model object
        unpickled_model = pickle.loads(tmf.initialModel)
    
        # Set for all possible resource keys (both lowercase and uppercase)
        for key in ['cpu', 'memory', 'network', 'disk', 'enb', 'CPU', 'MEMORY', 'NETWORK', 'DISK', 'ENB']:
            self.global_model[key] = clone_keras_model(unpickled_model)
            self.global_control_variate[key] = [np.zeros_like(w) for w in unpickled_model.get_weights()]
            self.local_control_variate[key] = [np.zeros_like(w) for w in unpickled_model.get_weights()]

        self.model_type = tmf.modelType
        self.output_folder = tmf.folder_string
    
        t_end = time.time()
        total_time = t_end - t_start
    
        print_orange(f"Model size {sys.getsizeof(tmf.initialModel):,} bytes")
        self.model_size = sys.getsizeof(tmf.initialModel)
        self.global_model_size = self.model_size

    def get_segment_data(self, training_data, resource_name=None):
        inter_round = self.current_round % self.rounds
        resource_name = resource_name.lower()

        print_blue("training data shape: ", training_data.shape)

        self.training_data_shape[resource_name] = training_data.shape

        if len(self.running_mode.keys()) > 0:
            if self.running_mode["mode"].lower() == "batch":

                segment_size = self.segment_size  # self.running_mode["batch_size"]
                train_data_segment = get_data_batch(training_data, segment_size, inter_round, self.rounds)

                print_purple(f"data size: {sys.getsizeof(self.training_data[resource_name]):,} byte")

                print_purple(f"training data segment: {sys.getsizeof(train_data_segment):,} byte")

                print_orange(f"training sigment shape: {train_data_segment.shape} of  {training_data.shape}")
                return train_data_segment
            
            
    def train_keras_model(self, train_data_segment, resource_name, k_model):
        resource_name = resource_name.lower()
        if k_model is None:
            raise ValueError(f"[ERROR] Keras model for resource '{resource_name}' is None. Check model initialization.")
        # if self.current_round <= self.total_rounds - 1:
        print_cyne(f'{resource_name} Starting ->[Round  : {self.current_round}/{self.total_rounds}]')
        print_cyne(f'{resource_name} Starting ->[Inter round  : {self.current_epochs_before_aggregation}/{self.epochs_before_aggregation}]')

        is_file_loaded = False


        x_train = train_data_segment
        y_train = x_train

        print_cyne('Got From Server')


        # insure they have same no of samples: utilize the larger
        if x_train.shape[0] < y_train.shape[0]:
            y_train = y_train[:x_train.shape[0]]
        elif x_train.shape[0] > y_train.shape[0]:
            x_train = x_train[:y_train.shape[0]]

        x_train = np.flip(x_train, axis=0)
        y_train = np.flip(y_train, axis=0)

        vr = KerasVerbosity.progress_bar.value
        vr = KerasVerbosity.one_line.value
        vr = KerasVerbosity.silent.value

        s_t = time.time()
        st_t = time.strftime("%H:%M:%S", time.localtime())
        print(f'Training started at:{st_t}')

        st = datetime.now()
        updated_weights, updated_local_cv = self.training_func(
            x_train, y_train,
            self.epochs,
            k_model,
            self.batch_size,
            self.learning_rate,
            "Adam",
            self.local_control_variate[resource_name],
            self.global_control_variate[resource_name]
        )
        k_model.set_weights(updated_weights)
        self.local_control_variate[resource_name] = updated_local_cv

        '''
        else:
            # Default training (FedAvg/FedProx)
            self.training_func(x_train, y_train,
                            self.epochs,
                            k_model,
                            batch_size=self.batch_size,
                            verbose=vr)
		

            edt = datetime.now()

            ed_t = time.strftime("%H:%M:%S", time.localtime())
            print(f'Trining ended at:{ed_t}')
            e_t = time.time()
            t_t = e_t - s_t
            self.training_time = str(edt - st)   #get_formatted_elapsed_time(s_t)
            t_t = time.strftime("%H:%M:%S", time.gmtime(t_t))
            print(f'Training time {self.training_time}')

            print("=" * 80)
            return k_model
		'''
        return k_model  # <-- ADD THIS LINE
    def inter_round_data_analysis(self, x_train, resource_name, k_model, ds_mnt):
        resource_name = resource_name.lower()
        train_MSE, round_threshold = self.get_data_analysis(resource_name, k_model, x_train,
                                                            self.test_dataset[resource_name])

        self.incremental_MSE = np.concatenate((self.incremental_MSE, train_MSE))


        # Added Later
        # threshold = np.quantile(self.incremental_MSE, self.quantile)
        threshold = np.mean(self.incremental_MSE)

        print("=" * 80)

        dtan = {"client_id": self.client_id,
                "round_no": int(self.current_round),
                "training_time (secs)": self.training_time,
                # "train_MSE": train_MSe,
                "incremental_MSE_Min": np.min(self.incremental_MSE),
                "incremental_MSE_Median": np.median(self.incremental_MSE),
                "incremental_MSE_Max": np.max(self.incremental_MSE),
                "average_MSE": np.mean(train_MSE),
                # "test_MSE_loss": test_MSE_loss,
                "round_threshold": round_threshold,
                "threshold": threshold,
                "model_round_trip": self.model_exchange_time
                }
        
        dir_name = f'{self.analysis_data_dir_path}/{self.layer}'
        fullpath = os.path.join(dir_name, f'runtime_analysis.json')
        fullpath_history = os.path.join(dir_name, f'runtime_analysis_history.json')
        do.save_json_to_file(dtan, fullpath)
        self.round_analysis.append(dtan)
        appends_to_json_file(dtan, fullpath_history)
    def train_torch_model(self, resource_name, nn_model):
        resource_name = resource_name.lower()
        inputDim = 1  # takes variable 'x'
        outputDim = 1  # takes variable 'y'
        learningRate = 0.01

        # model = nn_model(inputDim, outputDim)
        model = nn_model

        ##### For GPU #######
        if th.cuda.is_available():
            model.cuda()

        criterion = nn.MSELoss()
        optimizer = th.optim.SGD(model.parameters(), lr=learningRate)


        x_train = self.training_data[resource_name]
        y_train = self.evaluation_data[resource_name]

        local_weights = model.state_dict()

        print_cyne('Got From Server')
        for key in model.state_dict().keys():
            print_cyne(f'{key} = {local_weights[key]}')

        if not self.training_finished:
            self.training_func(x_train, y_train, self.epochs, model, optimizer, criterion)
            # self.local_model = model
            self.local_model[resource_name] = model.state_dict()

        print_cyne('Will Send To Server')
        for key in model.state_dict().keys():
            print(print_cyne(f'{key} = {local_weights[key]}'))

        """""
        self.global_model used for training and the result is saved in self.local_model 
        """""

    # TODO Be CHANGED FOR model JSON AND WEIGHTs
    def ready_to_start(self, resource_name):
        resource_name = resource_name.lower()

        response = self.conn.StartTraining(self.client_crd)
        self.start_timestamp = time.time()
        self.start_time = time.localtime()
        print(f'Ready to start = {response}   {time.strftime("%H:%M:%S", self.start_time)}')
        if response.response:
            # self.print[resource_name]("----> ", resource_name)
            print_purple("GET + RESPONSE")
            g_model = self.global_model[resource_name]
            # ----------------------------------------------
            init_model = g_model
            self.global_model_size = sys.getsizeof(init_model) 

            self.global_model[resource_name] = init_model

            self.do_training(db_splits_count=math.ceil(self.total_rounds / self.rounds), resource_name=resource_name,
                             init_model=init_model)
        else:
            print('Can not start training right now. Try to start another time')

    def feedback_type(self):
        print('feedback_type')
        

    # ML model summery: layers, neurons
    # TODO MAKE IT GENERAL for all resources
    def get_model_summery(self, resource_name):
        print('model_summery: ')
        # print_cyne(pickle.loads(self.global_model[resource_name]))

    # what extra params
    def get_required_params(self):
        print('get_required_params')

    # check files are in the file system, OR wait
    def check_file_exist(self, file):
        return True #TODO, debugings
        i = 0
        answer = False
        while True:
            if i > PATIENCE_NO_ITERATIONS:
                break

            if os.path.exists(file):
                j = 0
                for i in range(120):
                    if not os.access(file, os.R_OK):
                        time.sleep(5)

                answer = True
                break  # while loop
            else:
                time.sleep(WAIT_BEFORE_CHECK)  # seconds

            print(f"file exists check: {i}")

            i += 1

        return answer

    def load_client_dataset(self, training_data_splits_path, rsc_target, day_index, client_idx, no_training_samples):
        print_orange("Waiting for dataset files to be loaded")

        training_data_dir_path = f'{training_data_splits_path}/{rsc_target}_{day_index}_{client_idx}'
        training_data = []

        # wait longer for file splitting at the start
        if day_index == 1:
            time.sleep(WAIT_FOR_DS_SPLITS)  # Wait till the Splitter writes out the files to the file system

        if self.check_file_exist(training_data_dir_path):
            training_data, _ = load_database_file_split(training_data_dir_path,
                                                        day_index,
                                                        no_training_samples)


            self.segment_size = training_data.shape[0] / self.rounds

            print("segment_size: ", self.segment_size)
        else:
            print_purple("Sorry, no dataset file is provided in the file system.")
            exit()  # TODO pay attention

        return training_data

    # send if there are more params needed
    def send_extra_params(self):
        pass

    def get_data_analysis(self, resource_name, model, train_x, test_x, target=""):
        resource_name = resource_name.lower()
        train_MSe, threshold = da.get_tarin_MSE_and_threshold(model, train_x, test_x)

        #TODO
        # change resource_name
        print(f'Average MSE ({target} gnb): {np.mean(train_MSe)}')

        return train_MSe, threshold

    # start training
    def start_training(self, rsc_name=""):

        self.ready_to_start(self.rsc_target)

    def produce_json_result(self, resource_name):
        resource_name = resource_name.lower()
        self.print[resource_name]('training_finished')


        print_cyne(f"Analysis for client[{str(self.client_id)}]:\n", self.full_analysis)
        print_cyne(f"Analysis for client[{str(self.client_id)}]:\n", self.round_analysis)



        mode = self.running_mode['mode']
        if mode.upper() == "batch".upper():
            segment_size = self.segment_size  # self.running_mode['batch_size']
        else:
            segment_size = "-"

        training_settings = {
            "epochs": self.epochs,
            "mode": mode,
            "Rounds": self.total_rounds
        }
        w_t = get_formatted_elapsed_time(self.start_timestamp, True)

        self.full_analysis[str(self.client_id)]["node_name"] = f"{os.uname()[1]}  {os.uname()[2]}"
        self.full_analysis[str(self.client_id)]["resource_target"] = self.rsc_target
        self.full_analysis[str(self.client_id)]["training_settings"] = training_settings
        self.full_analysis[str(self.client_id)]["model_info"] = {"model_type": self.model_type.lower(),
                                                                 "model_size": f'{self.model_size:,} bytes'}
        self.full_analysis[str(self.client_id)]["FL total_training_time"] = \
            {"total": w_t,
             "start time": time.strftime("%H:%M:%S", self.start_time),
             "end time": time.strftime("%H:%M:%S", time.localtime())}
        self.full_analysis[str(self.client_id)][f"rounds_anl"] = self.round_analysis

        threshold = np.quantile(self.incremental_MSE, self.quantile)
        self.full_analysis[str(self.client_id)][f"final_threshold_[{self.quantile}_quantile]"] = threshold
        dir_path = do.create_dir(f"{self.analysis_data_dir_path}/{self.layer}", f"{self.rsc_target}{self.output_folder}")

        fullpath = os.path.join(dir_path, f'{str(self.client_id)}.json')
        do.save_json_to_file(self.full_analysis, fullpath)


        self.training_finished = True

    def init(self):
        # NOTE(gRPC Python Team): .close() is possible on a channel and should be
        # used in circumstances in which the with statement does not fit the needs
        # of the code.
        self.register_client()

        print("-------------- Ready --------------")

    def do_training(self, db_splits_count, resource_name, init_model):
        resource_name = resource_name.lower()
        # DEBUG: Print model keys and value
        print(f"DEBUG: resource_name={resource_name}, self.global_model keys={list(self.global_model.keys())}")
        print(f"DEBUG: self.global_model[{resource_name}] = {self.global_model[resource_name]}")
        # Force model assignment if needed
        if self.global_model[resource_name] is None:
            print(f"DEBUG: Forcing model assignment for {resource_name}")
            self.global_model[resource_name] = clone_keras_model(init_model)
        first_pass = True
        file_loaded_dic = {}
        g_model = init_model
        for day in range(1, db_splits_count + 1):

            print_purple(f"loading database files split [{day}]")

            t_fl = time.time()
            s_time = time.strftime("%H:%M:%S", time.localtime())


            is_file_loaded = True

            e_time = time.strftime("%H:%M:%S", time.localtime())
            time_fld = get_formatted_elapsed_time(t_fl)
            file_loaded_dic[day] ={'real time data'}# {"split_index": day, "start_time": s_time, "end_time": e_time, "duration": time_fld}
            
            producer = Producer(self.data_pipeline_parameter, self.producer_sleep_time_interval)
            consumer = OnlineConsumer(END_TO_END_TIME_THRESHOLD, self.epochs_before_aggregation)
            producer.start()
            consumer.start()
            
            for round_ in range(self.rounds):
                # print_blue("Get data from Data Pipeline System")
                training_round_start = time.time()
                local_model = g_model
                epoch_ = 1
                count_discarded_data = 0
                deployment_state ={}
                training_time = 0
                learning_time = 0
                self.current_round = round_+1
                while epoch_  <= self.epochs_before_aggregation:
                    
                    self.current_epochs_before_aggregation = epoch_
                    timed_training_data = consumer.get_training_data()
                    segment_training_data = timed_training_data.get("timed_training")
                    column_names = timed_training_data.get("column_names")
                    column_values = timed_training_data.get("column_values")
                    to_discard = timed_training_data.get("to_discard")
                    preprocessing_time = timed_training_data.get("preprocessing_time")
                    end_to_end_time = timed_training_data.get("end_to_end_time")
                    if to_discard:
                        
                        count_discarded_data = count_discarded_data + 1
                        deployment_state["preprocessing_time"]=preprocessing_time
                        deployment_state["end_to_end_time"]=end_to_end_time
                        deployment_state["learning_time"]=learning_time
                        deployment_state["training_time"]=training_time
                        print_purple(f"Set discarded data for inter_round  [{epoch_}] where the end-to-end time is {end_to_end_time}")
                        #epoch_ = epoch_ + 1
        
                    training_epoch_start = time.time()
                    print(segment_training_data.shape)

                    if first_pass:
                        csv_file_name = 'timestamp-values_'+str(self.rounds)+'.csv'
                        utils.delete_csv_file(csv_file_name)
                        first_pass = False
                        l_model = self.select_training_frmwrk(segment_training_data, resource_name,self.global_model[resource_name])
                    else:
                        l_model = self.select_training_frmwrk(segment_training_data, resource_name, local_model)

                    local_model = l_model
                    training_epoch_end = time.time()
                    if training_time > (training_epoch_end-training_epoch_start):
                        training_time = training_epoch_end-training_epoch_start
                    column_names.append('training_epoch_start')
                    column_names.append('training_epoch_end')
                    column_values.append(training_epoch_start)
                    column_values.append(training_epoch_end)
                    column_values.append(count_discarded_data)
                    column_names.append('training_round_start')
                    column_names.append('training_round_end')
                    column_values.append(training_round_start)
                    if epoch_ < self.epochs_before_aggregation:
                        column_values.append('')
                    else:
                        learning_time = training_epoch_end-training_epoch_start
                        training_round_end = time.time()
                        column_values.append(training_round_end)
                        deployment_state["preprocessing_time"]=preprocessing_time
                        deployment_state["end_to_end_time"]=end_to_end_time
                        deployment_state["learning_time"]=learning_time
                        deployment_state["training_time"]=training_time

                       
                    column_names.append('local_model_size')
                    column_names.append('global_model_size')
                    column_values.append(self.local_model_size)
                    column_values.append(self.global_model_size)
                    csv_file_name = 'timestamp-values_'+str(self.rounds)+'.csv'
                    utils.write_to_csv_file(csv_file_name,column_names,column_values)
                    epoch_ =  epoch_ +1
                    count_discarded_data = 0
                
                g_model_ = self.transmit_local_model(resource_name=resource_name, local_model=local_model.get_weights())
                #unpickle parameters and model
                g_model = self.get_fl_global_model(resource_name=resource_name, model=g_model_)
                
                self.inter_round_data_analysis(segment_training_data, resource_name, g_model, file_loaded_dic)
                learning_time = self.model_exchange_time + deployment_state.get("learning_time")
                deployment_state["learning_time"]=learning_time
                self.publish_to_topic(deployment_state)
            file_loaded_dic = {}

            gc.collect()
            consumer.join()
            producer.join()
            # Save training process outputs
        self.produce_json_result(resource_name)
        

    
    
    def publish_to_topic(self, json_data):
        
        topic_name="deployment_state_topic"
        json_dumped = json.dumps(json_data)
        producer = KafkaProducer(bootstrap_servers=self.data_pipeline_parameter.get("bootstrap_server"), api_version=(0, 10))
        producer.send(topic_name, value=json_dumped.encode('utf-8'))   
       
    

from tensorflow.keras.callbacks import EarlyStopping





    
def current_time():
        
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
    
def keras_lstm_training(X, y, epochs, model, batch_size=64, verbose=1):
    # fit model

    my_callbacks = [
        EarlyStopping(monitor='loss', patience=4),
    ]
    model.fit(X, y, epochs=epochs,
              batch_size=batch_size,
              verbose=verbose,
              callbacks=[EarlyStopping(monitor='loss', patience=4)],
              shuffle=False)
    return model.get_weights(), None

def keras_lstm_fedprox_training(X, y, epochs, model, global_weights, mu=0.01, batch_size=64, verbose=1):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping

    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()

    # Create early stopping
    early_stop = EarlyStopping(monitor='loss', patience=4)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            predictions = model(X, training=True)
            base_loss = loss_fn(y, predictions)

            # FedProx regularization term
            prox_term = tf.add_n([tf.reduce_sum(tf.square(a - b)) for a, b in zip(model.get_weights(), global_weights)])
            loss = base_loss + (mu / 2) * prox_term

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.numpy()}")

def keras_scaffold_training(model, train_data, epochs, batch_size, lr, global_cv, local_cv):
    import tensorflow as tf
    import numpy as np

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    # Use model.loss if it's a function, else default to MSE
    loss_fn = model.loss if callable(model.loss) else tf.keras.losses.MeanSquaredError()

    # Store original weights
    original_weights = model.get_weights()

    for epoch in range(epochs):
        for x_batch, y_batch in train_data:
            with tf.GradientTape() as tape:
                preds = model(x_batch, training=True)
                loss = loss_fn(y_batch, preds)  # <-- This is a scalar tensor
            grads = tape.gradient(loss, model.trainable_weights)

            # Corrected gradients: g - c + c_i
            corrected_grads = [
                g - gc + lc
                for g, gc, lc in zip(grads, global_cv, local_cv)
            ]

            optimizer.apply_gradients(zip(corrected_grads, model.trainable_weights))

    # Compute updated control variate delta
    new_weights = model.get_weights()
    delta_weights = [new - old for new, old in zip(new_weights, original_weights)]
    updated_local_cv = [lc - gc + (dw / (lr * epochs)) for lc, gc, dw in zip(local_cv, global_cv, delta_weights)]

    return model.get_weights(), updated_local_cv



def get_formatted_elapsed_time(t, dated=False):
    if dated:
        dt = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(t)
        return f"{dt}"
    else:
        return time.strftime("%H:%M:%S", time.gmtime(time.time() - t))
    

def appends_to_json_file(data, filepath):
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Load existing data if file exists, else start with empty list
    try:
        with open(filepath, 'r') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = []
    
    # Ensure existing_data is a list
    if not isinstance(existing_data, list):
        existing_data = [existing_data]
    
    # Append new data
    existing_data.append(data)
    
    # Save updated data back to file
    with open(filepath, 'w') as f:
        json.dump(existing_data, f, indent=4)

rsc_name_dir = {"cpu": "cpu", "memory": "memory", "network": "network", "disk": "disk", "enb": "enb"}
def clone_keras_model(model):
    from keras.models import model_from_json
    import keras
    model_config = model.to_json()
    cloned = model_from_json(model_config)
    cloned.set_weights(model.get_weights())
    return cloned
