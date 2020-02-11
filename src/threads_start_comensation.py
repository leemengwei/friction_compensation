import threading
import time,os,sys
import glob
from IPython import embed
import numpy as np
import NN_deploy
import torch
import data_stuff
import warnings
warnings.filterwarnings("ignore")

class Worker():
    def __init__(self, model_name, time_num=None):
        self.model = torch.load(model_name, map_location=torch.device("cpu"))
        self.normer = data_stuff.normalizer(np.zeros((input_dim, buffer_length)))
        print("Model Intialized.")
    def do(self, planned_data, SOMETHING=None):
        #Normalization:
        normed_data_X = self.normer.normalize_X(planned_data)
        #Forward to get output:
        inputs = torch.FloatTensor(normed_data_X.T)
        outputs = self.model.cpu()(inputs).detach().numpy()
        outputs = outputs.reshape(-1)
        #Denormalization:
        result = self.normer.denormalize_Y(outputs)
        return result

class thread_manager(threading.Thread):
    def __init__(self, worker, SOMETHING):
        threading.Thread.__init__(self)
        self.worker = worker
        self.planned_data = None
        self.SOMETHING = SOMETHING
        self.result = []
    def set_values(self, planned_data):
        self.planned_data = planned_data
    def run(self):
        self.result = self.worker.do(self.planned_data, self.SOMETHING)

def python_get_plans():
    planned_data1 = np.ones((input_dim, buffer_length))
    planned_data2 = np.ones((input_dim, buffer_length))
    planned_data3 = np.ones((input_dim, buffer_length))
    planned_data4 = np.ones((input_dim, buffer_length))
    planned_data5 = np.ones((input_dim, buffer_length))
    planned_data6 = np.ones((input_dim, buffer_length))
    return planned_data1, planned_data2, planned_data3, planned_data4, planned_data5, planned_data6

class workers_cluster():
    def __init__(self, model_names):
        self.worker1 = Worker(model_names[1])
        self.worker2 = Worker(model_names[2])
        self.worker3 = Worker(model_names[3])
        self.worker4 = Worker(model_names[4])
        self.worker5 = Worker(model_names[5])
        self.worker6 = Worker(model_names[6])
        sys.stdout.flush()
        self.workers_list = [ \
                self.worker1, \
                self.worker2, \
                self.worker3, \
                self.worker4, \
                self.worker5, \
                self.worker6, \
                ]

def algorithm_compensation(workers_list, \
                     planned_data1, planned_data2, planned_data3, planned_data4, planned_data5, planned_data6):
    SOMETHING = None
    #Axis 1~6:
    _thread1 = thread_manager(workers_list[0], SOMETHING)
    _thread2 = thread_manager(workers_list[1], SOMETHING)
    _thread3 = thread_manager(workers_list[2], SOMETHING)
    _thread4 = thread_manager(workers_list[3], SOMETHING)
    _thread5 = thread_manager(workers_list[4], SOMETHING)
    _thread6 = thread_manager(workers_list[5], SOMETHING)
    _thread1.set_values(planned_data1)
    _thread2.set_values(planned_data2)
    _thread3.set_values(planned_data3)
    _thread4.set_values(planned_data4)
    _thread5.set_values(planned_data5)
    _thread6.set_values(planned_data6)
    #Wait for each other.   TODO: To rethink if it's necessary
    #_thread1.start()
    #_thread2.start()
    #_thread3.start()
    #_thread4.start()
    #_thread5.start()
    #_thread6.start()
    #_thread1.join()
    #_thread2.join()
    #_thread3.join()
    #_thread4.join()
    #_thread5.join()
    #_thread6.join()
    _thread1.run()
    _thread2.run()
    _thread3.run()
    _thread4.run()
    _thread5.run()
    _thread6.run()
    #Gather results:
    #results = np.array([_thread1.result, _thread2.result, _thread3.result, _thread4.result, _thread5.result, _thread6.result])
    results = None
    return results

if __name__ == "__main__":
    #Config:
    buffer_length = 6
    input_dim = 25
    model_names = [ "None",
        "/mfs/home/limengwei/friction_compensation/models_save/NN_weights_best_all_1",
        "/mfs/home/limengwei/friction_compensation/models_save/NN_weights_best_all_2",
        "/mfs/home/limengwei/friction_compensation/models_save/NN_weights_best_all_3",
        "/mfs/home/limengwei/friction_compensation/models_save/NN_weights_best_all_4",
        "/mfs/home/limengwei/friction_compensation/models_save/NN_weights_best_all_5",
        "/mfs/home/limengwei/friction_compensation/models_save/NN_weights_best_all_6",
    ]
    interval_t = 8  #ms
    total_t_consumption = 0
    num_of_test = 1000
    t_history = []
    #Get model:
    print("Getting model...")
    workers = workers_cluster(model_names) 
    workers_list = workers.workers_list
    
    #Get data：
    planned_data1, planned_data2, planned_data3, planned_data4, planned_data5, planned_data6 = python_get_plans()

    #Run:
    print("Computing...")
    for i in range(num_of_test):
        print(i)
        tic = time.time()
        #Compute:
        results = algorithm_compensation(workers_list, planned_data1, planned_data2, planned_data3, planned_data4, planned_data5, planned_data6)
        toc = time.time()
        total_t_consumption += toc-tic
        t_history.append(toc-tic)
        time.sleep(interval_t*1e-3)
    average_t = 1000*total_t_consumption/(num_of_test)

    #Report:
    print("Average time used: {0} ms".format(average_t))
    print("Done")

    #embed()
