#coding: utf-8
#python NN_deploy.py  -V --mode='low_high'
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import data_stuff
from IPython import embed
import evaluate
import NN_model
import warnings
import time
import copy
import plot_utils
warnings.filterwarnings("ignore")

def get_part_model(shape_X, name, axis_num):
    model_path = "../models/NN_weights_best_%s_%s"%(name, axis_num)
    print("Loading part model:%s"%model_path)
    #model = torch.load(model_path, map_location=torch.device(device_type))
    #model = NN_model.NeuralNet(input_size=25, hidden_size=25, hidden_depth=3, output_size=1, device=torch.device(device_type))
    model = NN_model.NeuralNetSimple(input_size=shape_X, hidden_size=shape_X*5, hidden_depth=3, output_size=1, device=torch.device(device_type))
    model.load_state_dict(torch.load(model_path).state_dict())
    model = model.to(device_type)
    return model

def to_C(args, model_part1, model_part2, inputs):
    if 'all' not in args.mode:
        name1 = args.mode.split('_')[0]     #'acc'
        name2 = args.mode.split('_')[1]     #'uniform'
    else:   # 'acc_uniform_all', 'low_high_all'
        name1 = args.mode.split('_')[-1]    #'all'
        name2 = args.mode.split('_')[-1]    #'all' as well
    #Trace with jit:
    model_part1.eval()
    model_part2.eval()
    traced_module1 = torch.jit.trace(model_part1, inputs)
    traced_module2 = torch.jit.trace(model_part2, inputs)
    model1_path = "../models/NN_weights_%s_C.pt"%name1
    model2_path = "../models/NN_weights_%s_C.pt"%name2
    traced_module1.save(model1_path)
    traced_module2.save(model2_path)
    print("C models saved")

def get_data_six(args, raw_plan, mode):
    #Take out what we need:
    #Make inputs:
    raw_data_X = np.empty(shape=(0, len(raw_plan)))
    input_columns_names = []
    #data are index from 0, thus
    local_axis_num = args.axis_num - 1
    input_columns_names += ['axc_pos_%s'%local_axis_num]
    input_columns_names += ['axc_speed_%s'%local_axis_num]
    input_columns_names += ['axc_torque_ffw_gravity_%s'%local_axis_num]
    input_columns_names += ['axc_torque_ffw_%s'%local_axis_num]
    input_columns_names += ['Temp']
    raw_data_X = raw_plan[input_columns_names].values.T
    raw_data_Y = raw_plan['need_to_compensate'].values
   
    #Normalize data:
    normer = data_stuff.normalizer(raw_data_X, raw_data_Y, args)
    normer.get_statistics(raw_data_X.shape[1])
    normed_data_X, normed_data_Y = normer.normalize_XY(raw_data_X, raw_data_Y)
    return normed_data_X, normed_data_Y, normer

def get_model_six(args, shape_X):
    #Get model:
    assert 'all' in args.mode, "I assume this is new version which train acc uniform together as all"
    model = get_part_model(shape_X, args.mode.split('_')[-1], args.axis_num)    #'all'
    return model

def get_compensate_force(args, normed_data_X, normed_data_Y, model_part1, model_part2, part1_index, part2_index):
    #Forward to get output:
    inputs = torch.FloatTensor(normed_data_X.T).to(device_type)
    output_part1 = model_part1(inputs[part1_index]).detach().cpu().numpy()
    output_part2 = model_part2(inputs[part2_index]).detach().cpu().numpy()

    #Speed test:
    #if args.speed_test:
    #    inputs[:,:]=1
    #    #On default device:
    #    for i in range(10): 
    #        model_part1(inputs[:args.buffer_length])
    #    tic = time.time() 
    #    for i in range(1000): 
    #        model_part1(inputs[:args.buffer_length])
    #    print((time.time()-tic)/1000*1000, "ms on %s"%device_type)  
    #    #On cpu deivce:
    #    cpu_model = copy.deepcopy(model_part1).cpu()
    #    cpu_inputs = inputs.cpu()
    #    for i in range(10): 
    #        cpu_model(cpu_inputs[:args.buffer_length])
    #    tic = time.time() 
    #    for i in range(1000): 
    #        cpu_model(cpu_inputs[:args.buffer_length])
    #    print((time.time()-tic)/1000*1000, "ms on cpu")  

    #import my_analyzer
    #my_analyzer.performance_shape(raw_plan, inputs, model_part1)
    #Compose together:
    #TODO: solve the switch point.
    output_full_series = np.zeros(normed_data_X.shape[1])
    output_full_series[part1_index] = output_part1.reshape(-1)
    output_full_series[part2_index] = output_part2.reshape(-1)

    #Save for Production:
    to_C(args, model_part1.cpu(), model_part2.cpu(), inputs.cpu())
    model_returned = model_part1
    return output_full_series, inputs


if __name__ == "__main__":
    print("COMPOSE TEST...")

    #Configs:
    parser = argparse.ArgumentParser(description='Friction.')
    parser.add_argument('--max_force', type=int, default = 10)
    parser.add_argument('--buffer_length', type=int, default = 6)
    parser.add_argument('--mode', type=str, choices=["acc_uniform", "low_high", "acc_uniform_all", "low_high_all"], default='acc_uniform_all')
    parser.add_argument('--data_path', type=str, default = "../data/standard_path/realtime-20200326-171651.rec-data-testzhixian.prb-log")
    parser.add_argument('--VISUALIZATION', "-V", action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    args = parser.parse_args()
    cuda_is_available = torch.cuda.is_available()
    if args.no_cuda:
        cuda_is_available = False
    if not cuda_is_available:
        device_type = 'cpu'
        print("Using cpu")
    else:
        device_type = 'cuda'
        print("Using gpu")
    mode = ['deploy', args.mode]

    #Get planned data and preprocess:
    normed_data_X_dict = {}
    normed_data_Y_dict = {}
    normer_dict = {}
    raw_plans_dict = {}
    part1_index_dict = {}
    part2_index_dict = {}
    for temp_axis in range(1,7):
        args.axis_num = temp_axis
        raw_plans_dict[temp_axis], part1_index_dict[temp_axis], part2_index_dict[temp_axis] = data_stuff.get_data(args, mode)
        normed_data_X_dict[temp_axis], normed_data_Y_dict[temp_axis], normer_dict[temp_axis] = get_data_six(args, raw_plans_dict[temp_axis], mode)

    #Get models:
    models_dict = {}
    for temp_axis in range(1,7):
        args.axis_num = temp_axis
        model = get_model_six(args, normed_data_X_dict[temp_axis].shape[0])
        models_dict[temp_axis] = model
   
    #Forwards:
    output_full_series_dict = {}
    inputs_dict = {}
    for temp_axis in range(1,7):
        args.axis_num = temp_axis
        output_full_series_dict[temp_axis], inputs_dict[temp_axis] = get_compensate_force(args, normed_data_X_dict[temp_axis], normed_data_Y_dict[temp_axis], models_dict[temp_axis], models_dict[temp_axis], part1_index_dict[temp_axis], part2_index_dict[temp_axis])
   
    #Evaluations:
    error_original_dict = {}
    error_treated_dict = {}
    for temp_axis in range(1,7):
        args.axis_num = temp_axis
        #Note evaluation take place with normed data, and then denormed within.
        error_original_dict[temp_axis], _, error_treated_dict[temp_axis] = evaluate.evaluate_error_rate(args, output_full_series_dict[temp_axis], normed_data_Y_dict[temp_axis], normer_dict[temp_axis], raw_plans_dict[temp_axis], showup=False)

    #Denormalize and Safety restrictions:
    compensate_full_series_dict = {}
    for temp_axis in range(1,7):
        args.axis_num = temp_axis
        compensate_full_series_dict[temp_axis] = normer_dict[temp_axis].denormalize_Y(output_full_series_dict[temp_axis])
        compensate_full_series_dict[temp_axis] = np.clip(compensate_full_series_dict[temp_axis], -args.max_force, args.max_force)

    #Get show data:
    meassured_dict = {}
    planned_dict = {}
    compensated_dict = {}
    for temp_axis in range(1,7):
        args.axis_num = temp_axis
        local_axis_num = args.axis_num - 1
        meassured_dict[temp_axis] = raw_plans_dict[temp_axis]['servo_feedback_torque_%s'%local_axis_num].values  #same as plan+target
        planned_dict[temp_axis] = raw_plans_dict[temp_axis]['axc_torque_ffw_gravity_%s'%local_axis_num].values
        compensated_dict[temp_axis] = planned_dict[temp_axis] + compensate_full_series_dict[temp_axis]

    #And show:
    axes = []
    #Visualize:
    plt.ion()
    if args.VISUALIZATION:
        fig = plt.figure(figsize=(16,9))
        for temp_axis in range(1,7):
            axes.append(fig.add_subplot(2,3,temp_axis))
            args.axis_num = temp_axis
            local_axis_num = args.axis_num - 1
            axes[temp_axis-1].plot(meassured_dict[temp_axis], label=r'real_target', alpha=0.5)
            axes[temp_axis-1].scatter(range(len(raw_plans_dict[temp_axis])), planned_dict[temp_axis], label=r'dynamic_model+gravity', color='gray', s=1, alpha=0.5)
            axes[temp_axis-1].scatter(part1_index_dict[temp_axis], compensated_dict[temp_axis][part1_index_dict[temp_axis]], label=r'after compensate', color='green', s=1)
            axes[temp_axis-1].scatter(part2_index_dict[temp_axis], compensated_dict[temp_axis][part2_index_dict[temp_axis]], label=r'after compensate', color='red', s=1)
            axes[temp_axis-1].legend()
            axes[temp_axis-1].set_title("Axis:{2:d}, Error treated: {0:.2f}%, original:{1:.2f}%".format(error_treated_dict[temp_axis], error_original_dict[temp_axis], int(temp_axis)))
            #print("One of the reason it may shows higher error rate than expected is because some uniform speed range be ommited during train/val evaluation")
        fig.suptitle("Path name: %s"%args.data_path.split('/')[-1])
        plt.savefig("../pngs/%s"%args.data_path.split('/')[-1].replace('prb-log','png'))
        plt.show()

        #Response surface:
        axes = []
        fig = plt.figure(figsize=(16,9))
        from mpl_toolkits.mplot3d import Axes3D
        for temp_axis in range(1,7):
            axes.append(fig.add_subplot(2,3,temp_axis,projection='3d'))
            args.axis_num = temp_axis
            if args.VISUALIZATION:
                plot_utils.response_surface(axes[temp_axis-1], models_dict[temp_axis], inputs_dict[temp_axis])
            np.savetxt("../output/NN_compensation_%s.txt"%temp_axis, compensate_full_series_dict[temp_axis])
        plt.savefig("../pngs/response_surf.png")
        plt.show()
    print("Done")




