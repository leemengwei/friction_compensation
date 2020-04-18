#coding: utf-8
#python NN_deploy.py  -V --mode='low_high'
import sys
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
from NN_test import *
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    print("COMPOSE TEST...")

    #Configs:
    parser = argparse.ArgumentParser(description='Friction.')
    parser.add_argument('--max_force', type=int, default = 1e9)
    parser.add_argument('--time_to_plot', type=int, default = 1500)
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
        args.device_type = 'cpu'
        print("Using cpu")
    else:
        args.device_type = 'cuda'
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
        normed_data_X_dict[temp_axis], normed_data_Y_dict[temp_axis], normer_dict[temp_axis] = get_data_one(args, raw_plans_dict[temp_axis], mode)

    #Get models:
    models_dict = {}
    for temp_axis in range(1,7):
        args.axis_num = temp_axis
        model,_ = get_model_one(args, normed_data_X_dict[temp_axis].shape[0])
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
        args.rated_torque = [5.7, 5.7, 1.02, 0.318, 0.318, 0.143][temp_axis-1]
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
    #plt.ion(
    fig = plt.figure(figsize=(16,9))
    length_of_plot = args.time_to_plot
    for temp_axis in range(1,7):
        axes.append(fig.add_subplot(2,3,temp_axis))
        args.axis_num = temp_axis
        local_axis_num = args.axis_num - 1
        axes[temp_axis-1].plot(list(range(len(meassured_dict[temp_axis])))[:length_of_plot], meassured_dict[temp_axis][:length_of_plot], label=r'real_target', alpha=0.5)
        axes[temp_axis-1].scatter(list(range(len(raw_plans_dict[temp_axis])))[:length_of_plot], planned_dict[temp_axis][:length_of_plot], label=r'dynamic_model+gravity', color='gray', s=1, alpha=0.5)
        axes[temp_axis-1].scatter(part1_index_dict[temp_axis][:length_of_plot], compensated_dict[temp_axis][part1_index_dict[temp_axis]][:length_of_plot], label=r'after compensate', color='green', s=1)
        axes[temp_axis-1].scatter(part2_index_dict[temp_axis][:length_of_plot], compensated_dict[temp_axis][part2_index_dict[temp_axis]][:length_of_plot], label=r'after compensate', color='red', s=1)
        axes[temp_axis-1].legend()
        axes[temp_axis-1].set_title("Axis:{2:d}, Error treated: {0:.2f}%, original:{1:.2f}%".format(error_treated_dict[temp_axis], error_original_dict[temp_axis], int(temp_axis)))
        #print("One of the reason it may shows higher error rate than expected is because some uniform speed range be ommited during train/val evaluation")
        fig.suptitle("Path name: %s"%args.data_path.split('/')[-1])
    plt.savefig("../pngs/%s"%args.data_path.split('/')[-1].replace('prb-log','png'), dpi=500)
    if args.VISUALIZATION:
        plt.show()
    plt.close()

    #Response surface:
    axes = []
    fig = plt.figure(figsize=(16,9))
    from mpl_toolkits.mplot3d import Axes3D
    for temp_axis in range(1,7):
        axes.append(fig.add_subplot(2,3,temp_axis,projection='3d'))
        args.axis_num = temp_axis
        plot_utils.response_surface(axes[temp_axis-1], models_dict[temp_axis], inputs_dict[temp_axis])
        np.savetxt("../output/NN_compensation_%s.txt"%temp_axis, compensate_full_series_dict[temp_axis])
    plt.savefig("../pngs/response_surf.png",dpi=500)
    if args.VISUALIZATION:
        plt.show()
    plt.close()

    print("Original: %s,\n Treated: %s"%(error_original_dict, error_treated_dict))
    print("Done")




