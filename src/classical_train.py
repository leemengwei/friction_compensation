#数学模型训练主程序：
#输出模型文件说明：
#在声明训练数据的分割方式后，（mode=匀速加速or低速高速），将使用线性、非线性两种方式，训练各两个模型。（两种分割数据的方式都做完，将会共有四个模型。）此时，由于拟合求解，所以这里对于每种分割方式，两个模型一起串行训练，执行一次脚本即可生成该mode下所有关于传统数学模型的models。
#如下得到四个：
#python classical_train.py  --mode=acc_uniform -V -Q
#python classical_train.py  --mode=low_high -V -Q

#coding:utf-8
import time
from IPython import embed
import numpy as np
import os,sys,time
import warnings
warnings.filterwarnings("ignore")

import matplotlib
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import scipy
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

#my modules:
import data_stuff
import classical_model
import plot_utils
import evaluate

import pickle as pkl
import PIL

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Friction.')
    parser.add_argument('--learning_rate', '-LR', type=float, default=0.01)
    parser.add_argument('--test_ratio', '-TR', type=float, default=0.2)
    parser.add_argument('--criterion', '-C', type=float, default=1e-5)
    parser.add_argument('--mode', type=str, choices=["acc_uniform", "low_high"], required=True)
    parser.add_argument('--axis_num', type=int, default = 4)
    parser.add_argument('--VISUALIZATION', '-V', action='store_true', default=False)
    parser.add_argument('--Quick_data', "-Q", action='store_true', default=False)
    args = parser.parse_args()
    args.further_mode = args.mode
    print("Start...%s"%args)
    args.axis_num = args.axis_num - 1
    

    #---------------------------------Do polyfit---------------------------:
    #Get data:
    mode = ['train', args.mode]
    raw_data, part1_index, part2_index = data_stuff.get_data(args, mode)

    plt.figure(figsize=(14, 8))
    for _idx_, this_part in enumerate([part1_index, part2_index]):
        print("PART start...")
        raw_data_part = raw_data.iloc[this_part]
        data_Y = raw_data_part['need_to_compensate'].values
    
        #Take variables we concerned:
        data_X = np.array([raw_data_part['axc_speed_%s'%args.axis_num], raw_data_part['axc_torque_ffw_gravity_%s'%args.axis_num], raw_data_part['Temp'], raw_data_part['axc_pos_%s'%args.axis_num]]) 
        data_X = np.insert(data_X, 0, 1, axis=0)
        
        #Normalize data:
        normer = data_stuff.normalizer(data_X)
        X_normed, Y_normed = normer.normalize_XY(data_X, data_Y)
        #mannual split dataset:
        X_train, Y_train, X_val, Y_val, raw_data_part_train, raw_data_part_val = data_stuff.split_dataset(args, X_normed, Y_normed, raw_data_part)
    
        #Train linear model:
        print("Training on linear...")
        linear_names_note = ['c0', 'c1', 'c2', 'c3', 'c4']
        linear_weights, linear_cov = curve_fit(classical_model.linear_friction_model, X_train, Y_train, maxfev=5000000)
        #Train nonlinear model:
        print("Training on nonlinear...")
        nonlinear_names_note = ['c0', 'v_brk', 'F_brk', 'F_C', 'c1', 'c2', 'c3', 'c4']
        nonlinear_weights, nonlinear_cov = curve_fit(classical_model.nonlinear_friction_model, X_train, Y_train, maxfev=5000000)
    
        #Linear model loss:
        linear_loss, _J, linear_friction_predicted, _, _ = classical_model.linear_compute_loss(Y_val, X_val, linear_weights)
        #Nonlinear model loss:
        nonlinear_loss, _J, nonlinear_friction_predicted, _, _ = classical_model.nonlinear_compute_loss(Y_val, X_val, nonlinear_weights)
        #Linear model error:
        linear_error_ratio = evaluate.evaluate_error_rate(args, linear_friction_predicted, Y_val, normer, raw_data_part_val, showup=True)
        plot_utils.visual(Y_val, linear_friction_predicted, "Linear", args, title=linear_error_ratio)
        #Nonlinear model error:
        nonlinear_error_ratio = evaluate.evaluate_error_rate(args, nonlinear_friction_predicted, Y_val, normer, raw_data_part_val)
        plot_utils.visual(Y_val, nonlinear_friction_predicted, "Nonlinear", args, title=nonlinear_error_ratio)
        plt.clf()

        #Report linear:
        print("Linear Names note:", linear_names_note, "Linear Error rate:", linear_error_ratio)
        print("Linear weights:", linear_weights)
        np.savetxt("../models/linear_weights_%s"%args.mode.split("_")[_idx_], linear_weights)
    
        #Report nonlinear:
        print("Nonlinear Names note:", nonlinear_names_note, "Nonlinear Error rate:", nonlinear_error_ratio)
        print("Nonlinear weights:", nonlinear_weights)
        np.savetxt("../models/nonlinear_weights_%s"%args.mode.split("_")[_idx_], nonlinear_weights)
    
    #sys.exit()
    #--------------------------Do Batch gradient descent---------------------: 
    #linear_params = np.array([-1.83062977e-09, 1.50000000e+00, 6.82213855e-09, 3.00000000e+00, -4.12101372e-09])
    #nonlinear_params = np.array([-1.78197004e-09, 1.99621299e+00, 9.82765471e-08, -6.02984398e-09, 1.49999993e+00, 1.78627117e-09, 5.00000000e+00, 2.69899575e-09])

    #J_history = []
    #if VISUALIZATION:
    #    fig=plt.figure()
    #BGD_lr = args.learning_rate
    #for iters in range(int(args.max_epoch)):
    #    BGD_loss, J, BGD_friction_predicted, params, partials = classical_model.compute_loss(Y, X, params, args)
    #    gradients = np.dot(partials, BGD_loss)/batch_size
    #    params = params - BGD_lr*gradients
    #    if iters>10:
    #        if np.abs(J-J_old)<args.criterion:
    #            BGD_lr *= 0.2
    #            args.criterion *= 0.2
    #            print("Epoch:", iters, "Reducing lr to %s, and setting criterion to %s"%(BGD_lr, args.criterion))
    #            if BGD_lr < 1e-9:
    #               break
    #            pass
    #    J_old = J
    #    print(iters, 'of', args.max_epoch, J)
    #    J_history.append(J)
    #    error_ratio = evaluate.evaluate_error_rate(args, BGD_friction_predicted, data, normer)
    #    plot_utils.visual(Y.values, BGD_friction_predicted, 'BGD', args, title=error_ratio)
    #print("Names note:", names_note)
    #print("BGD:", params)
    #print("Error rate:", error_ratio)
    #embed()
    #sys.exit()

   



