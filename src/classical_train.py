#coding:utf-8
import time
from IPython import embed
import numpy as np
import os,sys,time

import matplotlib
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import scipy
import argparse
from tqdm import tqdm

#my modules:
import data_stuff
import classical_model
import plot_utils
import evaluate

import pickle as pkl
import PIL

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Friction.')
    parser.add_argument('--mode', '-M', default='linear')
    parser.add_argument('--learning_rate', '-LR', type=float, default=0.01)
    parser.add_argument('--test_ratio', '-TR', type=float, default=0.2)
    parser.add_argument('--criterion', '-C', type=float, default=1e-5)

    parser.add_argument('--axis_num', type=int, default = 4)
    parser.add_argument('--VISUALIZATION', '-V', action='store_true', default=False)
    parser.add_argument('--Quick_data', "-Q", action='store_true', default=False)
    args = parser.parse_args()
    args.axis_num = args.axis_num - 1
    
    print("Start...%s"%args)
 
    #---------------------------------Do polyfit---------------------------:
    #Get data:
    raw_data = data_stuff.get_useful_data(args)
    data_Y = raw_data['need_to_compensate'].values

    #Take variables we concerned:
    #Use planned data:
    data_X = np.array([raw_data['axc_speed_%s'%args.axis_num], raw_data['axc_torque_ffw_gravity_%s'%args.axis_num], raw_data['Temp'], raw_data['axc_pos_%s'%args.axis_num]])   #TODO: this is axis 4
    #Use real-time data:
    #data_X = np.array([raw_data['servo_feedback_speed_%s'%args.axis_num], raw_data['axc_torque_ffw_gravity_%s'%args.axis_num], raw_data['Temp'], raw_data['servo_feedback_pos_%s'%args.axis_num]])  #TODO: this is for axis 4
    data_X = np.insert(data_X, 0, 1, axis=0)

    #Normalize data:
    normer = data_stuff.normalizer(data_X)
    X_normed, Y_normed = normer.normalize_XY(data_X, data_Y)

    #mannual split dataset:
    X_train, Y_train, X_val, Y_val, raw_data_train, raw_data_val = data_stuff.split_dataset(args, X_normed, Y_normed, raw_data)

    #Init unknown params:
    if args.mode == 'linear':
        names_note = ['c0', 'c1', 'c2', 'c3', 'c4']
        #params = np.array([-1.83062977e-09, 1.50000000e+00, 6.82213855e-09, 3.00000000e+00, -4.12101372e-09])
        #params = np.array([1,1,1,1,1])

    else:
        names_note = ['c0', 'v_brk', 'F_brk', 'F_C', 'c1', 'c2', 'c3', 'c4']
        #params = np.array([-1.78197004e-09, 1.99621299e+00, 9.82765471e-08, -6.02984398e-09, 1.49999993e+00, 1.78627117e-09, 5.00000000e+00, 2.69899575e-09])
        #params = np.array([1,1,1,1,1,1,1,1])

    #Train model:
    if args.mode == 'linear':
        opt, cov = curve_fit(classical_model.linear_friction_model, X_train, Y_train, maxfev=500000)
    else:
        opt, cov = curve_fit(classical_model.nonlinear_friction_model, X_train, Y_train, maxfev=500000)

    #Test loss:
    poly_loss, _J, poly_friction_predicted, _, _ = classical_model.compute_loss(Y_val, X_val, opt, args)
    error_ratio = evaluate.evaluate_error_rate(args, poly_friction_predicted, Y_val, normer, raw_data_val, showup=True)
    plot_utils.visual(Y_val, poly_friction_predicted, "poly", args, title=error_ratio)

    #Report:
    print("Names note:", names_note, 'J:', _J)
    print("Poly:", opt)
    print("Error rate:", error_ratio)
    np.savetxt("classical_weights", opt)


    #embed()
    #sys.exit()
    #--------------------------Do Batch gradient descent---------------------:
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

   



