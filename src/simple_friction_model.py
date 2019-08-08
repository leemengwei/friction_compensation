#coding:utf-8
#simple model
import time
from IPython import embed
import numpy as np
import pandas as pd
import os,sys,time
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import scipy
#from sklearn.linear_model import LinearRegression 
import argparse

#plt.ion()
DEBUG = False
DEBUG = True
VISUALIZATION = False
VISUALIZATION = True

def report_it(func):
    def run(*argv):
        if DEBUG:
            print (func.__name__, "called")
        if argv:
            ret = func(*argv)
        else:
            ret = func()
        return ret
    return run

def get_data():
    column_names = ['v', 'q', 'Temp', 'fai_a', 'fai_b', 'fai_c', 'fai_d', 'fai_e', 'fai_f', 'tao_f', 'tao_t_g', 'tao_t_ng']
    data = pd.DataFrame(np.random.random((300, len(column_names))), columns=column_names)
    friction_value = data['v']*1.5 + data['Temp']*5 + 0*np.random.rand(300)
    data['friction_values'] = friction_value
    return data

def linear_friction_model(X, c0, c1, c2, c3, c4):
    f = X[0]*c0 + X[1]*c1 + X[2]*c2 + X[3]*c3 + X[4]*c4
    return f

def linear_friction_model_derivatives(X):
    partial_f_c0 = X[0]
    partial_f_c1 = X[1]
    partial_f_c2 = X[2]
    partial_f_c3 = X[3]
    partial_f_c4 = X[4]
    partials = np.vstack((partial_f_c0, partial_f_c1, partial_f_c2, partial_f_c3, partial_f_c4))
    return partials 

def nonlinear_friction_model(X, c0, v_brk, F_brk, F_C, c1, c2, c3, c4):
    f = X[0]*c0 + (np.sqrt(2*np.e)*(F_brk-F_C)*np.exp(-(X[1]/(v_brk*np.sqrt(2)))**2)*(X[1]/(v_brk*np.sqrt(2))) + F_C*np.tanh(X[1]/(v_brk/10)) + X[1]*c1) + X[2]*c2 + X[3]*c3 + X[4]*c4
    return f

def nonlinear_friction_model_derivatives(X, v_brk, F_C, F_brk):
    partial_f_c0 = X[0]
    partial_f_v_brk = -10*F_C*X[1]*(1 - np.tanh(10*X[1]/v_brk)**2)/v_brk**2 + (1/np.sqrt(2))*X[1]**3*((-np.sqrt(2*np.e))*F_C + (np.sqrt(2*np.e))*F_brk)*np.exp(-0.5*X[1]**2/v_brk**2)/v_brk**4 - (1/np.sqrt(2))*X[1]*((-np.sqrt(2*np.e))*F_C + (np.sqrt(2*np.e))*F_brk)*np.exp(-0.5*X[1]**2/v_brk**2)/v_brk**2
    partial_f_F_brk = (np.sqrt(2*np.e)/np.sqrt(2))*X[1]*np.exp(-0.5*X[1]**2/v_brk**2)/v_brk
    partial_f_F_C = -(np.sqrt(2*np.e)/np.sqrt(2))*X[1]*np.exp(-0.5*X[1]**2/v_brk**2)/v_brk + np.tanh(10*X[1]/v_brk)
    partial_f_c1 = X[1]
    partial_f_c2 = X[2]
    partial_f_c3 = X[3]
    partial_f_c4 = X[4]
    partials = np.vstack((partial_f_c0, partial_f_v_brk, partial_f_F_brk, partial_f_F_C, partial_f_c1, partial_f_c2, partial_f_c3, partial_f_c4))
    return partials 

def compute_loss(friction_label, X, params):
    c0, c1, c2, c3, c4 = params[0], params[1], params[2], params[3], params[4]
    if args.mode == 'linear':
        friction_predicted = linear_friction_model(X, c0, c1, c2, c3, c4)
        partials = linear_friction_model_derivatives(X)
    else:
        v_brk = params[5]
        F_brk = params[6]
        F_C = params[7]
        friction_predicted = nonlinear_friction_model(X, c0, v_brk, F_brk, F_C, c1, c2, c3, c4)
        partials = nonlinear_friction_model_derivatives(X, v_brk, F_C, F_brk)
    loss = friction_predicted - friction_label
    J = np.sum(loss**2)/(2*X.shape[1]) 
    return loss, J, friction_predicted, params, partials

def visual():
    if VISUALIZATION:
        plt.clf()
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_trisurf(data['v'], data['Temp'], friction_label, cmap='summer', alpha=0.9, label='label')
        ax.plot_trisurf(data['v'], data['Temp'], friction_predicted, cmap='winter', alpha=0.6, label='predicted')
        ax.plot_trisurf(data['v'], data['Temp'], loss, cmap='hot', alpha=0.3, label='error')
        ax.scatter3D(data['v'], data['Temp'], friction_label, color='k')
        ax.set_xlabel('v(rad/s)',fontsize =20)
        ax.set_ylabel('Temp(degree)',fontsize =20)
        ax.set_zlabel('friction(N.m)',fontsize =20)
        ax2 = fig.add_subplot(122)
        ax2.plot(np.array(range(len(J_history))), np.log(J_history))

        #plt.draw()
        #plt.pause(0.02)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Friction.')
    parser.add_argument('--mode', '-M', default='linear')
    parser.add_argument('--learning_rate', '-LR', type=float, default=0.2)
    parser.add_argument('--criterion', '-C', type=float, default=1e-199)
    parser.add_argument('--max_epoch', '-E', type=int, default=1e4)
    args = parser.parse_args()
    
    print("Start...%s"%args)
    #Get data:
    data = get_data()
    batch_size = data.shape[0]
    friction_label = data['friction_values']
    #Input variables for models:
    X = np.array([data['v'], data['tao_f'], data['Temp'], data['q']])
    X = np.insert(X, 0, 1, axis=0)
    #Do polyfit:
    if args.mode == 'linear':
        opt, cov = curve_fit(linear_friction_model, X, friction_label, maxfev=500000)
    else:
        opt, cov = curve_fit(nonlinear_friction_model, X, friction_label, maxfev=500000)
    #Do BGD:
    #Init unknown params:
    if args.mode == 'linear':
        names_note = ['c0', 'c1', 'c2', 'c3', 'c4']
        params = np.array([1, 2, 3, 4, 5])
        params = np.array([-1.83062977e-09, 1.50000000e+00, 6.82213855e-09, 3.00000000e+00, -4.12101372e-09])
    else:
        names_note = ['c0', 'v_brk', 'F_brk', 'F_C', 'c1', 'c2', 'c3', 'c4']
        params = np.array([1, 2, 3, 4, 5, 1, 1, 1]) 
        params = np.array([-1.78197004e-09, 1.99621299e+00, 9.82765471e-08, -6.02984398e-09, 1.49999993e+00, 1.78627117e-09, 5.00000000e+00, 2.69899575e-09])
    J_history = []
    fig=plt.figure()
    for iters in range(int(args.max_epoch)):
        loss, J, friction_predicted, params, partials = compute_loss(friction_label, X, params)
        gradients = np.dot(partials, loss)/batch_size
        params = params - args.learning_rate*gradients
        if iters>10:
            if np.abs(J-J_old)<args.criterion:
                args.learning_rate = args.learning_rate*0.9
                print("Epoch:", iters, "Reducing lr to %s, and setting criterion to %s"%(args.learning_rate, args.criterion))
                if args.learning_rate < 1e-10:
                   break
                pass
        J_old = J
        print(J)
        J_history.append(J)
    visual()

    print("Names note:", names_note)
    print("Poly:", opt)
    print("BGD:", params)
    embed()
    
