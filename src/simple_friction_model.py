#coding:utf-8
#simple model
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

#plt.ion()
DEBUG = False
#DEBUG = True
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
    friction_value = data['v']*1.5 + data['Temp']*3 + 0.1*np.random.rand(300)
    data['friction_values'] = friction_value
    return data

def friction_model(X, c0, c1, c2, c3, c4):
        f = 1*c0 + X[0]*c1 + X[1]*c2 + X[2]*c3 + X[3]*c4
        return f

def compute_loss(friction_label, X, params):
    c0, c1, c2, c3, c4 = params[0], params[1], params[2], params[3], params[4]
    friction_predicted = friction_model(X, c0, c1, c2, c3, c4)
    losses = np.abs(friction_label-friction_predicted)
    print("Loss:", losses.sum())
    return friction_predicted, params, losses

def visual():
    if VISUALIZATION:
        plt.clf()
        ax = Axes3D(fig)
        ax.plot_trisurf(data['v'], data['Temp'], friction_label, cmap='summer', alpha=0.9, label='label')
        ax.plot_trisurf(data['v'], data['Temp'], friction_predicted, cmap='winter', alpha=0.5, label='predicted')
        ax.plot_trisurf(data['v'], data['Temp'], losses, cmap='hot', alpha=0.1, label='error')
        ax.scatter3D(data['v'], data['Temp'], friction_label, color='k')
        ax.set_xlabel('v(rad/s)',fontsize =20)
        ax.set_ylabel('Temp(degree)',fontsize =20)
        ax.set_zlabel('friction(N.m)',fontsize =20)
        #plt.show()
        plt.draw()
        plt.pause(0.02)

if __name__ == "__main__":
    print("Start...")
    #Get data:
    data = get_data()
    friction_label = data['friction_values']

    #Do polyfit:
    X = np.array([data['v'], data['tao_f'], data['Temp'], data['q']])
    #X = np.insert(X, 0, 1, axis=0)
    popt, pcov = curve_fit(friction_model, X, friction_label)
    print(popt)
    sys.exit()
    
    #Do SGD: 
    params = np.array([1, 2, 3, 4, 5]) 
    #update_params()
    fig=plt.figure()
    while True:
        print(np.round(params, 4))
        friction_predicted, params, losses = compute_loss(friction_label, X, params)
        c0, c1, c2, c3, c4 = params
        lr = 0.01/data.shape[0]
        #embed()
        params = params - lr*sum((friction_label-friction_predicted)*friction_predicted*(1-friction_predicted))*(data.values[:,:5].mean(axis=0))
        #if lr*losses.sum()*0.1<1e-5:
           # break
        visual()
        #input()



    #embed()
    
