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

def linear_friction_model(X, c0, c1, c2, c3, c4):
        #f = np.dot(X, params)     # verbose form in case of regression interface
        f = X[0]*c0 + X[1]*c1 + X[2]*c2 + X[3]*c3 + X[4]*c4
        return f

def nonlinear_friction_model(X, c0, c1, c2, c3, c4):
        f = X[0]*c0 + X[1]*c1 + X[2]*c2 + X[3]*c3 + X[4]*c4
        return f

def compute_loss(friction_label, X, params):
    c0, c1, c2, c3, c4 = params[0], params[1], params[2], params[3], params[4]
    friction_predicted = linear_friction_model(X, c0, c1, c2, c3, c4)
    loss = friction_predicted - friction_label
    J = np.sum(loss**2)/(2*X.shape[1])   #cost function
    print("Loss:", loss.sum())
    return loss, J, friction_predicted, params

def visual():
    if VISUALIZATION:
        plt.clf()
        ax = Axes3D(fig)
        ax.plot_trisurf(data['v'], data['Temp'], friction_label, cmap='summer', alpha=0.9, label='label')
        ax.plot_trisurf(data['v'], data['Temp'], friction_predicted, cmap='winter', alpha=0.5, label='predicted')
        ax.plot_trisurf(data['v'], data['Temp'], loss, cmap='hot', alpha=0.1, label='error')
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
    X = np.insert(X, 0, 1, axis=0)
    popt, pcov = curve_fit(linear_friction_model, X, friction_label)
    print(popt)
    #sys.exit()
    
    #Do BGD:
    params = np.array([1, 2, 3, 4, 5]) 
    batch_size = data.shape[0]
    lr = 0.1
    fig=plt.figure()
    while True:
        print(np.round(params, 4))
        loss, J, friction_predicted, params = compute_loss(friction_label, X, params)
        c0, c1, c2, c3, c4 = params
        gradients = np.dot(X, loss)/batch_size
        params = params - lr*gradients
        if max(lr*gradients)<1e-5:
            break
        visual()
        #input()



    #embed()
    
