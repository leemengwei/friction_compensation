import time
from IPython import embed
import numpy as np
import os,sys,time

def linear_compute_loss(Y, X, params):
    c0, c1, c2, c3, c4 = params[0], params[1], params[2], params[3], params[4]
    friction_predicted = linear_friction_model(X, c0, c1, c2, c3, c4)
    partials = linear_friction_model_derivatives(X)
    loss = friction_predicted - Y
    J = np.sum(loss**2)/(2*X.shape[1]) 
    return loss, J, friction_predicted, params, partials

def nonlinear_compute_loss(Y, X, params):
    c0, v_brk, F_brk, F_C, c1, c2, c3, c4 = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]
    friction_predicted = nonlinear_friction_model(X, c0, v_brk, F_brk, F_C, c1, c2, c3, c4)
    partials = nonlinear_friction_model_derivatives(X, v_brk, F_C, F_brk)
    loss = friction_predicted - Y
    J = np.sum(loss**2)/(2*X.shape[1]) 
    return loss, J, friction_predicted, params, partials

def linear_friction_model(X, c0, c1, c2, c3, c4):
    #f = c0 + c1*v +c2*t + c3*T + c4*q
    Linear_f = X[0]*c0 + X[1]*c1 + X[2]*c2 + X[3]*c3 + X[4]*c4
    return Linear_f

def linear_friction_model_derivatives(X):
    partial_f_c0 = X[0]
    partial_f_c1 = X[1]
    partial_f_c2 = X[2]
    partial_f_c3 = X[3]
    partial_f_c4 = X[4]
    partials = np.vstack((partial_f_c0, partial_f_c1, partial_f_c2, partial_f_c3, partial_f_c4))
    return partials 

def nonlinear_friction_model(X, c0, v_brk, F_brk, F_C, c1, c2, c3, c4):
    v_St = v_brk**np.sqrt(2)
    v_Coul = v_brk/10
    v = X[1]
    f = c1  #Viscous friction term
    #matlab nonlinear velocity-related term:
    F = np.sqrt(2*np.e)*(F_brk-F_C)*np.exp(-(v/v_St)**2)*(v/v_St) + F_C*np.tanh(v/v_Coul) + f*v
    #f = c0 + c1*v +c2*t + c3*T + c4*q
    Nonlinear_f = X[0]*c0 + F + X[2]*c2 + X[3]*c3 + X[4]*c4
    return Nonlinear_f

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


