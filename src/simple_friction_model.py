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
import argparse
import pandas as pd
from tqdm import tqdm

import pickle as pkl
import PIL

import torch.nn as nn
import torch
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
torch.manual_seed(44)

import sav_smooth

#plt.ion()
DEBUG = False
DEBUG = True
VISUALIZATION = False
VISUALIZATION = True

def evaluate_error_rate(outputs, targets):
    return np.abs(((outputs-targets)/targets)).mean()*100

def data_retrieve(data):
    def _data_drop(data):
        smoothed_v = sav_smooth.savitzky_golay(data['servo_feedback_speed_3'].values, 101, 1)
        deltas = smoothed_v[1:] - smoothed_v[:-1]
        _threshold = 0.1  #threshold for v average
        #plt.plot(data['ms'], data['servo_feedback_speed_3'])
        #plt.scatter(data.iloc[list(np.where(np.abs(deltas)>_threshold)[0])].ms, data.iloc[list(np.where(np.abs(deltas)>_threshold)[0])].servo_feedback_speed_3, 3, 'r')
        data = data.drop(list(np.where(np.abs(deltas)>_threshold)[0]))
        #plt.plot(data['ms'], data['servo_feedback_speed_3'])
        #plt.scatter(data.ms, data.servo_feedback_speed_3, 3, 'r')
        return data
    def _data_stay(data):
        compensator = 0.9   #to make sure all data at fastest routine (yet turbulent) are kept.
        chance_to_stay = np.abs(data['servo_feedback_speed_3'])/(np.abs(data['servo_feedback_speed_3']).max()*compensator)
        which_to_stay = (np.random.random(len(chance_to_stay)) < chance_to_stay)
        data = data[which_to_stay]
        return data
    #Get rid of accelerating data
    data_dropped = _data_drop(data)
    #And re-scale by position (achived by chance)
    data_stayed = _data_stay(data_dropped)
    return data_stayed

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

def train(args, model, device, train_loader, optimizer, epoch):
    model = model.to(device)
    model.train()
    LOSS = 0
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model((data))
        loss = F.mse_loss(output, target)
        loss.backward()
        LOSS += F.mse_loss(output, target, reduction='sum').item() # pile up all loss
        optimizer.step()
        if (batch_idx % args.log_interval == 0 or batch_idx == len(train_loader)-1)and(batch_idx!=0):
            #pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]. Loss: {:.8f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))
            pass
    train_loss_mean = LOSS/len(train_loader.dataset)
    print("Train Epoch: {} LOSS:{:.1f}, Average loss: {:.8f}".format(epoch, LOSS, train_loss_mean))
    print(output[0], target[0])
    return train_loss_mean, output, target

def validate(args, model, device, validate_loader):
    model = model.to(device)
    model.eval()
    LOSS = 0
    with torch.no_grad():
        pbar = tqdm(validate_loader)
        for idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            output = model((data))
            LOSS += F.mse_loss(output, target, reduction='sum').item() # pile up all loss
            #pbar.set_description('Validate: [{}/{} ({:.0f}%)]'.format(idx*len(data), len(validate_loader.dataset), 100.*idx/len(validate_loader)))
    validate_loss_mean = LOSS/len(validate_loader.dataset)
    print('Validate set LOSS: {:.1f}, Average loss: {:.8f}'.format(LOSS, validate_loss_mean))
    print(output[0], target[0])
    return validate_loss_mean, output, target

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_depth, output_size, device):
        super(NeuralNet, self).__init__()
        self.fc_first1 = nn.Linear(input_size, 10)
        self.fc_first2 = nn.Linear(10, hidden_size)
        self.hidden_depth = hidden_depth
        self.fc_last1 = nn.Linear(hidden_size, 10)
        self.fc_last2 = nn.Linear(10, output_size)
        self.relu = nn.ReLU()
        self.sp = nn.Softplus()
        self.th = nn.Tanh()
        self.ths = nn.Tanhshrink()
        self.sg = nn.Sigmoid()
        self.fcs = nn.ModuleList()   #collections.OrderedDict()
        self.bns = nn.ModuleList()   #collections.OrderedDict()
        for i in range(self.hidden_depth):
            self.bns.append(nn.BatchNorm1d(hidden_size, track_running_stats=True).to(device))
            self.fcs.append(nn.Linear(hidden_size, hidden_size).to(device))
    def forward(self, x):
        out = self.fc_first1(x)
        out = self.fc_first2(out)
        for i in range(self.hidden_depth):
            out = self.bns[i](out)
            out = self.fcs[i](out)
            out = self.relu(out)
        #out = self.th(out)
        out = self.fc_last1(out)
        out = self.fc_last2(out)
        out = self.ths(out)
        return out

def get_data():
    data = pd.read_csv("../data/realtime-20190827-165608.rec-data.prb-log", sep=' ')
    data = data.drop(data.shape[0]-1).astype(float)
    data = data.drop('line', axis=1)
    data = data_retrieve(data)
    times = data['ms']
    friction_value_axis_3 = data['servo_feedback_torque_3']-data['axc_torque_ffw_gravity_3']
    data['friction_values'] = friction_value_axis_3
    data['Temp'] = 0
    #Finally, normalize. Must at final stage.
    norm_data = (data-data.mean())/(data.std()+1e-99)
    return norm_data, data

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
    #F_brk = F_C
    f = X[0]*c0 + 0*(np.sqrt(2*np.e)*(F_brk-F_C)*np.exp(-(X[1]/(v_brk*np.sqrt(2)))**2)*(X[1]/(v_brk*np.sqrt(2))) + F_C*np.tanh(X[1]/(v_brk/10))) + X[1]*c1 + X[2]*c2 + X[3]*c3 + X[4]*c4
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

def compute_loss(Y, X, params):
    if args.mode == 'linear':
        c0, c1, c2, c3, c4 = params[0], params[1], params[2], params[3], params[4]
        friction_predicted = linear_friction_model(X, c0, c1, c2, c3, c4)
        partials = linear_friction_model_derivatives(X)
    else:
        c0, v_brk, F_brk, F_C, c1, c2, c3, c4 = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]
        friction_predicted = nonlinear_friction_model(X, c0, v_brk, F_brk, F_C, c1, c2, c3, c4)
        partials = nonlinear_friction_model_derivatives(X, v_brk, F_C, F_brk)
    loss = friction_predicted - Y
    J = np.sum(loss**2)/(2*X.shape[1]) 
    return loss, J, friction_predicted, params, partials

def visual(Y, predicted, method, extra=None, title=None):
    #Y = np.array(Y.reshape(-1))
    plt.clf()
    predicted = np.array(predicted.reshape(-1))
    if VISUALIZATION:
        #plt.plot(Y.reshape(-1)[::100], label='real')
        #plt.plot(predicted[::100], label='%s predicted'%method)
        if extra==None:
            plt.scatter(range(len(Y.reshape(-1)[::10])), Y.reshape(-1)[::10], label='real', s=2)
            plt.scatter(range(len(predicted[::10])), predicted[::10], label='%s predicted'%method, s=2)
        else:
            train_index = extra[0]
            val_index = extra[1]
            plt.scatter(range(len(Y.reshape(-1)))[::10], Y.reshape(-1)[::10], label='real', s=2)
            plt.scatter(np.array(range(len(predicted)))[train_index][::10], predicted[train_index][::10], label='trainset %s predicted'%method, s=2)
            plt.scatter(np.array(range(len(predicted)))[val_index][::10], predicted[val_index][::10], label='valset %s predicted'%method, s=2)
        plt.title("Relative error rate: {:.2f} %".format(np.array(title)))
        plt.ylim(-3, 3)
        plt.legend()
        if method!="poly":
            plt.draw()
            plt.pause(1e-10)
        else:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Friction.')
    parser.add_argument('--mode', '-M', default='linear')
    parser.add_argument('--learning_rate', '-LR', type=float, default=0.01)
    parser.add_argument('--criterion', '-C', type=float, default=1e-5)
    parser.add_argument('--max_epoch', '-E', type=int, default=100)

    parser.add_argument('--hidden_width_scaler', type=int, default = 10)
    parser.add_argument('--hidden_depth', type=int, default = 3)
    parser.add_argument('--Cuda_number', type=int, default = 0)
    parser.add_argument('--num_of_batch', type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=100)
    args = parser.parse_args()
    
    print("Start...%s"%args)
    #Get data:
    data, _data = get_data()
    batch_size = data.shape[0]
    Y = data['friction_values']
    #Input variables for models:
    #When in plan we only have:
    #_X = np.array([data['axc_speed_3'], data['axc_torque_ffw_gravity_3'], data['Temp'], data['axc_pos_3']])
    #In use we MUST have real_time_feedback, compensate from axc_torque to feedback_torque:
    _X = np.array([data['servo_feedback_speed_3'], data['axc_torque_ffw_gravity_3'], data['Temp'], data['servo_feedback_pos_3']])
    X = np.insert(_X, 0, 1, axis=0)

    #---------------------------------Do polyfit---------------------------:
    #Init unknown params:
    if args.mode == 'linear':
        names_note = ['c0', 'c1', 'c2', 'c3', 'c4']
        params = np.array([1, 2, 3, 4, 5])
        params = np.array([-1.83062977e-09, 1.50000000e+00, 6.82213855e-09, 3.00000000e+00, -4.12101372e-09])
        opt, cov = curve_fit(linear_friction_model, X, Y, maxfev=500000)
    else:
        names_note = ['c0', 'v_brk', 'F_brk', 'F_C', 'c1', 'c2', 'c3', 'c4']
        params = np.array([1, 2, 3, 4, 5, 1, 1, 1]) 
        params = np.array([-1.78197004e-09, 1.99621299e+00, 9.82765471e-08, -6.02984398e-09, 1.49999993e+00, 1.78627117e-09, 5.00000000e+00, 2.69899575e-09])
        opt, cov = curve_fit(nonlinear_friction_model, X, Y, maxfev=500000)
    poly_loss, _J, poly_friction_predicted, _, _ = compute_loss(Y, X, opt)
    print("Names note:", names_note, 'J:', _J)
    print("Poly:", opt)
    error_ratio = evaluate_error_rate(poly_friction_predicted, Y)
    visual(Y.values, poly_friction_predicted, method="poly", title=error_ratio)
    #embed()
    #sys.exit()

    #--------------------------Do Batch gradient descent---------------------:
    J_history = []
    fig=plt.figure()
    BGD_lr = args.learning_rate
    for iters in range(int(args.max_epoch)):
        BGD_loss, J, BGD_friction_predicted, params, partials = compute_loss(Y, X, params)
        gradients = np.dot(partials, BGD_loss)/batch_size
        params = params - BGD_lr*gradients
        if iters>10:
            if np.abs(J-J_old)<args.criterion:
                BGD_lr *= 0.2
                args.criterion *= 0.2
                print("Epoch:", iters, "Reducing lr to %s, and setting criterion to %s"%(BGD_lr, args.criterion))
                if BGD_lr < 1e-9:
                   break
                pass
        J_old = J
        print(iters, 'of', args.max_epoch, J)
        J_history.append(J)
        error_ratio = evaluate_error_rate(BGD_friction_predicted, Y)
        visual(Y.values, BGD_friction_predicted, method='BGD', title=error_ratio)
    print("Names note:", names_note)
    print("BGD:", params)
    #embed()
    #sys.exit()

    #-------------------------------------Do NN--------------------------------:
    #Data:
    _X = _X.T
    Y = Y.values.reshape(-1, 1)
    nn_X = torch.autograd.Variable(torch.FloatTensor(_X))
    #nn_X2 = torch.autograd.Variable(torch.FloatTensor(__X))
    nn_Y = torch.autograd.Variable(torch.FloatTensor(Y))
    test_ratio = 0.4
    #whole_dataset = Data.TensorDataset(nn_X, nn_Y)
    #Not random split train and val, explictly specify.
    #train_dataset, validate_dataset = Data.random_split(whole_dataset, (len(whole_dataset)-int(len(whole_dataset)*test_ratio),int(len(whole_dataset)*test_ratio)))
    all_index = list(range(len(nn_Y)))
    val_index = list(range(int(len(nn_Y)*(0.5-test_ratio/2)), int(len(nn_Y)*(0.5+test_ratio/2))))
    train_index = list(set(all_index) - set(val_index))
    nn_X_train = nn_X[train_index]
    nn_Y_train = nn_Y[train_index]
    nn_X_val = nn_X[val_index]
    nn_Y_val = nn_Y[val_index]
    train_dataset = Data.TensorDataset(nn_X_train, nn_Y_train)
    validate_dataset = Data.TensorDataset(nn_X_val, nn_Y_val)
    train_loader = Data.DataLoader( 
            dataset=train_dataset, 
            batch_size=int(len(train_dataset)/args.num_of_batch),
            shuffle=True,
            drop_last=True,
	        num_workers=4,
            pin_memory=True
            )
    validate_loader = Data.DataLoader( 
            dataset=validate_dataset, 
            batch_size=int(len(validate_dataset)/args.num_of_batch),
            shuffle=True,
            drop_last=True,
	        num_workers=4,
            pin_memory=True
            )
    #Model:
    device = torch.device("cuda", args.Cuda_number)
    #device = torch.device("cpu")
    input_size = nn_X.shape[1]                          
    hidden_size = nn_X.shape[1]*args.hidden_width_scaler
    hidden_depth = args.hidden_depth
    output_size = nn_Y.shape[1]
    model = NeuralNet(input_size, hidden_size, hidden_depth, output_size, device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    #embed()
    #Train and Validate:
    train_loss_history = []
    validate_loss_history = []
    for epoch in range(int(args.max_epoch+1)):
        train_loss, train_outputs, train_targets = train(args, model, device, train_loader, optimizer, epoch)
        validate_loss, validate_outputs, validate_targets = validate(args, model, device, validate_loader)
        train_loss_history.append(train_loss)
        validate_loss_history.append(validate_loss)
        train_loss_history[0] = validate_loss_history[0]
        error_ratio = evaluate_error_rate(validate_outputs.cpu(), validate_targets.cpu())

        NN_friction_predicted = np.array(model.cpu()((nn_X)).detach())
        visual(Y, NN_friction_predicted, method='NN', extra=[train_index, val_index], title=error_ratio)

    plt.plot(train_loss_history)
    plt.plot(validate_loss_history)

    names_note = "NN weights"
    print("Names note:", names_note)
    print("NN:", "NONE")

    embed()
    sys.exit()
   
