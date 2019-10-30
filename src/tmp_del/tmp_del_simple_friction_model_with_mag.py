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

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    LOSS = 0
    pbar = tqdm(train_loader)
    for batch_idx, (data, extra, target) in enumerate(pbar):
        data, extra, target = data.to(device), extra.to(device), target.to(device)
        optimizer.zero_grad()
        output = model((data, extra))
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
    return train_loss_mean

def validate(args, model, device, validate_loader):
    model.eval()
    LOSS = 0
    outputs_record = np.array([])
    targets_record = np.array([])
    with torch.no_grad():
        pbar = tqdm(validate_loader)
        for idx, (data, extra, target) in enumerate(pbar):
            data, extra, target = data.to(device), extra.to(device), target.to(device)
            output = model((data, extra))
            LOSS += F.mse_loss(output, target, reduction='sum').item() # pile up all loss
            outputs_record = np.append(outputs_record, output.cpu())
            targets_record = np.append(targets_record, target.cpu())
            #pbar.set_description('Validate: [{}/{} ({:.0f}%)]'.format(idx*len(data), len(validate_loader.dataset), 100.*idx/len(validate_loader)))
    validate_loss_mean = LOSS/len(validate_loader.dataset)
    print('Validate set LOSS: {:.1f}, Average loss: {:.8f}'.format(LOSS, validate_loss_mean))
    print(output[0], target[0])
    return validate_loss_mean, outputs_record, targets_record

'''
class MagNet(nn.Module):
    def __init__(self, input_size, output_size, device):
        super(MagNet, self).__init__()
        self.avg = nn.AvgPool2d(2)
        self.fc0 = nn.Linear(2500, 2500)
        self.bn0 = nn.BatchNorm1d(2500, track_running_stats=True)
        self.fc1 = nn.Linear(2500, 2500)
        self.bn1 = nn.BatchNorm1d(2500, track_running_stats=True)
        self.fc2 = nn.Linear(2500, 2500)
        self.bn2 = nn.BatchNorm1d(2500, track_running_stats=True)
        self.fc3 = nn.Linear(2500, 100)
        self.bn3 = nn.BatchNorm1d(100, track_running_stats=True)

        self.fc4 = nn.Linear(103, 103)
        self.bn4 = nn.BatchNorm1d(103, track_running_stats=True)
        self.fc5 = nn.Linear(103, 103)
        self.bn5 = nn.BatchNorm1d(103, track_running_stats=True)
        self.fc6 = nn.Linear(103, 20)
        self.bn6 = nn.BatchNorm1d(20, track_running_stats=True)
        self.fc7 = nn.Linear(20, 3)

        self.relu = nn.ReLU()
        self.sp = nn.Softplus()
        self.th = nn.Tanh()
        self.sg = nn.Sigmoid()
    def forward(self, X):
        x = X[0]
        x2 = X[1]
        out = self.avg(x)

        out = self.fc0(out.view(out.shape[0], -1))
        out = self.bn0(out)
        out = self.th(out)

        out = self.fc1(out)
        out = self.bn1(out)
        out = self.th(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.th(out)

        out = self.fc3(out)
        out = self.bn3(out)
        out = self.th(out)

        out = torch.cat((out,x2), 1)

        out = self.fc4(out)
        out = self.bn4(out)
        out = self.th(out)

        out = self.fc5(out)
        out = self.bn5(out)
        out = self.th(out)

        out = self.fc6(out)
        out = self.bn6(out)
        out = self.th(out)

        out = self.fc7(out)
        out = self.sp(out)
        return out
'''

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
        out = self.th(out)
        out = self.fc_last1(out)
        out = self.fc_last2(out)
        out = self.sp(out)
        return out

def get_data():
    data = pd.read_csv("./realtime-20190827-165608.rec-data.prb-log", sep=' ', index_col='line')
    data = data.drop("start_tick").astype(float)
    size = int(1e3)
    column_names = ['v', 'q', 'Temp', 'fai_a', 'fai_b', 'fai_c', 'fai_d', 'fai_e', 'fai_f', 'tao_f', 'tao_t_g', 'tao_t_ng']
    data = pd.DataFrame(np.random.random((size, len(column_names))), columns=column_names)
    friction_value = data['v']*1.5 + data['Temp']*5 + 0*np.random.rand(size)
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

def compute_loss(Y, X, params):
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
    loss = friction_predicted - Y
    J = np.sum(loss**2)/(2*X.shape[1]) 
    return loss, J, friction_predicted, params, partials

def visual(_predicted):
    #Y = np.array(Y.reshape(-1))
    _predicted = np.array(_predicted.reshape(-1))
    if VISUALIZATION:
        plt.clf()
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_trisurf(data['v'], data['Temp'], Y, cmap='summer', alpha=0.9, label='label')
        ax.plot_trisurf(data['v'], data['Temp'], _predicted, cmap='winter', alpha=0.6, label='predicted')
        ax.plot_trisurf(data['v'], data['Temp'], Y-_predicted, cmap='hot', alpha=0.3, label='error')
        ax.scatter3D(data['v'], data['Temp'], Y, color='k')
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
    parser.add_argument('--learning_rate', '-LR', type=float, default=0.01)
    parser.add_argument('--criterion', '-C', type=float, default=1e-199)
    parser.add_argument('--max_epoch', '-E', type=int, default=300)

    parser.add_argument('--hidden_width_scaler', type=int, default = 10)
    parser.add_argument('--hidden_depth', type=int, default = 3)
    parser.add_argument('--Cuda_number', type=int, default = 0)
    parser.add_argument('--num_of_batch', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=100)
    args = parser.parse_args()
    
    print("Start...%s"%args)
    #Get data:
    data = get_data()
    batch_size = data.shape[0]
    Y = data['friction_values']
    #Input variables for models:
    _X = np.array([data['v'], data['tao_f'], data['Temp'], data['q']])
    X = np.insert(_X, 0, 1, axis=0)

    #---------------------------------Do polyfit---------------------------:
    if args.mode == 'linear':
        opt, cov = curve_fit(linear_friction_model, X, Y, maxfev=500000)
    else:
        opt, cov = curve_fit(nonlinear_friction_model, X, Y, maxfev=500000)
    poly_loss, _, poly_friction_predicted, _, _ = compute_loss(Y, X, opt)

    #--------------------------Do Batch gradient descent---------------------:
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
    BGD_lr = args.learning_rate
    for iters in range(int(args.max_epoch/100)):
        BGD_loss, J, BGD_friction_predicted, params, partials = compute_loss(Y, X, params)
        gradients = np.dot(partials, BGD_loss)/batch_size
        params = params - BGD_lr*gradients
        if iters>10:
            if np.abs(J-J_old)<args.criterion:
                BGD_lr = BGD_lr*0.2
                print("Epoch:", iters, "Reducing lr to %s, and setting criterion to %s"%(BGD_lr, args.criterion))
                if BGD_lr < 1e-6:
                   break
                pass
        J_old = J
        print(J)
        J_history.append(J)

    #-------------------------------------Do NN--------------------------------:
    #Data:
    #Magnet:
    '''
    min_speed = 0.3
    enough_shape = (123, 250)
    norm_size = (100, 100)
    tmp_data = pkl.load(open("/mfs/home/wangke/magnetic-detect/data/data0.pkl", 'rb'))
    imgs_data = []
    speeds = []
    shapes = []
    depths = []
    scalers = []
    for idx in range(len(tmp_data)):
        imgs_data.append(tmp_data[idx]['data'])
        speeds.append(tmp_data[idx]['speed']/min_speed)
        shapes.append(tmp_data[idx]['data'].shape)
        scalers.append(list(np.array(norm_size)/np.array(tmp_data[idx]['data'].shape)))
        depths.append(tmp_data[idx]['depth'])
    #substruct mean
    widths = []
    lengths = []
    for idx in range(len(tmp_data)):
        imgs_data[idx] = imgs_data[idx]-imgs_data[idx].mean()
        widths.append(tmp_data[idx]['width'])
        lengths.append(tmp_data[idx]['length'])
    #prepare paddings
    streched_data = []
    new_shapes = []
    pads_needed = []
    for i in range(len(tmp_data)):
        new_shape = np.array([int(shapes[i][0]), int(shapes[i][1]*speeds[i])])
        need_to_pad = (int((enough_shape[0]-new_shape[0])/2), int((enough_shape[1]-new_shape[1])/2))
        new_shapes.append(new_shape)
        pads_needed.append(need_to_pad)
        streched_data.append(np.array(PIL.Image.fromarray(imgs_data[i]).resize(norm_size, PIL.Image.BICUBIC)))
    #make pad
    padded_data = []
    varations = []
    for i in range(len(tmp_data)):
        varations.append(np.var(streched_data[i]))
        padded_data.append(np.pad(streched_data[i], ((pads_needed[i][0], enough_shape[0]-pads_needed[i][0]-new_shapes[i][0]),(pads_needed[i][1], enough_shape[1]-pads_needed[i][1]-new_shapes[i][1])), 'constant', constant_values=0))
    _X = (np.array(streched_data)+1.0833372e-06)/0.07874487
    __X = np.vstack((np.array(scalers)[:,0], np.array(scalers)[:,1], np.array(speeds))).T
    Y = np.vstack((lengths, widths, depths)).T
    embed()
    for i in range(len(tmp_data)):
        tmp_data[i]['data'] = np.array(PIL.Image.fromarray(padded_data[i]).resize((224,224)))
    #pkl.dump(tmp_data[:int(len(tmp_data)*0.9)], open("/mfs/home/wangke/magnetic-detect/data/data99_train.pkl", 'wb'))
    #pkl.dump(tmp_data[-int(0.1*len(tmp_data)):], open("/mfs/home/wangke/magnetic-detect/data/data99_val.pkl", 'wb'))
    hist_features = []
    steps = 11
    hist_step = np.linspace(-5*np.array(varations).mean(), 5*np.array(varations).mean(), steps)
    for i in range(len(tmp_data)):
        hist_features.append(streched_data[i])
    #mag_data = pd.read_csv("/mfs/home/wangke/magnetic-detect/data/data0.csv")
    embed()
    '''
    #_X = _X.T
    #Y = Y.values.reshape(-1, 1)
    nn_X = torch.autograd.Variable(torch.FloatTensor(_X))
    nn_X2 = torch.autograd.Variable(torch.FloatTensor(__X))
    nn_Y = torch.autograd.Variable(torch.FloatTensor(Y))
    whole_dataset = Data.TensorDataset(nn_X, nn_X2, nn_Y)
    train_dataset, validate_dataset = Data.random_split(whole_dataset, (len(whole_dataset)-int(len(whole_dataset)*0.1),int(len(whole_dataset)*0.1)))
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
    input_size = nn_X.shape[1]                          
    hidden_size = nn_X.shape[1]*args.hidden_width_scaler
    hidden_depth = args.hidden_depth
    output_size = nn_Y.shape[1]
    #model = NeuralNet(input_size, hidden_size, hidden_depth, output_size, device).to(device)
    model = MagNet(100*100, 3, device).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    #embed()
    #Train and Validate:
    train_loss_history = []
    validate_loss_history = []
    for epoch in range(int(args.max_epoch+1)):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        validate_loss, validate_outputs, validate_targets = validate(args, model, device, validate_loader)
        train_loss_history.append(train_loss)
        validate_loss_history.append(validate_loss)
        train_loss_history[0] = validate_loss_history[0]
    NN_friction_predicted = np.array(model.cpu()((nn_X, nn_X2)).detach())
    plt.plot(train_loss_history)
    plt.plot(validate_loss_history)
    plt.show()
    embed()
    #visual(NN_friction_predicted)

    print("Names note:", names_note)
    print("Poly:", opt)
    print("BGD:", params)
    print("NN:", "NONE")
    
