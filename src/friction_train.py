#coding:utf-8
import time
from IPython import embed
import numpy as np
import os,sys,time

import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
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
import NN_model

import pickle as pkl
import PIL
import torch
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
torch.manual_seed(44)

plt.ion()
DEBUG = False
DEBUG = True
epsilon = 1e-10

def evaluate_error_rate(outputs, data, normer, showup=False):
    outputs = normer.denormlize(outputs)
    value_threshold = 0.01
    low_value_region = np.where(np.abs(data['servo_feedback_torque_3'])<value_threshold)
    high_value_region = np.where(np.abs(data['servo_feedback_torque_3'])>=value_threshold)
    if args.VISUALIZATION and showup:
        plt.hist(data['servo_feedback_torque_3'].values[low_value_region], bins=50)
        plt.hist(data['servo_feedback_torque_3'].values[high_value_region], bins=1000)
        plt.title("Split of feedback (as denominator)")
        plt.show()
    error_rate_low = np.abs((data.iloc[low_value_region]['axc_torque_ffw_gravity_3'] + outputs[low_value_region] - data.iloc[low_value_region]['servo_feedback_torque_3'])/(data.iloc[low_value_region]['servo_feedback_torque_3']+epsilon)).mean()*100
    error_rate_high = np.abs((data.iloc[high_value_region]['axc_torque_ffw_gravity_3'] + outputs[high_value_region] - data.iloc[high_value_region]['servo_feedback_torque_3'])/(data.iloc[high_value_region]['servo_feedback_torque_3']+epsilon)).mean()*100
    return (error_rate_low, error_rate_high)


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
    return validate_loss_mean, output, target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Friction.')
    parser.add_argument('--mode', '-M', default='linear')
    parser.add_argument('--learning_rate', '-LR', type=float, default=0.01)
    parser.add_argument('--test_ratio', '-TR', type=float, default=0.4)
    parser.add_argument('--criterion', '-C', type=float, default=1e-5)
    parser.add_argument('--max_epoch', '-E', type=int, default=100)

    parser.add_argument('--hidden_width_scaler', type=int, default = 10)
    parser.add_argument('--hidden_depth', type=int, default = 3)
    parser.add_argument('--Cuda_number', type=int, default = 0)
    parser.add_argument('--num_of_batch', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--VISUALIZATION', action='store_true', default=False)
    args = parser.parse_args()
    
    print("Start...%s"%args)
    #Get data:
    data = data_stuff.get_standarlized_data()
    batch_size = data.shape[0]
    Y = data['need_to_compensate']
    #Input variables for models:
    #When in plan we only have:
    _X_planned = np.array([data['axc_speed_3'], data['axc_torque_ffw_gravity_3'], data['Temp'], data['axc_pos_3']])   #TODO: this is axis 4
    #In use we MUST have real_time_feedback, compensate from axc_torque to feedback_torque:
    _X_feedbacked = np.array([data['servo_feedback_speed_3'], data['axc_torque_ffw_gravity_3'], data['Temp'], data['servo_feedback_pos_3']])  #TODO: this is for axis 4
    X = np.insert(_X_feedbacked, 0, 1, axis=0)
    normer = data_stuff.normalizer(X, Y)
    X, Y = normer.normalize(X, Y)
    #embed()

    #---------------------------------Do polyfit---------------------------:
    #Init unknown params:
    if args.mode == 'linear':
        names_note = ['c0', 'c1', 'c2', 'c3', 'c4']
        params = np.array([-1.83062977e-09, 1.50000000e+00, 6.82213855e-09, 3.00000000e+00, -4.12101372e-09])
        params = np.array([1,1,1,1,1])
        opt, cov = curve_fit(classical_model.linear_friction_model, X, Y, maxfev=500000)
    else:
        names_note = ['c0', 'v_brk', 'F_brk', 'F_C', 'c1', 'c2', 'c3', 'c4']
        params = np.array([-1.78197004e-09, 1.99621299e+00, 9.82765471e-08, -6.02984398e-09, 1.49999993e+00, 1.78627117e-09, 5.00000000e+00, 2.69899575e-09])
        params = np.array([1,1,1,1,1,1,1,1])
        opt, cov = curve_fit(classical_model.nonlinear_friction_model, X, Y, maxfev=500000)
    poly_loss, _J, poly_friction_predicted, _, _ = classical_model.compute_loss(Y, X, opt, args)
    error_ratio = evaluate_error_rate(poly_friction_predicted, data, normer, showup=True)
    plot_utils.visual(Y.values, poly_friction_predicted, "poly", args, title=error_ratio)
    print("Names note:", names_note, 'J:', _J)
    print("Poly:", opt)
    print("Error rate:", error_ratio)

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
    #    error_ratio = evaluate_error_rate(BGD_friction_predicted, data, normer)
    #    plot_utils.visual(Y.values, BGD_friction_predicted, 'BGD', args, title=error_ratio)
    #print("Names note:", names_note)
    #print("BGD:", params)
    #print("Error rate:", error_ratio)
    #embed()
    #sys.exit()

    #-------------------------------------Do NN--------------------------------:
    #Data:
    _X = _X_feedbacked.T
    Y = Y.values.reshape(-1, 1)
    nn_X = torch.autograd.Variable(torch.FloatTensor(_X))
    nn_Y = torch.autograd.Variable(torch.FloatTensor(Y))
    #mannual split dataset:
    all_index = list(range(len(nn_Y)))
    val_index = list(range(int(len(nn_Y)*(0.5-args.test_ratio/2)), int(len(nn_Y)*(0.5+args.test_ratio/2))))
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
    model = NN_model.NeuralNet(input_size, hidden_size, hidden_depth, output_size, device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    #embed()
    #Train and Validate:
    train_loss_history = []
    validate_loss_history = []
    for epoch in range(int(args.max_epoch+1)):
        NN_friction_predicted = np.array(model.cpu()((nn_X)).detach()).reshape(-1)
        error_ratio_val = evaluate_error_rate(NN_friction_predicted[val_index], data.iloc[val_index], normer)
        error_ratio_train = evaluate_error_rate(NN_friction_predicted[train_index], data.iloc[train_index], normer)
        plot_utils.visual(Y, NN_friction_predicted, 'NN', args, extra=[train_index, val_index], title=error_ratio_val)

        train_loss, train_outputs, train_targets = train(args, model, device, train_loader, optimizer, epoch)
        validate_loss, validate_outputs, validate_targets = validate(args, model, device, validate_loader)
        train_loss_history.append(train_loss)
        validate_loss_history.append(validate_loss)
        train_loss_history[0] = validate_loss_history[0]
        print("Epoch: %s"%epoch)
        print("Train set  Average loss: {:.8f}".format(train_loss), "error ratio:", error_ratio_train)
        print('Validate set Average loss: {:.8f}'.format(validate_loss), "error ratio:", error_ratio_val)
    if args.VISUALIZATION:
        plt.plot(train_loss_history)
        plt.plot(validate_loss_history)

    names_note = "NN weights"
    print("Names note:", names_note)
    print("NN:", "NONE")
    print("Error rate:", error_ratio)

    embed()
    sys.exit()
   
