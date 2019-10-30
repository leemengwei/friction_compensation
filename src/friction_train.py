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

DEBUG = False
DEBUG = True
epsilon = 1e-10

def evaluate_error_rate(outputs, targets, normer, raw_data, showup=False):
    outputs = normer.denormlize(outputs)
    targets = normer.denormlize(targets)
    value_threshold = 0.01

    _plan = raw_data['axc_torque_ffw_gravity_%s'%args.axis_num].values
    _compensate = outputs
    _meassure = raw_data['servo_feedback_torque_%s'%args.axis_num].values  #same as plan+target

    low_value_region = np.where(np.abs(_meassure)<value_threshold)
    high_value_region = np.where(np.abs(_meassure)>=value_threshold)
   
    error_rate_low = np.abs((_plan[low_value_region] + _compensate[low_value_region] - _meassure[low_value_region])/(_meassure[low_value_region]+epsilon)).mean()*100
    error_rate_high = np.abs((_plan[high_value_region] + _compensate[high_value_region] - _meassure[high_value_region])/(_meassure[high_value_region]+epsilon)).mean()*100
    if args.VISUALIZATION and showup:
        bins = np.linspace(targets.min(), targets.max(), 1000)
        plt.hist(_meassure[low_value_region], bins=bins, color='gray', label="low values")
        plt.hist(_meassure[high_value_region], bins=bins, color='black', label='high values')
        plt.title("High/Low value seperation in val set")
        plt.legend()
        plt.show()
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

def split_dataset(X, Y, raw_data):
    all_index = list(range(len(Y)))
    val_index = list(range(int(len(Y)*(0.5-args.test_ratio/2)), int(len(Y)*(0.5+args.test_ratio/2))))
    train_index = list(set(all_index) - set(val_index))
    X_train = X[:, train_index]
    Y_train = Y[train_index]
    X_val = X[:, val_index]
    Y_val = Y[val_index]
    if args.VISUALIZATION:
        plt.scatter(np.array(range(len(Y)))[train_index][::10], Y_train[::10], s=0.5, label="train")
        plt.scatter(np.array(range(len(Y)))[val_index][::10], Y_val[::10], s=0.5, label="val")
        plt.xlabel("Sample points")
        plt.ylabel("Forces need to compensate")
        plt.title("Split of rain and val set")
        plt.legend()
        plt.show()
    raw_data_train = raw_data.iloc[train_index]
    raw_data_val = raw_data.iloc[val_index]
    return X_train, Y_train, X_val, Y_val, raw_data_train, raw_data_val

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Friction.')
    parser.add_argument('--mode', '-M', default='linear')
    parser.add_argument('--learning_rate', '-LR', type=float, default=0.01)
    parser.add_argument('--test_ratio', '-TR', type=float, default=0.2)
    parser.add_argument('--criterion', '-C', type=float, default=1e-5)
    parser.add_argument('--max_epoch', '-E', type=int, default=100)

    parser.add_argument('--hidden_width_scaler', type=int, default = 10)
    parser.add_argument('--hidden_depth', type=int, default = 3)
    parser.add_argument('--axis_num', type=int, default = 4)
    parser.add_argument('--Cuda_number', type=int, default = 0)
    parser.add_argument('--num_of_batch', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--VISUALIZATION', action='store_true', default=False)
    parser.add_argument('--NO_CUDA', action='store_true', default=False)
    args = parser.parse_args()
    args.axis_num = args.axis_num - 1
    
    print("Start...%s"%args)
    #Get data:
    raw_data = data_stuff.get_useful_data()
    data_Y = raw_data['need_to_compensate'].values

    #Take variables we concerned:
    #Use planned data:
    data_X = np.array([raw_data['axc_speed_%s'%args.axis_num], raw_data['axc_torque_ffw_gravity_%s'%args.axis_num], raw_data['Temp'], raw_data['axc_pos_%s'%args.axis_num]])   #TODO: this is axis 4
    #Use real-time data:
    #data_X = np.array([raw_data['servo_feedback_speed_%s'%args.axis_num], raw_data['axc_torque_ffw_gravity_%s'%args.axis_num], raw_data['Temp'], raw_data['servo_feedback_pos_%s'%args.axis_num]])  #TODO: this is for axis 4

    poly_data_X = np.insert(data_X, 0, 1, axis=0)
    #Normalize data:
    normer = data_stuff.normalizer(poly_data_X, data_Y)
    poly_X_normed, Y_normed = normer.normalize(poly_data_X, data_Y)

    #mannual split dataset:
    X_train, Y_train, X_val, Y_val, raw_data_train, raw_data_val = split_dataset(poly_X_normed, Y_normed, raw_data)

    #---------------------------------Do polyfit---------------------------:
    #Init unknown params:
    if args.mode == 'linear':
        names_note = ['c0', 'c1', 'c2', 'c3', 'c4']
        params = np.array([-1.83062977e-09, 1.50000000e+00, 6.82213855e-09, 3.00000000e+00, -4.12101372e-09])
        params = np.array([1,1,1,1,1])
        opt, cov = curve_fit(classical_model.linear_friction_model, X_train, Y_train, maxfev=500000)
    else:
        names_note = ['c0', 'v_brk', 'F_brk', 'F_C', 'c1', 'c2', 'c3', 'c4']
        params = np.array([-1.78197004e-09, 1.99621299e+00, 9.82765471e-08, -6.02984398e-09, 1.49999993e+00, 1.78627117e-09, 5.00000000e+00, 2.69899575e-09])
        params = np.array([1,1,1,1,1,1,1,1])
        opt, cov = curve_fit(classical_model.nonlinear_friction_model, X_train, Y_train, maxfev=500000)
    poly_loss, _J, poly_friction_predicted, _, _ = classical_model.compute_loss(Y_val, X_val, opt, args)
    error_ratio = evaluate_error_rate(poly_friction_predicted, Y_val, normer, raw_data_val, showup=True)
    plot_utils.visual(Y_val, poly_friction_predicted, "poly", args, title=error_ratio)
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
    plt.ion()
    #Get data:
    raw_data = data_stuff.get_useful_data()
    data_Y = raw_data['need_to_compensate'].values

    #Take variables we concerned:
    #Use planned data:
    data_X = np.empty(shape=(0, len(raw_data)))
    for i in [0,1,2,3,4,5]:
        data_X = np.vstack((data_X, [raw_data['axc_speed_%s'%i], raw_data['axc_pos_%s'%i], raw_data['axc_torque_ffw_gravity_%s'%i], raw_data['axc_torque_ffw_%s'%i]]))
    data_X = np.vstack((data_X, raw_data['Temp']))
    #Use real-time data:
    #data_X = np.empty(shape=(0, len(raw_data)))
    #for i in [0,1,2,3,4,5]:
    #    data_X = np.vstack((raw_data_X, [raw_data['servo_feedback_speed_%s'%i], raw_data['servo_feedback_pos_%s'%i], raw_data['axc_torque_ffw_gravity_%s'%i],  raw_data['axc_torque_ffw_%s'%i]]))
    #data_X = np.vstack((data_X, raw_data['Temp']))

    #Normalize data:
    normer = data_stuff.normalizer(data_X, data_Y)
    X_normed, Y_normed = normer.normalize(data_X, data_Y)

    #mannual split dataset:
    X_train, Y_train, X_val, Y_val, raw_data_train, raw_data_val = split_dataset(X_normed, Y_normed, raw_data)
    nn_X_train = torch.autograd.Variable(torch.FloatTensor(X_train.T))
    nn_Y_train = torch.autograd.Variable(torch.FloatTensor(Y_train)).reshape(-1,1)
    nn_X_val =   torch.autograd.Variable(torch.FloatTensor(X_val.T))
    nn_Y_val =   torch.autograd.Variable(torch.FloatTensor(Y_val)).reshape(-1,1)
    
    #Form pytorch dataset:
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
    if not args.NO_CUDA:
        device = torch.device("cuda", args.Cuda_number)
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    #device = torch.device("cpu")
    input_size = nn_X_train.shape[1]                          
    hidden_size = nn_X_train.shape[1]*args.hidden_width_scaler
    hidden_depth = args.hidden_depth
    output_size = nn_Y_train.shape[1]
    model = NN_model.NeuralNet(input_size, hidden_size, hidden_depth, output_size, device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    #Train and Validate:
    print("Now training...")
    train_loss_history = []
    validate_loss_history = []
    for epoch in range(int(args.max_epoch+1)):
        #Test first:
        predicted_train = np.array(model.cpu()((nn_X_train)).detach()).reshape(-1)
        predicted_val = np.array(model.cpu()((nn_X_val)).detach()).reshape(-1)
        error_ratio_train = evaluate_error_rate(predicted_train, nn_Y_train, normer, raw_data_train)
        error_ratio_val = evaluate_error_rate(predicted_val, nn_Y_val, normer, raw_data_val)
        plot_utils.visual(nn_Y_val, predicted_val, 'NN', args, title=error_ratio_val)
        #Train/Val then:
        train_loss, train_outputs, train_targets = train(args, model, device, train_loader, optimizer, epoch)
        validate_loss, validate_outputs, validate_targets = validate(args, model, device, validate_loader)
        #Infos:
        train_loss_history.append(train_loss)
        validate_loss_history.append(validate_loss)
        train_loss_history[0] = validate_loss_history[0]
        print("Epoch: %s"%epoch)
        print("Train set  Average loss: {:.8f}".format(train_loss), "error ratio:", error_ratio_train)
        print('Validate set Average loss: {:.8f}'.format(validate_loss), "error ratio:", error_ratio_val)
        #embed()
    if args.VISUALIZATION:
        plt.plot(train_loss_history)
        plt.plot(validate_loss_history)
        plt.title("Train/Val loss history")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()

    names_note = "NN weights"
    print("Names note:", names_note)
    print("NN:", "NONE")
    print("Error rate:", error_ratio)

    torch.save(model, "NN.pth")
    embed()
    sys.exit()
   



