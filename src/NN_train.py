#神经网络训练主程序：
#输出模型文件说明：
#同样，声明训练数据的分割方式后，（mode=匀速加速or低速高速），仍然将需要NN针对不同mode训练两个阶段的不同的模型。但此时由于迭代训练速度很慢，所以这里不再像拟合传统数学模型那样串行训练，而是进一步根据further_mode，执行一次，得到具体的关于该分割mode下哪阶段的model。
#如下分别得到四个：
#python NN_train.py  --mode=acc_uniform -Q --further_mode=acc
#python NN_train.py  --mode=acc_uniform -Q --further_mode=uniform
#python NN_train.py  --mode=low_high -Q --further_mode=low
#python NN_train.py  --mode=low_high -Q --further_mode=high


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
import scipy
import argparse
from tqdm import tqdm

#my modules:
import data_stuff
import NN_model
import plot_utils
import evaluate

import pickle as pkl
import PIL
import torch
import torch.utils.data as Data
import torch.optim as optim
import torch.nn.functional as F
torch.manual_seed(44)

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
    parser.add_argument('--learning_rate', '-LR', type=float, default=1e-2)
    parser.add_argument('--test_ratio', '-TR', type=float, default=0.2)
    parser.add_argument('--max_epoch', '-E', type=int, default=100)

    parser.add_argument('--hidden_width_scaler', type=int, default = 1)
    parser.add_argument('--hidden_depth', type=int, default = 3)
    parser.add_argument('--axis_num', type=int, default = 4)
    parser.add_argument('--Cuda_number', type=int, default = 0)
    parser.add_argument('--num_of_batch', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--VISUALIZATION', "-V", action='store_true', default=False)
    parser.add_argument('--NO_CUDA', action='store_true', default=False)
    parser.add_argument('--Quick_data', "-Q", action='store_true', default=False)
    parser.add_argument('--mode', type=str, choices=["acc_uniform", "low_high"], required=True)
    parser.add_argument('--further_mode', type=str, choices=["acc", "uniform", "low", "high"], required=True)
    args = parser.parse_args()
    args.axis_num = args.axis_num - 1
    
    print("Start...%s"%args)
    if not args.NO_CUDA:
        device = torch.device("cuda", args.Cuda_number)
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    #-------------------------------------Do NN--------------------------------:
    #Get data:
    mode = ['train', args.mode]
    raw_data, part1_index, part2_index = data_stuff.get_data(args, mode)
    this_part = part1_index if args.further_mode=="acc" or args.further_mode=="low" else part2_index

    #Take variables we concerned:
    print("PART start...%s"%args.further_mode)
    raw_data_part = raw_data.iloc[this_part]
    data_X = np.empty(shape=(0, len(raw_data_part)))
    input_columns_names = []
    for i in [0,1,2,3,4,5]:
        input_columns_names += ['axc_speed_%s'%i, 'axc_pos_%s'%i, 'axc_torque_ffw_gravity_%s'%i, 'axc_torque_ffw_%s'%i]
    input_columns_names += ['Temp']
    output_columns_names = ['need_to_compensate']
    data_X = raw_data_part[input_columns_names].values.T
    data_Y = raw_data_part[output_columns_names].values
    print("Shape of all input: %s, shape of all output: %s"%(data_X.shape, data_Y.shape))

    #Normalize data:
    normer = data_stuff.normalizer(data_X)
    X_normed, Y_normed = normer.normalize_XY(data_X, data_Y)

    #MUST CHECK INPUT DISTRIBUTION!!!!!!
    print("Checking distribution of all data...")
    plot_utils.check_distribution(X_normed, input_columns_names, args)
    plot_utils.check_distribution(Y_normed.T, output_columns_names, args)

    #mannual split dataset:    #Don't push dataset on cuda now, do so later after form dataset_loader
    X_train, Y_train, X_val, Y_val, raw_data_train, raw_data_val = data_stuff.split_dataset(args, X_normed, Y_normed, raw_data_part)
    nn_X_train = torch.autograd.Variable(torch.FloatTensor(X_train.T))
    nn_Y_train = torch.autograd.Variable(torch.FloatTensor(Y_train)).reshape(-1,1)
    nn_X_val =   torch.autograd.Variable(torch.FloatTensor(X_val.T))
    nn_Y_val =   torch.autograd.Variable(torch.FloatTensor(Y_val)).reshape(-1,1)
    #Just for showup high/low region
    _ = evaluate.evaluate_error_rate(args, Y_val.reshape(-1)*0, Y_val, normer, raw_data_val, showup=True)
    
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
    input_size = nn_X_train.shape[1]                          
    hidden_size = nn_X_train.shape[1]*args.hidden_width_scaler
    hidden_depth = args.hidden_depth
    output_size = nn_Y_train.shape[1]
    model = NN_model.NeuralNet(input_size, hidden_size, hidden_depth, output_size, device)
    #optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    print(model)

    #Train and Validate:
    print("Now training...")
    plt.ion()
    train_loss_history = []
    validate_loss_history = []
    train_error_history = []
    validate_error_history = []
    plt.figure(figsize=(14, 8))
    for epoch in range(int(args.max_epoch+1)):
        print("Epoch: %s"%epoch, "TEST AND SAVE FIRST")
        #Test first:
        #Push ALL data together through the network
        predicted_train = np.array(model((nn_X_train.to(device))).detach().cpu()).reshape(-1)
        predicted_val = np.array(model((nn_X_val.to(device))).detach().cpu()).reshape(-1)
        error_ratio_train = evaluate.evaluate_error_rate(args, predicted_train, nn_Y_train, normer, raw_data_train)
        error_ratio_val = evaluate.evaluate_error_rate(args, predicted_val, nn_Y_val, normer, raw_data_val)
        train_error_history.append(error_ratio_train[1])
        validate_error_history.append(error_ratio_val[1])
        model.eval()
        if epoch>10:
            if validate_error_history[-1] < np.array(validate_error_history[:-1]).min():
                torch.save(model, "../models/NN_weights_best_%s"%args.further_mode)
        print("Train set error ratio:", error_ratio_train)
        print("Validate set error ratio:", error_ratio_val)
        plot_utils.visual(nn_Y_val, predicted_val, 'NN', args, title=error_ratio_val, epoch=epoch)
        #Train/Val then:
        train_loss, train_outputs, train_targets = train(args, model, device, train_loader, optimizer, epoch)
        validate_loss, validate_outputs, validate_targets = validate(args, model, device, validate_loader)
        #Infos:
        train_loss_history.append(train_loss)
        validate_loss_history.append(validate_loss)
        train_loss_history[0] = validate_loss_history[0]
        print("Train set  Average loss: {:.8f}".format(train_loss))
        print('Validate set Average loss: {:.8f}'.format(validate_loss))
    if True:
        plt.ioff()
        plt.clf()
        plt.plot(train_loss_history, label='train loss')
        plt.plot(validate_loss_history, label = 'val loss')
        plt.title("Train/Val loss history")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    names_note = "NN weights"
    print("Names note:", names_note)
    print("NN:", "NONE")
    print("Error rate:", error_ratio_val)

    torch.save(model, "../models/NN_weights_%s"%args.further_mode)
    #embed()


