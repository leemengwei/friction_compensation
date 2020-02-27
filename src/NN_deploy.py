#python NN_deploy.py  -V --mode='low_high'
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import data_stuff
from IPython import embed
import evaluate
import NN_model
import warnings
import time
import copy
import plot_utils
warnings.filterwarnings("ignore")

def get_part_model(X, name):
    model_path = "../models/NN_weights_best_%s"%name
    print("Loading part model:%s"%model_path)
    #model = torch.load(model_path, map_location=torch.device(device_type))
    #model = NN_model.NeuralNet(input_size=25, hidden_size=25, hidden_depth=3, output_size=1, device=torch.device(device_type))
    model = NN_model.NeuralNetSimple(input_size=X.shape[0], hidden_size=X.shape[0]*5, hidden_depth=3, output_size=1, device=torch.device(device_type))
    model.load_state_dict(torch.load(model_path).state_dict())
    embed()
    model = model.to(device_type)
    return model

def to_C(args, model_part1, model_part2, inputs):
    if 'all' not in args.mode:
        name1 = args.mode.split('_')[0]     #'acc'
        name2 = args.mode.split('_')[1]     #'uniform'
    else:   # 'acc_uniform_all', 'low_high_all'
        name1 = args.mode.split('_')[-1]    #'all'
        name2 = args.mode.split('_')[-1]    #'all' as well
    #Trace with jit:
    traced_module1 = torch.jit.trace(model_part1, inputs)
    traced_module2 = torch.jit.trace(model_part2, inputs)
    model1_path = "../models/NN_weights_%s_C.pt"%name1
    model2_path = "../models/NN_weights_%s_C.pt"%name2
    traced_module1.save(model1_path)
    traced_module2.save(model2_path)
    print("C models saved")

def get_compensate_force(args, raw_plan, part1_index, part2_index):
    #Take out what we need:
    #Make inputs:
    raw_data_X = np.empty(shape=(0, len(raw_plan)))
    input_columns_names = []
    for i in [args.axis_num]:
        input_columns_names += ['axc_pos_%s'%i]
    for i in [args.axis_num]:
        input_columns_names += ['axc_speed_%s'%i]
    for i in [args.axis_num]:
        input_columns_names += ['axc_torque_ffw_gravity_%s'%i]
    for i in [args.axis_num]:
        input_columns_names += ['axc_torque_ffw_%s'%i]
    input_columns_names += ['Temp']
    raw_data_X = raw_plan[input_columns_names].values.T
    raw_data_Y = raw_plan['need_to_compensate'].values
   
    #Normalize data:
    normer = data_stuff.normalizer(raw_data_X)
    normed_data_X, normed_data_Y = normer.normalize_XY(raw_data_X, raw_data_Y)

    #Get model:
    if 'all' not in args.mode:
        model_part1 = get_part_model(normed_data_X, args.mode.split('_')[0])     #'acc'
        model_part2 = get_part_model(normed_data_X, args.mode.split('_')[1])     #'uniform'
    else:   # 'acc_uniform_all', 'low_high_all'
        model_part1 = get_part_model(normed_data_X, args.mode.split('_')[-1])    #'all'
        model_part2 = get_part_model(normed_data_X, args.mode.split('_')[-1])    #'all' as well

    #Forward to get output:
    inputs = torch.FloatTensor(normed_data_X.T).to(device_type)
    output_part1 = model_part1(inputs[part1_index]).detach().cpu().numpy()
    output_part2 = model_part2(inputs[part2_index]).detach().cpu().numpy()
    #Speed test:
    if args.speed_test:
        inputs[:,:]=1
        #On default device:
        for i in range(10): 
            model_part1(inputs[:args.buffer_length])
        tic = time.time() 
        for i in range(1000): 
            model_part1(inputs[:args.buffer_length])
        print((time.time()-tic)/1000*1000, "ms on %s"%device_type)  
        #On cpu deivce:
        cpu_model = copy.deepcopy(model_part1).cpu()
        cpu_inputs = inputs.cpu()
        for i in range(10): 
            cpu_model(cpu_inputs[:args.buffer_length])
        tic = time.time() 
        for i in range(1000): 
            cpu_model(cpu_inputs[:args.buffer_length])
        print((time.time()-tic)/1000*1000, "ms on cpu")  

    #import my_analyzer
    #my_analyzer.performance_shape(raw_plan, inputs, model_part1)
    #Compose together:
    #TODO: solve the switch point.
    output_full_series = np.zeros(len(raw_plan))
    output_full_series[part1_index] = output_part1.reshape(-1)
    output_full_series[part2_index] = output_part2.reshape(-1)

    #Denormalize and Safety restrictions:
    reference_full_series = raw_plan['need_to_compensate'].values    #show be same with reference = normer.denormalize_Y(normed_data_Y)
    compensate_full_series = normer.denormalize_Y(output_full_series)
    compensate_full_series = np.clip(compensate_full_series, -args.max_force, args.max_force)

    #Save for Production:
    to_C(args, model_part1.cpu(), model_part2.cpu(), inputs.cpu())
    error_original, _, error_treated = evaluate.evaluate_error_rate(args, output_full_series, normed_data_Y, normer, raw_plan, showup=False)
    model_returned = model_part1
    return compensate_full_series, reference_full_series, error_original, error_treated, model_returned, inputs

if __name__ == "__main__":
    print("Deploy time NN model...")

    #Configs:
    parser = argparse.ArgumentParser(description='Friction.')
    parser.add_argument('--max_force', type=int, default = 1)
    parser.add_argument('--axis_num', type=int, default = 4)
    parser.add_argument('--buffer_length', type=int, default = 6)
    parser.add_argument('--mode', type=str, choices=["acc_uniform", "low_high", "acc_uniform_all", "low_high_all"], required=True)
    parser.add_argument('--data_path', type=str, default = "../data/planning.csv")
    parser.add_argument('--Quick_data', '-Q', action='store_true', default=False)
    parser.add_argument('--VISUALIZATION', "-V", action='store_true', default=False)
    parser.add_argument('--speed_test', "-SPD", action='store_true', default=False)
    parser.add_argument('--no_cuda', action='store_true', default=False)
    args = parser.parse_args()
    args.axis_num = args.axis_num - 1    #joint index
    cuda_is_available = torch.cuda.is_available()
    if args.no_cuda:
        cuda_is_available = False
    if not cuda_is_available:
        device_type = 'cpu'
        print("Using cpu")
    else:
        device_type = 'cuda'
        print("Using gpu")

    #Get plans:
    mode = ['deploy', args.mode]
    raw_plan, part1_index, part2_index = data_stuff.get_data(args, mode)

    #Data_preprocess, and, Get result:
    compensate, reference, error_original, error_treated, model_returned, inputs = get_compensate_force(args, raw_plan, part1_index, part2_index)

    #Visualize:
    if args.VISUALIZATION:
        plt.plot(reference, label='reference_feedback', alpha=0.5)
        plt.scatter(range(len(raw_plan)), raw_plan['axc_torque_ffw_gravity_%s'%args.axis_num], label='original', color='gray', s=1, alpha=0.5)
        plt.scatter(part1_index, compensate[part1_index], label='part1_predicted_feedback', color='green', s=1)
        plt.scatter(part2_index, compensate[part2_index], label='part2_predicted_feedback', color='red', s=1)
        plt.legend()
        plt.title("Error treated: {0:.2f}%, original:{1:.2f}%".format(error_treated, error_original))
        print("One of the reason it may shows higher error rate than expected is because some uniform speed range be ommited during train/val evaluation")
        plt.show()

    np.savetxt("../output/NN_compensation.txt", compensate)
    print("Done")

    response_surface = True
    if response_surface and args.VISUALIZATION:
        plot_utils.response_surface(model_returned, inputs)





