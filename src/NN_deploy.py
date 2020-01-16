#python NN_deploy.py  -V --mode='low_high'
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import data_stuff
from IPython import embed
import evaluate

def get_part_model(name):
    model_path = "../models/NN_weights_best_%s"%name
    print("Loading part model:%s"%model_path)
    model = torch.load(model_path, map_location=torch.device('cpu') if torch.cuda.is_available() is False else torch.device('cuda'))
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
    raw_data_X = np.empty(shape=(0, len(raw_plan)))
    for i in [0,1,2,3,4,5]:
        raw_data_X = np.vstack((raw_data_X, [raw_plan['axc_speed_%s'%i], raw_plan['axc_pos_%s'%i], raw_plan['axc_torque_ffw_gravity_%s'%i], raw_plan['axc_torque_ffw_%s'%i]]))
    raw_data_X = np.vstack((raw_data_X, raw_plan['Temp']))
    raw_data_Y = raw_plan['need_to_compensate'].values
   
    #Normalize data:
    normer = data_stuff.normalizer(raw_data_X)
    normed_data_X, normed_data_Y = normer.normalize_XY(raw_data_X, raw_data_Y)

    #Get model:
    if 'all' not in args.mode:
        model_part1 = get_part_model(args.mode.split('_')[0])     #'acc'
        model_part2 = get_part_model(args.mode.split('_')[1])     #'uniform'
    else:   # 'acc_uniform_all', 'low_high_all'
        model_part1 = get_part_model(args.mode.split('_')[-1])    #'all'
        model_part2 = get_part_model(args.mode.split('_')[-1])    #'all' as well

    #Forward to get output:
    inputs = torch.FloatTensor(normed_data_X.T)
    output_part1 = model_part1.cpu()(inputs[part1_index]).detach().numpy()
    output_part2 = model_part2.cpu()(inputs[part2_index]).detach().numpy()
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
    to_C(args, model_part1, model_part2, inputs)
    error_original, _, error_treated = evaluate.evaluate_error_rate(args, output_full_series, normed_data_Y, normer, raw_plan, showup=False)
    return compensate_full_series, reference_full_series, error_original, error_treated

if __name__ == "__main__":
    print("Deploy time NN model...")

    #Configs:
    parser = argparse.ArgumentParser(description='Friction.')
    parser.add_argument('--max_force', type=int, default = 1)
    parser.add_argument('--axis_num', type=int, default = 4)
    parser.add_argument('--mode', type=str, choices=["acc_uniform", "low_high", "acc_uniform_all", "low_high_all"], required=True)
    parser.add_argument('--data_path', type=str, default = "../data/planning.csv")
    parser.add_argument('--VISUALIZATION', "-V", action='store_true', default=False)
    args = parser.parse_args()
    args.axis_num = args.axis_num - 1    #joint index

    #Get plans:
    mode = ['deploy', args.mode]
    raw_plan, part1_index, part2_index = data_stuff.get_data(args, mode)

    #Data_preprocess, and, Get result:
    compensate, reference, error_original, error_treated = get_compensate_force(args, raw_plan, part1_index, part2_index)

    #Visualize:
    if args.VISUALIZATION:
        plt.plot(reference, label='reference_feedback', alpha=0.5)
        plt.scatter(range(len(raw_plan)), raw_plan['axc_torque_ffw_gravity_%s'%args.axis_num], label='original', color='gray', s=1, alpha=0.5)
        plt.scatter(part1_index, compensate[part1_index], label='part1_predicted_feedback', color='green', s=1)
        plt.scatter(part2_index, compensate[part2_index], label='part2_predicted_feedback', color='red', s=1)
        plt.legend()
        plt.title("Error treated: {0:.2f}%, original:{1:.2f}%".format(error_treated, error_original))
        plt.show()

    np.savetxt("../output/NN_compensation.txt", compensate)
    print("Done")

