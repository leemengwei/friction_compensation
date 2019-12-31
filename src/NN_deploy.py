#python NN_deploy.py  -V --mode='low_high'
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import data_stuff
from IPython import embed

def get_part1_model():
    model_path = "../models/NN_weights_%s"%args.mode.split('_')[0]
    print("Loading part1 model:%s"%model_path)
    model = torch.load(model_path, map_location=torch.device('cpu') if torch.cuda.is_available() is False else torch.device('cuda'))
    return model

def get_part2_model():
    model_path = "../models/NN_weights_%s"%args.mode.split('_')[1]
    print("Loading part2 model:%s"%model_path)
    model = torch.load(model_path, map_location=torch.device('cpu') if torch.cuda.is_available() is False else torch.device('cuda'))
    return model

def to_C(model_part1, model_part2, inputs):
    #Trace with jit:
    traced_module1 = torch.jit.trace(model_part1, inputs)
    traced_module2 = torch.jit.trace(model_part2, inputs)
    model1_path = "../models/NN_weights_%s_C.pt"%args.mode.split('_')[0]
    model2_path = "../models/NN_weights_%s_C.pt"%args.mode.split('_')[1]
    traced_module1.save(model1_path)
    traced_module2.save(model2_path)
    print("C models saved")

def get_compensate_force(raw_plan, part1_index, part2_index):
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
    model_part1 = get_part1_model()
    model_part2 = get_part2_model()

    #Forward to get output:
    inputs = torch.FloatTensor(normed_data_X.T)
    output_part1 = model_part1.cpu()(inputs[part1_index]).detach().numpy()
    output_part2 = model_part2.cpu()(inputs[part2_index]).detach().numpy()
    reference = normer.denormalize_Y(normed_data_Y)
    compensate_part1 = normer.denormalize_Y(output_part1)
    compensate_part2 = normer.denormalize_Y(output_part2)

    #Save for Production:
    to_C(model_part1, model_part2, inputs)

    #Denormalize Safety restrictions:
    compensate_part1 = np.clip(compensate_part1, -args.max_force, args.max_force)
    compensate_part2 = np.clip(compensate_part2, -args.max_force, args.max_force)

    #Compose together:
    #TODO: solve the switch point.
    compensate_full_series = np.zeros(len(raw_plan))
    compensate_full_series[part1_index] = compensate_part1.reshape(-1)
    compensate_full_series[part2_index] = compensate_part2.reshape(-1)
    reference_full_series = raw_plan['need_to_compensate'].values
    return compensate_full_series, reference_full_series

if __name__ == "__main__":
    print("Deploy time NN model...")

    #Configs:
    parser = argparse.ArgumentParser(description='Friction.')
    parser.add_argument('--max_force', type=int, default = 1)
    parser.add_argument('--axis_num', type=int, default = 4)
    parser.add_argument('--mode', type=str, choices=["acc_uniform", "low_high"], required=True)
    parser.add_argument('--data_path', type=str, default = "../data/planning.csv")
    parser.add_argument('--VISUALIZATION', "-V", action='store_true', default=False)
    args = parser.parse_args()
    args.axis_num = args.axis_num - 1    #joint index

    #Get plans:
    mode = ['deploy', args.mode]
    raw_plan, part1_index, part2_index = data_stuff.get_data(args, mode)

    #Get result:
    compensate, reference = get_compensate_force(raw_plan, part1_index, part2_index)

    #Visualize:
    if args.VISUALIZATION:
        plt.plot(reference, label='reference_feedback', alpha=0.5)
        plt.scatter(part1_index, compensate[part1_index], label='part1_predicted_feedback', color='green', s=1)
        plt.scatter(part2_index, compensate[part2_index], label='part2_predicted_feedback', color='red', s=1)
        plt.legend()
        plt.show()

    np.savetxt("../output/NN_compensation.txt", compensate)
    print("Done")

