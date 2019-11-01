import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import data_stuff
from IPython import embed


def get_model():
    model = torch.load("../models/NN_weights")
    return model

def get_compensate_force(raw_plan, stayed_plan, index_to_work_on):
    #Take out what we need:
    raw_data_X = np.empty(shape=(0, len(stayed_plan)))
    for i in [0,1,2,3,4,5]:
        raw_data_X = np.vstack((raw_data_X, [stayed_plan['axc_speed_%s'%i], stayed_plan['axc_pos_%s'%i], stayed_plan['axc_torque_ffw_gravity_%s'%i], stayed_plan['axc_torque_ffw_%s'%i]]))
    raw_data_X = np.vstack((raw_data_X, stayed_plan['Temp']))
    raw_data_Y = stayed_plan['need_to_compensate'].values

    #Normalize data:
    normer = data_stuff.normalizer(raw_data_X)
    normed_data_X, normed_data_Y = normer.normalize_XY(raw_data_X, raw_data_Y)

    #Get model:
    model = get_model()

    #Forward to get output:
    input = torch.FloatTensor(normed_data_X.T)
    output = model.cpu()(input).detach().numpy()
    #error_rate_low, error_rate_high = evaluate.evaluate_error_rate(args, outputs, normed_data_Y, normer, stayed_plan)

    #Denormalize Safety restrictions:
    reference = normer.denormalize_Y(normed_data_Y)
    compensate = normer.denormalize_Y(output)
    compensate = np.clip(compensate, -args.max_force, args.max_force)

    #Insert back on all series:
    compensate_full_series = np.zeros(len(raw_plan))
    compensate_full_series[index_to_work_on] = compensate.reshape(-1)
    reference_full_series = raw_plan['need_to_compensate'].values
    return compensate_full_series, reference_full_series

if __name__ == "__main__":
    print("Deploy time NN model...")

    #Configs:
    parser = argparse.ArgumentParser(description='Friction.')
    parser.add_argument('--max_force', type=int, default = 1)
    parser.add_argument('--axis_num', type=int, default = 4)
    parser.add_argument('--data_path', type=str, default = "../data/planning.csv")
    parser.add_argument('--on_whole_data', "-W", action='store_true', default=False, help="whether let engine to work on the entire data instead of just on filtered data(as train), default is false due to possible unstableness")
    parser.add_argument('--VISUALIZATION', "-V", action='store_true', default=False)
    args = parser.parse_args()
    args.axis_num = args.axis_num - 1    #joint index

    #Get plans:
    raw_plan, stayed_plan, stay_index = data_stuff.get_planning_data_all(args.data_path, args.axis_num, run_on_whole_data=args.on_whole_data)

    #Get result:
    compensate, reference = get_compensate_force(raw_plan, stayed_plan, stay_index)

    #Visualize:
    if args.VISUALIZATION:
        plt.plot(reference, label='reference')
        plt.plot(compensate, label='predicted')
        plt.legend()
        plt.show()

    np.savetxt("NN_compensation.txt", compensate)
    print("Done")

