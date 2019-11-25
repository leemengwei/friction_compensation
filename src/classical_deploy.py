#python classical_deploy.py  --mode=acc_uniform -V
import matplotlib.pyplot as plt
import argparse
import numpy as np
import data_stuff
from IPython import embed
import classical_model

def get_part1_model_params(model):
    model_path = "../models/%s_weights_%s"%(model, args.mode.split('_')[0])
    print("Loading part1 model:%s"%model_path)
    params = np.loadtxt(model_path)
    return params

def get_part2_model_params(model):
    model_path = "../models/%s_weights_%s"%(model, args.mode.split('_')[1])
    print("Loading part2 model:%s"%model_path)
    params = np.loadtxt(model_path)
    return params

def get_compensate_force(raw_plan, part1_index, part2_index, model='linear'):
    #Take out what we need:
    raw_data_X = np.array([raw_plan['axc_speed_%s'%args.axis_num], raw_plan['axc_torque_ffw_gravity_%s'%args.axis_num], raw_plan['Temp'], raw_plan['axc_pos_%s'%args.axis_num]])  
    raw_data_X = np.insert(raw_data_X, 0, 1, axis=0)
    raw_data_Y = raw_plan['need_to_compensate'].values

    #Normalize data:
    normer = data_stuff.normalizer(raw_data_X)
    normed_data_X, normed_data_Y = normer.normalize_XY(raw_data_X, raw_data_Y)

    #Get model:
    params_part1 = get_part1_model_params(model)
    params_part2 = get_part2_model_params(model)

    #Forward to get output:
    input = normed_data_X
    if model == 'linear':
        c0, c1, c2, c3, c4 = params_part1[0], params_part1[1], params_part1[2], params_part1[3], params_part1[4]
        output_part1 = classical_model.linear_friction_model(input[:,part1_index], c0, c1, c2, c3, c4)
        c0, c1, c2, c3, c4 = params_part2[0], params_part2[1], params_part2[2], params_part2[3], params_part2[4]
        output_part2 = classical_model.linear_friction_model(input[:,part2_index], c0, c1, c2, c3, c4)
    elif model == 'nonlinear':
        c0, v_brk, F_brk, F_C, c1, c2, c3, c4 = params_part1[0], params_part1[1], params_part1[2], params_part1[3], params_part1[4], params_part1[5], params_part1[6], params_part1[7]
        output_part1 = classical_model.nonlinear_friction_model(input[:,part1_index], c0, v_brk, F_brk, F_C, c1, c2, c3, c4)
        c0, v_brk, F_brk, F_C, c1, c2, c3, c4 = params_part2[0], params_part2[1], params_part2[2], params_part2[3], params_part2[4], params_part2[5], params_part2[6], params_part2[7]
        output_part2 = classical_model.nonlinear_friction_model(input[:,part2_index], c0, v_brk, F_brk, F_C, c1, c2, c3, c4)
    else:
        print("Either linear or nonlinear should be given in args")
    reference = normer.denormalize_Y(normed_data_Y)
    compensate_part1 = normer.denormalize_Y(output_part1)
    compensate_part2 = normer.denormalize_Y(output_part2)

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
    print("Deploy time Classical model...")

    #Configs:
    parser = argparse.ArgumentParser(description='Friction.')
    parser.add_argument('--max_force', type=int, default = 1)
    parser.add_argument('--axis_num', type=int, default = 4)
    parser.add_argument('--mode', type=str, default = 'acc_uniform')
    parser.add_argument('--data_path', type=str, default = "../data/planning.csv")
    parser.add_argument('--VISUALIZATION', "-V", action='store_true', default=False)
    args = parser.parse_args()
    args.axis_num = args.axis_num - 1    #joint index

    #Get plans:
    mode = ['deploy', args.mode]
    raw_plan, part1_index, part2_index = data_stuff.get_data(args, mode)

    #-------------------Linear-----------------
    #Get result:
    compensate, reference = get_compensate_force(raw_plan, part1_index, part2_index, model='linear')

    #GPU render:
    try:
        import plot_utils
        from vispy import app
        plot_utils.GPU_2d_scatter_plot(reference)
        app.run()
    except Exception as e:
        print(e)

    #Visualize:
    if args.VISUALIZATION:
        plt.plot(reference, label='reference_feedback', alpha=0.5)
        plt.scatter(part1_index, compensate[part1_index], label='part1_predicted_feedback', color='green', s=1)
        plt.scatter(part2_index, compensate[part2_index], label='part2_predicted_feedback', color='red', s=1)
        plt.legend()
        plt.show()

    np.savetxt("../output/linear_compensation.txt", compensate)
    print("Done")

    #-------------------Nonlinear--------------
    compensate, reference = get_compensate_force(raw_plan, part1_index, part2_index, model='nonlinear')
    #Visualize:
    if args.VISUALIZATION:
        plt.plot(reference, label='reference_feedback', alpha=0.5)
        plt.scatter(part1_index, compensate[part1_index], label='part1_predicted_feedback', color='green', s=1)
        plt.scatter(part2_index, compensate[part2_index], label='part2_predicted_feedback', color='red', s=1)
        plt.legend()
        plt.show()

    np.savetxt("../output/nonlinear_compensation.txt", compensate)
    print("Done")


