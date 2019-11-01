import numpy as np
import numpy as np
import data_stuff
from IPython import embed
import classical_model

def get_model_params():
    params = np.loadtxt("classical_weights")
    return params

if __name__ == "__main__":
    print("Deploy time Classical model...")
    #Configs:
    max_force = 1
    axis_num = 4
    axis_num = axis_num - 1
    data_path = "../data/planning.csv"

    #Get data:
    raw_data = data_stuff.get_planning_data(data_path, axis_num)
    raw_data_X = np.array([raw_data['axc_speed_%s'%axis_num], raw_data['axc_torque_ffw_gravity_%s'%axis_num], raw_data['Temp'], raw_data['axc_pos_%s'%axis_num]])   #TODO: this is axis 4
    raw_data_X = np.insert(raw_data_X, 0, 1, axis=0)
    raw_data_Y = raw_data['need_to_compensate'].values

    #Normalize data:
    normer = data_stuff.normalizer(raw_data_X)
    normed_data_X, normed_data_Y = normer.normalize_XY(raw_data_X, raw_data_Y)

    #Get model:
    params = get_model_params()

    #Forward to get output:
    c0, c1, c2, c3, c4 = params[0], params[1], params[2], params[3], params[4]
    input = normed_data_X
    output = classical_model.linear_friction_model(input, c0, c1, c2, c3, c4)

    #Denormalize Safety restrictions:
    compensate = normer.denormalize_Y(output)
    compensate = np.clip(compensate, -max_force, max_force)

    print("Done")
    embed()

