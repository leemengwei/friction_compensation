import matplotlib.pyplot as plt
import numpy as np
from IPython import embed

epsilon = 1e-10
def evaluate_error_rate(args, outputs, targets, normer, raw_data, showup=False):
    local_axis_num = args.axis_num - 1
    outputs = normer.denormalize_Y(outputs)
    targets = normer.denormalize_Y(targets)
    value_threshold = 0.01

    _plan = raw_data['axc_torque_ffw_gravity_%s'%local_axis_num].values
    _compensate = outputs
    _meassure = raw_data['servo_feedback_torque_%s'%local_axis_num].values  #same as plan+target

    low_value_region = np.where(np.abs(_meassure)<value_threshold)
    high_value_region = np.where(np.abs(_meassure)>=value_threshold)
   
    error_rate_original = np.abs((_plan[high_value_region] - _meassure[high_value_region])/(_meassure[high_value_region]+epsilon)).mean()*100
    error_rate_low = np.abs((_plan[low_value_region] + _compensate[low_value_region] - _meassure[low_value_region])/(_meassure[low_value_region]+epsilon)).mean()*100
    error_rate_high = np.abs((_plan[high_value_region] + _compensate[high_value_region] - _meassure[high_value_region])/(_meassure[high_value_region]+epsilon)).mean()*100
    if showup:
        bins = np.linspace(targets.min(), targets.max(), 1000)
        plt.hist(_meassure[low_value_region], bins=bins, color='gray', label="Ignored")
        plt.hist(_meassure[high_value_region], bins=bins, color='black', label='Taken')
        plt.title("Val set: data taken and ignored")
        plt.legend()
        plt.show()
    if error_rate_high > 1000:
        print("Relative Error Should not be this large"*10)
        embed()
    return (error_rate_original, error_rate_low, error_rate_high)


