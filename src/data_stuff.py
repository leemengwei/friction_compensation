import glob
import pandas as pd
import time
from IPython import embed
import numpy as np
import os,sys,time
import matplotlib.pyplot as plt
from math import factorial

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def split_dataset(args, X, Y, raw_data):
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

def get_useful_data(args):
    files = glob.glob("../data/*.prb-log")
    datas = pd.DataFrame()
    quick_path = "../data/tmp_del/useful_data.csv"
    if os.path.exists(quick_path) and args.Quick_data:
        print("Read from quick data: %s"%quick_path)
        datas = pd.read_csv("%s"%quick_path, index_col=0)
        return datas
    for _file_ in files:
        data = pd.read_csv(_file_, sep=' ', low_memory=False)
        data = data.drop(data.shape[0]-1).astype(float)
        data = data.drop('line', axis=1)
        data, stay_index = data_retrieve(data)
        times = data['ms']
        target = data['servo_feedback_torque_%s'%args.axis_num]-data['axc_torque_ffw_gravity_%s'%args.axis_num]    #TODO: this is friction on axis 4
        data['need_to_compensate'] = target
        data['Temp'] = 0
        datas = pd.concat((datas, data))
        print(_file_, len(datas))
    datas.to_csv(quick_path)
    print("Dumped to quick data: %s"%quick_path)
    return datas

def get_planning_data_all(data_path, axis_num, run_on_whole_data=False):
    data = pd.read_csv(data_path, sep=' ', low_memory=False)
    data = data.drop(data.shape[0]-1).astype(float)
    data = data.drop('line', axis=1)
    times = data['ms']
    try:
        target = data['servo_feedback_torque_%s'%axis_num]-data['axc_torque_ffw_gravity_%s'%axis_num]    #TODO: this is friction on axis 4
    except:
        print("No reference info")
        target = np.tile(0, len(data))
    data['need_to_compensate'] = target
    data['Temp'] = 0
    if run_on_whole_data:    #more danger
        data_returned, stay_index = data, None
    else:                    #more safe
        data_returned, stay_index = data_retrieve(data, deploy_time=True)
    print(data_path, len(data_returned), "stayed")
    return data, data_returned, stay_index

class normalizer(object):
    def __init__(self, X):
        self._X_mean_never_touch_5 =  np.tile(np.array([ 1.00000000e+00, -1.22205200e+00, -3.62142402e-03,  0.00000000e+00, 4.36526466e+00]), (X.shape[1], 1)).T
        self._X_std_never_touch_5 = np.tile(np.array([1.00000000e-07, 1.03640789e+02, 4.98661209e-02, 1.00000000e-07, 1.31607409e+02]), (X.shape[1], 1)).T

        self._X_mean_never_touch_25 = np.tile(np.array([2.51065213e-04,  2.26194300e+02, -3.87426864e-03, -3.87426074e-03, 2.22841566e-04,  3.23421159e+01, -3.59547080e-01, -3.30278722e-03, -6.44383032e-04,  8.86972255e+01,  8.88570940e-02,  2.07215009e-03, -5.10639766e-02, -1.07646179e+00, -5.16060366e-03, -1.96551706e-05, -3.95824362e-04, -4.44822192e-01,  1.01560208e-03, -7.17683819e-05, 1.0,  8.42130893e-06,  1.0,  1.0, 1.0]), (X.shape[1], 1)).T
        self._X_std_never_touch_25 = np.tile(np.array([1.49259451e-02, 3.22330006e-03, 1.96332955e-02, 1.96332663e-02, 8.86414915e-03, 1.26311744e+02, 1.40872237e+00, 1.98642367e-02, 2.82137393e-02, 1.65190416e+02, 3.03534892e-01, 1.14498188e-02, 2.04909340e+02, 1.20522046e+02, 5.97290116e-02, 4.98756114e-02, 1.37543738e-01, 1.08568109e+02, 1.89032465e-02, 6.36430634e-04, 1.0, 1.48271675e-04, 1.0, 1.0, 1.0]), (X.shape[1],1)).T
        self._Y_mean_never_touch = -0.0014823869995695225
        self._Y_std_never_touch = 0.10306605601654062
    def normalize_XY(self, X, Y):
        if len(X)==5:
            print("Normalized dim %s for classical model"%len(X))
            return (X-self._X_mean_never_touch_5)/self._X_std_never_touch_5, (Y-self._Y_mean_never_touch)/self._Y_std_never_touch
        else:
            print("Normalized dim %s for NN model"%len(X))
            return (X-self._X_mean_never_touch_25)/self._X_std_never_touch_25, (Y-self._Y_mean_never_touch)/self._Y_std_never_touch
    def denormalize_Y(self, Y):
        return Y*self._Y_std_never_touch+self._Y_mean_never_touch

def data_retrieve(data, deploy_time=False):
    #deploy time will retrieve more data ignore of banlance
    def _data_drop_acc(data):
        smoothed_v = savitzky_golay(data['servo_feedback_speed_3'].values, 101, 1)
        deltas = smoothed_v[1:] - smoothed_v[:-1]
        _threshold = 0.1  #threshold for v average
        where_to_drop = list(np.where(np.abs(deltas)>_threshold)[0]+1)
        where_to_stay = list(set(range(len(data)))-set(where_to_drop))
        data = data.drop(where_to_drop)
        return data, where_to_stay
    def _data_stay(data):
        compensator = 0.9   #to make sure all data at fastest routine (yet turbulent) are kept.
        chance_to_stay = np.abs(data['servo_feedback_speed_3'])/(np.abs(data['servo_feedback_speed_3']).max()*compensator)
        which_to_stay = (np.random.random(len(chance_to_stay)) < chance_to_stay)
        data = data[which_to_stay]
        return data

    #Get rid of accelerating data
    data_dropped, stay_index = _data_drop_acc(data)

    #And re-scale by position (achived by chance)
    if not deploy_time:
        data_stayed = _data_stay(data_dropped)
    else:
        #deploy time should have all position, so we're here.
        data_stayed = data_dropped
        pass
    return data_stayed, stay_index


