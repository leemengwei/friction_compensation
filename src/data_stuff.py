import glob
import sav_smooth
import pandas as pd
import time
from IPython import embed
import numpy as np
import os,sys,time

def get_useful_data():
    files = glob.glob("../data/*.prb-log")
    datas = pd.DataFrame()
    quick_path = "../data/tmp_del/all_data.csv"
    if os.path.exists(quick_path):
        print("Read from quick data: %s"%quick_path)
        datas = pd.read_csv("%s"%quick_path, index_col=0)
        return datas
    for _file_ in files:
        data = pd.read_csv(_file_, sep=' ')
        data = data.drop(data.shape[0]-1).astype(float)
        data = data.drop('line', axis=1)
        data = data_retrieve(data)
        times = data['ms']
        friction_value_axis_3 = data['servo_feedback_torque_3']-data['axc_torque_ffw_gravity_3']    #TODO: this is friction on axis 4
        data['need_to_compensate'] = friction_value_axis_3
        data['Temp'] = 0
        datas = pd.concat((datas, data))
        print(_file_, len(datas))
    datas.to_csv("../data/tmp_del/all_data.csv")
    print("Dumped to quick data: %s"%quick_path)
    return datas


class normalizer(object):
    def __init__(self, X, Y):
        self._X_mean_never_touch_5 =  np.tile(np.array([ 1.00000000e+00, -1.22205200e+00, -3.62142402e-03,  0.00000000e+00, 4.36526466e+00]), (X.shape[1], 1)).T
        self._X_std_never_touch_5 = np.tile(np.array([1.00000000e-07, 1.03640789e+02, 4.98661209e-02, 1.00000000e-07, 1.31607409e+02]), (X.shape[1], 1)).T

        self._X_mean_never_touch_25 = np.tile(np.array([2.51065213e-04,  2.26194300e+02, -3.87426864e-03, -3.87426074e-03, 2.22841566e-04,  3.23421159e+01, -3.59547080e-01, -3.30278722e-03, -6.44383032e-04,  8.86972255e+01,  8.88570940e-02,  2.07215009e-03, -5.10639766e-02, -1.07646179e+00, -5.16060366e-03, -1.96551706e-05, -3.95824362e-04, -4.44822192e-01,  1.01560208e-03, -7.17683819e-05, 1.0,  8.42130893e-06,  1.0,  1.0, 1.0]), (X.shape[1], 1)).T
        self._X_std_never_touch_25 = np.tile(np.array([1.49259451e-02, 3.22330006e-03, 1.96332955e-02, 1.96332663e-02, 8.86414915e-03, 1.26311744e+02, 1.40872237e+00, 1.98642367e-02, 2.82137393e-02, 1.65190416e+02, 3.03534892e-01, 1.14498188e-02, 2.04909340e+02, 1.20522046e+02, 5.97290116e-02, 4.98756114e-02, 1.37543738e-01, 1.08568109e+02, 1.89032465e-02, 6.36430634e-04, 1.0, 1.48271675e-04, 1.0, 1.0, 1.0]), (X.shape[1],1)).T
        self._Y_mean_never_touch = -0.0014823869995695225
        self._Y_std_never_touch = 0.10306605601654062
    def normalize(self, X, Y):
        if len(X)==5:
            return (X-self._X_mean_never_touch_5)/self._X_std_never_touch_5, (Y-self._Y_mean_never_touch)/self._Y_std_never_touch
        else:
            return (X-self._X_mean_never_touch_25)/self._X_std_never_touch_25, (Y-self._Y_mean_never_touch)/self._Y_std_never_touch
    def denormlize(self, Y):
        return Y*self._Y_std_never_touch+self._Y_mean_never_touch


def data_retrieve(data):
    def _data_drop_acc(data):
        smoothed_v = sav_smooth.savitzky_golay(data['servo_feedback_speed_3'].values, 101, 1)
        deltas = smoothed_v[1:] - smoothed_v[:-1]
        _threshold = 0.1  #threshold for v average
        #plt.plot(data['ms'], data['servo_feedback_speed_3'])
        #plt.scatter(data.iloc[list(np.where(np.abs(deltas)>_threshold)[0])].ms, data.iloc[list(np.where(np.abs(deltas)>_threshold)[0])].servo_feedback_speed_3, 3, 'r')
        data = data.drop(list(np.where(np.abs(deltas)>_threshold)[0]))
        #plt.plot(data['ms'], data['servo_feedback_speed_3'])
        #plt.scatter(data.ms, data.servo_feedback_speed_3, 3, 'r')
        return data
    def _data_stay(data):
        compensator = 0.9   #to make sure all data at fastest routine (yet turbulent) are kept.
        chance_to_stay = np.abs(data['servo_feedback_speed_3'])/(np.abs(data['servo_feedback_speed_3']).max()*compensator)
        which_to_stay = (np.random.random(len(chance_to_stay)) < chance_to_stay)
        data = data[which_to_stay]
        return data
    #Get rid of accelerating data
    data_dropped = _data_drop_acc(data)
    #And re-scale by position (achived by chance)
    data_stayed = _data_stay(data_dropped)
    return data_stayed


