import glob
import pandas as pd
import time
from IPython import embed
import numpy as np
import os,sys,time
import matplotlib.pyplot as plt
from math import factorial
from tqdm import tqdm 
import pickle
np.set_printoptions(suppress=True)

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
    centers_around = [] 
    split_of_validation = 8 
    for i in range(1, split_of_validation+1): 
        centers_around.append(i/(split_of_validation+1))
    val_index = []
    for center_around in centers_around:
        val_index += list(range(int(len(Y)*(center_around-args.test_ratio/(2*split_of_validation))), int(len(Y)*(center_around+args.test_ratio/(2*split_of_validation)))))
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

def get_data(args, mode):
    #deploy time will retrieve more data ignore of banlance
    def get_index_not_acc(data):
        local_axis_num = args.axis_num - 1
        smoothed_v = savitzky_golay(data['axc_speed_%s'%local_axis_num].values, 11, 1)
        deltas = smoothed_v[1:] - smoothed_v[:-1]
        _threshold = 0.01   #TODO:vary with above sav param, perhaps auto
        where_acc = list(np.where(np.abs(deltas)>_threshold)[0]+1)
        where_uniform = list(set(range(len(data)))-set(where_acc))
        #过零点检测，并舍去（舍多了则更多的被视为加速状态, 舍少了则有一些加速点会被视为匀速，影响都不很大）
        for i in range(5):
            cross_zero_index = np.where((np.array(where_uniform)[1:]-np.array(where_uniform)[:-1])!=1)[0]+1
            where_uniform = list(np.delete(np.array(where_uniform), cross_zero_index))
        #data = data.drop(where_to_drop)
        #plt.plot(data['axc_speed_%s'%local_axis_num])
        #plt.scatter(where_uniform, data['axc_speed_%s'%local_axis_num][where_uniform])
        #plt.show()
        return where_uniform
    def get_low_high_index(data):
        local_axis_num = args.axis_num - 1
        speed_threshold = 100/60*2*np.pi   #100rpm 算大小速度的分界线
        low_speed_index = np.where(np.abs(data['axc_speed_%s'%local_axis_num])<speed_threshold)[0]
        high_speed_index = np.where(np.abs(data['axc_speed_%s'%local_axis_num])>=speed_threshold)[0]
        return list(low_speed_index), list(high_speed_index)
    def _data_balance(datas, uniform_index):
        local_axis_num = args.axis_num - 1
        #TODO: try another balance method, a backward one.
        assert "train" in mode, "data balance is called only in train mode"
        compensator = 0.9   #to make sure all data at fastest routine (yet turbulent) are kept.
        #只有train才会有balance，所以可以用servo feedback, No, back to axc's:
        chance_to_stay = np.abs(datas.iloc[uniform_index]['axc_speed_%s'%local_axis_num])/(np.abs(datas.iloc[uniform_index]['axc_speed_%s'%local_axis_num]).max()*compensator)
        which_to_stay = np.where(np.random.random(len(chance_to_stay)) < chance_to_stay)[0]
        uniform_index_balanced = list(np.array(uniform_index)[which_to_stay])
        return uniform_index_balanced

    #Get data in terms of key words [mode]:
    local_axis_num = args.axis_num - 1
    print("On axis %s, with local index %s"%(args.axis_num, local_axis_num))
    #Train time mode:
    if "train" in mode:
        if not args.finetune:
            quick_path = "../data/tmp_del/quick_%s.pkl"%args.pool_name.strip('/')
        else:
            quick_path = "../data/tmp_del/quick_finetune_%s.pkl"%args.pool_name.strip('/')
        #Quick or Normal?
        if os.path.exists(quick_path) and args.Quick_data:
            print("Reading from quick data: %s"%quick_path)
            datas = pickle.load(open(quick_path, 'rb'))
        else:
            print("No quick data: %s"%quick_path)
            files = glob.glob("../data/%s/*.prb-log"%args.pool_name)
            datas = pd.DataFrame()
            for _file_ in tqdm(files[:]):
                print("Getting data %s"%_file_)
                data = pd.read_csv(_file_, sep=' ', low_memory=False, index_col=None)
                data = data.drop(data.shape[0]-1).astype(float)
                data = data.drop('line', axis=1)
                datas = pd.concat((datas, data), ignore_index=True)
            target = datas['servo_feedback_torque_%s'%local_axis_num]-datas['axc_torque_ffw_gravity_%s'%local_axis_num]  
            datas['need_to_compensate'] = target
            datas['Temp'] = 0
            print("Dumping to quick data: %s"%quick_path)
            pickle.dump(datas, open(quick_path, 'wb'), protocol=4)
        #Mode?
        #说明： acc、uniform分法会有关于匀速阶段的数据均衡， low_high不太方便做就不做了。
        if "acc_uniform" in mode:
            uniform_index = get_index_not_acc(datas)
            uniform_index.sort()
            acc_index = list(set(list(range(len(datas)))) - set(get_index_not_acc(datas)))
            acc_index.sort()
            #a = hist(datas['need_to_compensate'],100)
            #chance = np.clip((a[0].sum()/a[0])/((a[0].sum()/a[0]).max()*0.001), 0, 1)
            uniform_index_balanced = _data_balance(datas, uniform_index)
            uniform_index_balanced.sort()
            print("Uniform motion data are balanced while acc's are not")
            return datas, acc_index, uniform_index_balanced
        elif "low_high" in mode:
            low_index, high_index = get_low_high_index(datas)
            low_index.sort()
            high_index.sort()
            return datas, low_index, high_index
        else:
            print("Neither acc_uniform or low_high in mode")
        return data_stayed

    #Deploy time mode:
    elif "deploy" in mode:
        print("Reading data", args.data_path)
        datas = pd.read_csv(args.data_path, sep=' ', low_memory=False, index_col=None)
        datas = datas.drop(datas.shape[0]-1).astype(float)
        try:
            datas = datas.drop('line', axis=1)
        except:
            pass
        try:
            target = datas['servo_feedback_torque_%s'%local_axis_num]-datas['axc_torque_ffw_gravity_%s'%local_axis_num] 
        except:
            target = 0
        datas['need_to_compensate'] = target
        datas['Temp'] = 0
        #使用两种模式返回deploy数据，供后续不同模型的predict，两种方式分别为：匀速/加速区分，低速/高速区分
        if "acc_uniform" in mode:
            #make sure model for acc is trained
            uniform_index = get_index_not_acc(datas)
            acc_index = list(set(range(len(datas))) - set(uniform_index))
            acc_index.sort()
            print("Be careful, will soon later compensate on whole data... Make sure you've solved the acc force")
            return datas, acc_index, uniform_index
        if "acc_uniform_all" in mode:
            #make sure model for acc is trained
            acc_index = list(range(len(datas)))
            uniform_index = list(range(len(datas)))
            print("Indentical part index returned")
            return datas, acc_index, uniform_index
        elif "low_high" in mode:
            #make sure model for low high is trained
            low_index, high_index = get_low_high_index(datas)
            low_index.sort()
            high_index.sort()
            print("Be careful, will soon later compensate on whole data... Make sure you've solved the acc force")
            return datas, low_index, high_index
        else:
            print("When deploy, must give acc_uniform or low_high in [mode]")
            sys.exit()
    else:
        print("Neither train nor deploy in given mode!")
        sys.exit()

class normalizer(object):
    def __init__(self, X, Y, args):
        self.axis_num = args.axis_num
        #use for classical models:
        #self._X_mean_never_touch_5 =  np.tile(np.array([ 1.00000000e+00, -1.22205200e+00, -3.62142402e-03,  0.00000000e+00, 4.36526466e+00]), (X.shape[1], 1)).T
        #self._X_std_never_touch_5 = np.tile(np.array([1.00000000e-07, 1.03640789e+02, 4.98661209e-02, 1.00000000e-07, 1.31607409e+02]), (X.shape[1], 1)).T
        #use for NN in 5
        #self._X_mean_never_touch_5 =  np.tile(np.array([2.41475122e-01, -7.08151114e+00,  4.70443217e-03,  2.18388745e-05, 1.0]), (X.shape[1], 1)).T
        #self._X_std_never_touch_5 = np.tile(np.array([2.04345352e+02, 1.75959748e+02, 6.61161496e-02, 5.75348903e-02, 1.0]), (X.shape[1], 1)).T
        #use for NN:
        #self._X_mean_never_touch_25 = np.tile(np.array([2.51065213e-04,  2.26194300e+02, -3.87426864e-03, -3.87426074e-03, 2.22841566e-04,  3.23421159e+01, -3.59547080e-01, -3.30278722e-03, -6.44383032e-04,  8.86972255e+01,  8.88570940e-02,  2.07215009e-03, -5.10639766e-02, -1.07646179e+00, -5.16060366e-03, -1.96551706e-05, -3.95824362e-04, -4.44822192e-01,  1.01560208e-03, -7.17683819e-05, 1.0,  8.42130893e-06,  1.0,  1.0, 1.0]), (X.shape[1], 1)).T
        #self._X_std_never_touch_25 = np.tile(np.array([1.49259451e-02, 3.22330006e-03, 1.96332955e-02, 1.96332663e-02, 8.86414915e-03, 1.26311744e+02, 1.40872237e+00, 1.98642367e-02, 2.82137393e-02, 1.65190416e+02, 3.03534892e-01, 1.14498188e-02, 2.04909340e+02, 1.20522046e+02, 5.97290116e-02, 4.98756114e-02, 1.37543738e-01, 1.08568109e+02, 1.89032465e-02, 6.36430634e-04, 1.0, 1.48271675e-04, 1.0, 1.0, 1.0]), (X.shape[1],1)).T
        #self._Y_mean_never_touch = -0.0014823869995695225
        #self._Y_std_never_touch = 0.10306605601654062
        #If other types for research:
        self._X_mean_never_touch_dynamic = X.mean(axis=1)
        self._X_std_never_touch_dynamic = X.std(axis=1)+1e-9
        self._Y_mean_never_touch_dynamic = np.array([Y.mean()])
        self._Y_std_never_touch_dynamic = np.array([Y.std()+1e-9])
        self.X = X
        self.Y = Y
    def generate_statistics(self):
        np.savetxt("../statistics/X_mean_%s"%self.axis_num, self._X_mean_never_touch_dynamic, fmt='%.12f')
        np.savetxt("../statistics/X_std_%s"%self.axis_num, self._X_std_never_touch_dynamic, fmt='%.12f')
        np.savetxt("../statistics/Y_mean_%s"%self.axis_num, self._Y_mean_never_touch_dynamic, fmt='%.12f')
        np.savetxt("../statistics/Y_std_%s"%self.axis_num, self._Y_std_never_touch_dynamic, fmt='%.12f')
    def generate_raw_secure(self):
        raw_secure_range = np.array([self.X.min(axis=1), self.X.max(axis=1)])
        np.savetxt("../statistics/X_secure_%s"%self.axis_num, raw_secure_range, fmt='%.12f')
    def get_raw_secure(self):
        raw_secure_range = np.loadtxt("../statistics/X_secure_%s"%self.axis_num)
        return raw_secure_range
    def get_statistics(self, X_shape):
        static_X_mean_never_touch_dynamic = np.loadtxt("../statistics/X_mean_%s"%self.axis_num)
        static_X_std_never_touch_dynamic = np.loadtxt("../statistics/X_std_%s"%self.axis_num)
        self.static_X_mean_never_touch_dynamic = np.tile(static_X_mean_never_touch_dynamic, (X_shape, 1)).T
        self.static_X_std_never_touch_dynamic = np.tile(static_X_std_never_touch_dynamic, (X_shape, 1)).T
        self.static_Y_mean_never_touch_dynamic = float(np.loadtxt("../statistics/Y_mean_%s"%self.axis_num))
        self.static_Y_std_never_touch_dynamic = float(np.loadtxt("../statistics/Y_std_%s"%self.axis_num))
    def normalize_X(self, X):
        #print("Normalized dim %s for NN model"%len(X))
        #return (X-self._X_mean_never_touch_25)/self._X_std_never_touch_25
        return (X-self.static_X_mean_never_touch_dynamic)/self.static_X_std_never_touch_dynamic
    def normalize_XY(self, X, Y):
#        if len(X)==5:
#            #print("Normalized dim %s for classical model"%len(X))
#            return (X-self._X_mean_never_touch_5)/self._X_std_never_touch_5, (Y-self._Y_mean_never_touch)/self._Y_std_never_touch
#        elif len(X)==25:
#            #print("Normalized dim %s for NN model"%len(X))
#            return (X-self._X_mean_never_touch_25)/self._X_std_never_touch_25, (Y-self._Y_mean_never_touch)/self._Y_std_never_touch
#        else:
#            #print("Normalized dim %s for NN model"%len(X))
        return (X-self.static_X_mean_never_touch_dynamic)/self.static_X_std_never_touch_dynamic, (Y-self.static_Y_mean_never_touch_dynamic)/self.static_Y_std_never_touch_dynamic
    def denormalize_Y(self, Y):
        #return Y*self._Y_std_never_touch+self._Y_mean_never_touch
        return Y*self.static_Y_std_never_touch_dynamic+self.static_Y_mean_never_touch_dynamic


