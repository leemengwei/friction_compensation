import numpy as np
import os,sys,time
import glob
from IPython import embed


win_data_path = "/run/user/1000/gvfs/smb-share:server=192.168.89.125,share=d$/linux_data_share/"
linux_data_path = "../data"

if __name__ == "__main__":
    print("Data transmit begin...")
    while True:
        time.sleep(1)
        if not os.path.exists(win_data_path):
            print("Check Network connection, can't find Windows data path! Retrying in 3 second")
            time.sleep(3)
            continue
        files = glob.glob("%s"%win_data_path+"/*.prb-log")
        for this_file in files:
            local_data_pool = glob.glob(linux_data_path+"/*.prb-log")
            print("mv %s %s, %s in local data pool."%(this_file, linux_data_path, len(local_data_pool)))
            os.system("mv %s %s"%(this_file, linux_data_path))
        print("Waiting...")
        time.sleep(5)
