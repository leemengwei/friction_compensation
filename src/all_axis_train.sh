#/bin/bash
python NN_train.py  --mode=acc_uniform --further_mode=all --Quick_data --learning_rate=0.05 --axis_num=2 --num_of_batch=10000 --hidden_width_scaler=5 --max_epoch=-2
python NN_train.py  --mode=acc_uniform --further_mode=all --Quick_data --learning_rate=0.05 --axis_num=3 --num_of_batch=10000 --hidden_width_scaler=5 --max_epoch=-2
python NN_train.py  --mode=acc_uniform --further_mode=all --Quick_data --learning_rate=0.05 --axis_num=4 --num_of_batch=10000 --hidden_width_scaler=5 --max_epoch=-2
python NN_train.py  --mode=acc_uniform --further_mode=all --Quick_data --learning_rate=0.05 --axis_num=5 --num_of_batch=10000 --hidden_width_scaler=5 --max_epoch=-2
python NN_train.py  --mode=acc_uniform --further_mode=all --Quick_data --learning_rate=0.05 --axis_num=6 --num_of_batch=10000 --hidden_width_scaler=5 --max_epoch=-2
python NN_train.py  --mode=acc_uniform --further_mode=all --Quick_data --learning_rate=0.05 --axis_num=1 --num_of_batch=10000 --hidden_width_scaler=5 --max_epoch=-2
