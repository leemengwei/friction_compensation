#/bin/bash

python NN_train.py --mode=acc_uniform --further_mode=all --axis_num=5 --num_of_batch=1000 --max_epoch=24 --finetune >& 5_finetune.log
python NN_train.py --mode=acc_uniform --further_mode=all --axis_num=2 --num_of_batch=1000 --max_epoch=24 --finetune >& 2_finetune.log
python NN_train.py --mode=acc_uniform --further_mode=all --axis_num=3 --num_of_batch=1000 --max_epoch=24 --finetune >& 3_finetune.log
python NN_train.py --mode=acc_uniform --further_mode=all --axis_num=4 --num_of_batch=1000 --max_epoch=24 --finetune >& 4_finetune.log
python NN_train.py --mode=acc_uniform --further_mode=all --axis_num=6 --num_of_batch=1000 --max_epoch=24 --finetune >& 6_finetune.log
python NN_train.py --mode=acc_uniform --further_mode=all --axis_num=1 --num_of_batch=1000 --max_epoch=24 --finetune >& 1_finetune.log

#for((i=2;i<=5;i++))
#do
#echo starting axis ${i}
#python NN_train.py  --mode=acc_uniform --further_mode=all --Quick_data --axis_num=${i} --num_of_batch=10000 --max_epoch=12 >& ${i}.log &
#done
#
#echo now on axis 6:
#python NN_train.py  --mode=acc_uniform --further_mode=all --Quick_data --axis_num=6 --num_of_batch=10000  --max_epoch=12 >& 6.log
#echo now on axis 1:
#python NN_train.py  --mode=acc_uniform --further_mode=all --Quick_data --axis_num=1 --num_of_batch=10000  --max_epoch=12 >& 1.log
