#/bin/bash
python NN_train.py --mode=acc_uniform --further_mode=all --Quick_data --axis_num=5 --num_of_batch=10000 --max_epoch=50 >& 5.log
python NN_train.py --mode=acc_uniform --further_mode=all --Quick_data --axis_num=2 --num_of_batch=10000 --max_epoch=50 >& 2.log
python NN_train.py --mode=acc_uniform --further_mode=all --Quick_data --axis_num=3 --num_of_batch=10000 --max_epoch=50 >& 3.log
python NN_train.py --mode=acc_uniform --further_mode=all --Quick_data --axis_num=4 --num_of_batch=10000 --max_epoch=50 >& 4.log
python NN_train.py --mode=acc_uniform --further_mode=all --Quick_data --axis_num=6 --num_of_batch=100 --max_epoch=5 >& 6.log
python NN_train.py --mode=acc_uniform --further_mode=all --Quick_data --axis_num=1 --num_of_batch=10000 --max_epoch=50 >& 1.log

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
