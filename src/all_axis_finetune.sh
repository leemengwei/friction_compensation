#/bin/bash
for((i=2;i<=5;i++))
do
echo starting axis ${i}
python NN_train.py  --mode=acc_uniform --further_mode=all --Quick_data --axis_num=${i} --num_of_batch=10000 --max_epoch=3 --finetune >& ${i}_finetune.log &
done

