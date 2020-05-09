# NN_compose itself works with all six axis, this script is to iterate through paths.

#CUDA_VISIBLE_DEVICES=1 python NN_compose.py --data_path=../data/standard_path/realtime-20200326-171242.rec-data-rrr.prb-log --time_to_plot=35000
#CUDA_VISIBLE_DEVICES=1 python NN_compose.py --data_path=../data/standard_path/realtime-20200326-171651.rec-data-testzhixian.prb-log --time_to_plot=35000
###
####Local test, repeated
#CUDA_VISIBLE_DEVICES=1 python NN_compose.py --data_path=../data/standard_path/realtime-20200413-150531-jiepai-low.rec-data.prb-log --time_to_plot=400
#CUDA_VISIBLE_DEVICES=1 python NN_compose.py --data_path=../data/standard_path/realtime-20200413-151740-bi-low.rec-data.prb-log --time_to_plot=1500
#CUDA_VISIBLE_DEVICES=1 python NN_compose.py --data_path=../data/standard_path/realtime-20200413-152042-bi-high.rec-data.prb-log --time_to_plot=1500
#CUDA_VISIBLE_DEVICES=1 python NN_compose.py --data_path=../data/standard_path/realtime-20200413-163945-welding-high.rec-data.prb-log --time_to_plot=7500
CUDA_VISIBLE_DEVICES=1 python NN_compose.py --data_path=../data/standard_path/realtime-20200414-184219.rec-data-shangxialiao-low.prb-log --time_to_plot=10000   --finetune 
CUDA_VISIBLE_DEVICES=1 python NN_compose.py --data_path=../data/standard_path/realtime-20200414-185114.rec-data-shangxialiao-meduim.prb-log --time_to_plot=7500 --finetune
CUDA_VISIBLE_DEVICES=1 python NN_compose.py --data_path=../data/standard_path/realtime-20200414-185858.rec-data-shangxialiao-high.prb-log --time_to_plot=2500   --finetune











#Transfer
#python NN_compose.py --data_path=../data/transfer_path/transfer_realtime-20200416-82115.rec-data-welding.prb-log --time_to_plot=7500
#python NN_compose.py --data_path=../data/transfer_path/transfer_realtime-20200416-83015.rec-data-bi.prb-log --time_to_plot=1500
#python NN_compose.py --data_path=../data/transfer_path/transfer_realtime-20200416-84227.rec-data-jiepai.prb-log --time_to_plot=400
#python NN_compose.py --data_path=../data/transfer_path/transfer_realtime-20200416-85206.rec-data-shangxialiao-low.prb-log --time_to_plot=10000
#python NN_compose.py --data_path=../data/transfer_path/transfer_realtime-20200416-90021.rec-data-shangxialiao-meduim.prb-log --time_to_plot=7500
#python NN_compose.py --data_path=../data/transfer_path/transfer_realtime-20200416-90821.rec-data-shangxialiao-high.prb-log --time_to_plot=2500
#
