# NN_test has a function of generating C models, Thus this script to get C models for each axis.
python NN_test.py --mode=acc_uniform_all --axis_num=1 -V --data_path=../data/data-j1/realtime-20200307-25609.rec-data-j1.prb-log
python NN_test.py --mode=acc_uniform_all --axis_num=2 -V --data_path=../data/data-j2/realtime-20200304-31127.rec-data-j2.prb-log
python NN_test.py --mode=acc_uniform_all --axis_num=3 -V --data_path=../data/data-j3/realtime-20200303-95431.rec-data-j3.prb-log
python NN_test.py --mode=acc_uniform_all --axis_num=4 -V --data_path=../data/data-j4/realtime-20200115-00731.rec-data-j4.prb-log
python NN_test.py --mode=acc_uniform_all --axis_num=5 -V --data_path=../data/data-j5/realtime-20200302-175748.rec-data-j5.prb-log
python NN_test.py --mode=acc_uniform_all --axis_num=5 -V --data_path=../data/standard_path/realtime-20200414-184219.rec-data-shangxialiao-low.prb-log
python NN_test.py --mode=acc_uniform_all --axis_num=6 -V --data_path=../data/data-j6/realtime-20200307-184847.rec-data-j6.prb-log
