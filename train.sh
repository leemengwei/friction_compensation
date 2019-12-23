#/bin/bash
set -o errexit


work_dir='/home/user/friction_compensation/'
cd $work_dir
model_deploy_dir='./'

#Training:
echo "*******************************Start traning********************************"
cd src/
python NN_train.py --max_epoch=1 --hidden_width_scaler=1 --hidden_depth=3 --axis_num=4 --mode=acc_uniform --further_mode=acc 
python NN_train.py --max_epoch=1 --hidden_width_scaler=1 --hidden_depth=3 --axis_num=4 --mode=acc_uniform -Q --further_mode=uniform 
echo "*****************************Training complete*****************************\n\n\n\n"


#Test, and meanwhile to C:
echo "*****************************Testing***************************************"
echo "Check if this plot okay?"
python NN_deploy.py  --mode='acc_uniform' --data_path="../data/planning.csv" --max_force=1 --V --axis_num=4

while true
do
    read -r -p "Use this model for compensation? [Y/n] " input

    case $input in
        [yY][eE][sS]|[yY])
            echo "Yes"
            cp ../models/NN_weights_acc_C.pt ${work_dir}
            cp ../models/NN_weights_uniform_C.pt ${work_dir}
            echo "******************************Test compelete, model copied******************************\n\n\n\n"
            exit 1
            ;; 

        [nN][oO]|[nN])
            echo "No"
            echo "******************************Model will not be used, may train more****************************\n\n\n\n"
            exit 1
            ;;

    *)
        echo "Invalid input..."
        ;;
    esac
done


cd ${work_dir}


