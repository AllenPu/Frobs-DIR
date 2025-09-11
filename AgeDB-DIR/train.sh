#for e in 10 20 21 22 23 24 25; do
#for e in 15 16 17 18 19; do
#    #for ee in 3 5 7 9 10 11 13 15; do
#    for ee in 2 4 6 8 12 14; do
#        jobs_name='regression_epoch'_${e}_'linear_epoch'_${ee}
#        echo $jobs_name
#        python train.py --resume --regression_epoch $e --linear_epoch $ee
#    done
#done
#####################
for name in './trained_models/MAE.pth' './trained_models/MAE_LDS.pth' './trained_models/MSE_LDS.pth'; do
    for e in 10 15 20 25 30; do
        for ee in 3 4 5 6 7 8 9 10; do
            python train.py --resume --model_name $name --regression_epoch $e --linear_epochh $ee
        done
    done
done