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
for name in 'MSE_LDS'; do
    for e in 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40; do
        for ee in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
        jobs_name='model_name'_${name}_'regression_epoch'_${e}_'linear_epoch'_${ee}
        echo $jobs_name
        python train.py --resume --model_name $name --regression_epoch $e --linear_epoch $ee
        done
    done
done