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
for name in 'ranksim'; do
    for e in 1 3 5 10 15 20 25 30 35 40; do
        for ee in 1 3 5 7 9 11 13 15 17 19; do
        jobs_name='model_name'_${name}_'regression_epoch'_${e}_'linear_epoch'_${ee}
        echo $jobs_name
        python train_methods.py --resume --model_name $name --regression_epoch $e --linear_epoch $ee
        done
    done
done