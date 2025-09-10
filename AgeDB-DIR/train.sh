for e in 10 20 21 22 23 24 25; do
    for ee in 3 5 7 9 10 11 13 15; do
        jobs_name='regression_epoch'_${e}_'linear_epoch'_${ee}
        echo $jobs_name
        python train.py --resume --regression_epoch $e --linear_epoch $ee
    done
done