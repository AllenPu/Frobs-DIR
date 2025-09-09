for e in 10 20 21 22 23 24 25; do
    for ee in 3 5 7 9 10 11 13 15; do
        job = 'regression_epoch'_${e}_'linear_epoch_'_${ee}
        echo ${job}
        python train.py --resume --regression_epoch ${e} --linear_epoch ${ee}
    done
done