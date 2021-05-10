#!/bin/bash
for acq_func in "bax" "rand" "uncert"
do
    for seed in 1 2 3
    do
        python sweep_dijkstras.py --acq_func $acq_func --seed $seed --plot --can_requery --n_path 20 --n_iter 50
    done
done
