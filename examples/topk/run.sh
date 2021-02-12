#!/bin/bash
for seed in 1 2 3 4 5
do
    for acq_str in "eig1" "eig2" "eig3" "rand" "uncert"
    do
        python examples/topk/02_viz.py --acq_str $acq_str --seed $seed --n_init 10 --n_iter 100
    done
done
