#!/bin/bash
for seed in 11 12 13 14 15
do
    python examples/ackley/rebuttal/01_bax_es.py --seed $seed
done
