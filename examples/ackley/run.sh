#!/bin/bash
for seed in 11 12 13 14 15
do
    python examples/ackley/00_true_es.py --seed $seed
    #python examples/ackley/01_bax_es.py --seed $seed
    #python examples/ackley/02_rs.py --seed $seed
    #python examples/ackley/03_mes.py --seed $seed
done
