#!/bin/bash
for seed in 11 12 13 14 15
do
    #python examples/hartmann/02_rs.py --seed $seed
    python examples/hartmann/03_mes.py --seed $seed
done