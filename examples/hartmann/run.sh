#!/bin/bash
for seed in 11 12
do
    python examples/hartmann/02_rs.py --seed $seed
done
