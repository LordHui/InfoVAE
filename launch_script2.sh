#!/usr/bin/env bash
python elbo_mi.py --gpu=1 --train_size=$1 --alpha=0.0 --beta=1.0 &
python elbo_mi.py --gpu=1 --train_size=$1 --alpha=0.0 --beta=10.0 &
python elbo_mi.py --gpu=1 --train_size=$1 --alpha=0.0 --beta=100.0 &
python elbo_mi.py --gpu=2 --train_size=$1 --alpha=10.0 --beta=11.0 &
python elbo_mi.py --gpu=2 --train_size=$1 --alpha=10.0 --beta=20.0 &
python elbo_mi.py --gpu=3 --train_size=$1 --alpha=100.0 --beta=101.0 &
python elbo_mi.py --gpu=3 --train_size=$1 --alpha=100.0 --beta=120.0 &
python elbo_mi.py --gpu=3 --train_size=$1 --alpha=1000.0 --beta=1001.0 &