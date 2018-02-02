#!/usr/bin/env bash
python elbo_mi.py --gpu=0 --train_size=$1 --alpha=0.0 --beta=1.0 &
python elbo_mi.py --gpu=0 --train_size=$1 --alpha=0.0 --beta=5.0 &
python elbo_mi.py --gpu=0 --train_size=$1 --alpha=0.0 --beta=10.0 &
python elbo_mi.py --gpu=0 --train_size=$1 --alpha=10.0 --beta=11.0 &
python elbo_mi.py --gpu=1 --train_size=$1 --alpha=10.0 --beta=15.0 &
python elbo_mi.py --gpu=1 --train_size=$1 --alpha=50.0 --beta=51.0 &
python elbo_mi.py --gpu=1 --train_size=$1 --alpha=50.0 --beta=55.0 &
python elbo_mi.py --gpu=1 --train_size=$1 --alpha=200.0 --beta=201.0 &