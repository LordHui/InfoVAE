python elbo_mi_binary.py --gpu=0 --train_size=$1 --alpha=0.0 --beta=1.0 &
python elbo_mi_binary.py --gpu=0 --train_size=$1 --alpha=0.0 --beta=5.0 &
python elbo_mi_binary.py --gpu=0 --train_size=$1 --alpha=0.0 --beta=10.0 &

python elbo_mi_binary.py --gpu=1 --train_size=$1 --alpha=0.0 --beta=1.0 &
python elbo_mi_binary.py --gpu=1 --train_size=$1 --alpha=0.0 --beta=5.0 &
python elbo_mi_binary.py --gpu=1 --train_size=$1 --alpha=0.0 --beta=10.0 &
python elbo_mi_binary.py --gpu=2 --train_size=$1 --alpha=10.0 --beta=10.0 &
python elbo_mi_binary.py --gpu=2 --train_size=$1 --alpha=10.0 --beta=11.0 &
python elbo_mi_binary.py --gpu=2 --train_size=$1 --alpha=10.0 --beta=15.0 &
python elbo_mi_binary.py --gpu=3 --train_size=$1 --alpha=50.0 --beta=50.0 &
python elbo_mi_binary.py --gpu=3 --train_size=$1 --alpha=50.0 --beta=51.0 &
python elbo_mi_binary.py --gpu=3 --train_size=$1 --alpha=50.0 --beta=60.0 &


python elbo_mi_binary.py --gpu=1 --train_size=$1 --alpha=0.0 --beta=1.0 &
python elbo_mi_binary.py --gpu=1 --train_size=$1 --alpha=0.0 --beta=5.0 &
python elbo_mi_binary.py --gpu=1 --train_size=$1 --alpha=0.0 --beta=10.0 &
python elbo_mi_binary.py --gpu=2 --train_size=$1 --alpha=10.0 --beta=10.0 &
python elbo_mi_binary.py --gpu=2 --train_size=$1 --alpha=10.0 --beta=11.0 &
python elbo_mi_binary.py --gpu=2 --train_size=$1 --alpha=10.0 --beta=15.0 &
python elbo_mi_binary.py --gpu=3 --train_size=$1 --alpha=50.0 --beta=50.0 &
python elbo_mi_binary.py --gpu=3 --train_size=$1 --alpha=50.0 --beta=51.0 &
python elbo_mi_binary.py --gpu=3 --train_size=$1 --alpha=50.0 --beta=60.0 &