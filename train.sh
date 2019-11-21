python -u train.py --epochs 1 --train_modality_net 0 --loss_function 1
sleep 30s

python -u train.py --epochs 1 --train_modality_net 1 --loss_function 1
sleep 30s

python -u train.py --epochs 1 --train_modality_net 1 --loss_function 0
sleep 30s

python -u train.py --epochs 1 --train_modality_net 0 --loss_function 0