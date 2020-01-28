python -u train_queue.py --epochs 100 --queue_size 1.0
sleep 30s
python -u train_queue.py --epochs 100 --queue_size 0.7
sleep 30s
python -u train_queue.py --epochs 100 --queue_size 0.5
sleep 30s
python -u train_queue.py --epochs 100 --queue_size 0.3
sleep 30s

python -u train_two_encoders.py --epochs 100 --batchsize 64
sleep 30s
python -u train_two_encoders.py --epochs 100 --batchsize 48
sleep 30s
python -u train_two_encoders.py --epochs 100 --batchsize 32
sleep 30s
python -u train_two_encoders.py --epochs 100 --batchsize 16
sleep 30s