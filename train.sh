python -u train_queue.py --epochs 150 --batchsize 32 --cache 1
sleep 30s
python -u train_two_encoders.py --epochs 150 --batchsize 64 --cache 1
sleep 30s
