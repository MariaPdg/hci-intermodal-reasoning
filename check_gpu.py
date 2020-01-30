import torch
import termcolor
import GPUtil
import sys
import time
import multiprocessing
import train
import train_two_encoders
import train_queue
import train_with_att_maps

from knockknock import slack_sender

"""
for sending notification when your code finishes
"""
sys.stdin = open("webhook_url.txt", "r")
SLACK_WEBHOOK = sys.stdin.readline().rstrip()


@slack_sender(webhook_url=SLACK_WEBHOOK, channel="bot")
def query_gpu():
    while True:
        try:
            GPUtil.showUtilization()
            GPUtil.getFirstAvailable(order='first', maxLoad=0.2, maxMemory=0.2, attempts=1, interval=900, verbose=False)
            break
        except RuntimeError:
            time.sleep(10)
            continue
    return "gpu available"


if __name__ == "__main__":
    # query_gpu()
    # train_with_att_maps.main(att_prob_override=0.0)
    query_gpu()
    train_with_att_maps.main(att_prob_override=0.3)
    query_gpu()
    train_with_att_maps.main(att_prob_override=0.5)
    query_gpu()
    train_with_att_maps.main(att_prob_override=0.7)

    query_gpu()
    train_queue.main(queue_size_override=1.0)
    query_gpu()
    train_queue.main(queue_size_override=0.7)
    query_gpu()
    train_queue.main(queue_size_override=0.5)
    query_gpu()
    train_queue.main(queue_size_override=0.3)

    query_gpu()
    train_two_encoders.main(batch_size_override=64)
    query_gpu()
    train_two_encoders.main(batch_size_override=48)
    query_gpu()
    train_two_encoders.main(batch_size_override=32)
    query_gpu()
    train_two_encoders.main(batch_size_override=16)
