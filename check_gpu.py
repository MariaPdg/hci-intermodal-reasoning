import torch
import termcolor
import GPUtil
import sys
import time
import train

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
            GPUtil.getFirstAvailable(order='first', maxLoad=0.1, maxMemory=0.1, attempts=1, interval=900, verbose=False)
            break
        except RuntimeError:
            time.sleep(10)
            continue
    return "gpu available"


if __name__ == "__main__":
    query_gpu()
    train.main()
    query_gpu()
    train.main(idloss_override=0)
