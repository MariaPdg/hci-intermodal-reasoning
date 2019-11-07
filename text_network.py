from transformers.modeling_distilbert import DistilBertForSequenceClassification
from transformers import DistilBertTokenizer

import utils
import torch

if __name__ == "__main__":
    captions, _ = utils.read_caption()
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    for counter, cap in enumerate(captions.values()):
        assert len(cap) == 1
        print(cap[0], tokenizer.encode(cap[0]), model(torch.tensor(tokenizer.encode(cap[0]))))
        if counter > 10:
            break

