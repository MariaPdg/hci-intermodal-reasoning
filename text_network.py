from transformers.modeling_distilbert import DistilBertForSequenceClassification
from transformers import DistilBertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import utils
import torch

if __name__ == "__main__":
    captions = torch.load("cached_data/val_cap")
    train_data = TensorDataset(captions)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=8, num_workers=2)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to("cuda") for t in batch)
        output = model(batch)
        print(output)

