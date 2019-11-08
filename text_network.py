from transformers.modeling_distilbert import DistilBertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import utils
import torch


class TextNet:
    def __init__(self):
        config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        config.output_hidden_states = True

        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", config=config)

    def forward(self, indices, masks):
        _, hidden_vec = self.model(indices, masks)
        

if __name__ == "__main__":
    device = "cpu"

    captions = torch.load("cached_data/val_cap")
    train_data = TensorDataset(captions)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=8, num_workers=2)
    
    config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
    config.output_hidden_states = True

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", config=config)
    model.to(device)
    model.eval()
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        logits, hidden = model(batch[0])
        print(hidden[0].size())

