from transformers.modeling_distilbert import DistilBertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import types
import utils
import torch
import torch.nn as nn


def forward_layer(self, input_ids,  attention_mask=None, head_mask=None):
    distilbert_output = self.distilbert(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        head_mask=head_mask)
    hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
    pooled_output = hidden_state[:, 0]  # (bs, dim)
    pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
    pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
    pooled_output = self.dropout(pooled_output)  # (bs, dim)
    return pooled_output


class TextNet:
    def __init__(self, dev="cpu"):
        config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        config.output_hidden_states = True

        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", config=config)
        self.model.forward_layer = types.MethodType(forward_layer, self.model)
        self.model.to(dev)

        self.model.eval()

    def forward(self, indices, masks):
        _, hidden_vec = self.model(indices, masks)
        hidden_vec2 = self.model.forward_layer(indices, masks)
        print(hidden_vec[-1].size(), hidden_vec2.size())
        

if __name__ == "__main__":
    device = "cpu"
    net = TextNet(device)
    captions = torch.load("cached_data/val_cap")
    masks = torch.load("cached_data/val_mask")
    train_data = TensorDataset(captions, masks)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=8, num_workers=2)

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        net.forward(batch[0], batch[1])

