from transformers.modeling_distilbert import DistilBertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import types
import utils
import torch
import torch.nn as nn


def forward_layer(self, input_ids, attention_mask=None, head_mask=None):
    distilbert_output = self.distilbert(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        head_mask=head_mask)
    hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
    pooled_output = hidden_state[:, 0]  # (bs, dim)
    # pooled_output = self.linear_mat(pooled_output)
    return pooled_output


class TextNet:
    def __init__(self, dev="cpu"):
        config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        config.output_hidden_states = True
        self.no_dropout = False
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", config=config)
        self.model.forward_layer = types.MethodType(forward_layer, self.model)
        self.model.linear_mat = nn.Linear(in_features=768, out_features=10)
        self.model.to(dev)

    def forward(self, indices, masks):
        hidden_vec2 = self.model.forward_layer(indices, masks)
        return hidden_vec2

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()


if __name__ == "__main__":
    val_cap = torch.load("cached_data/val_cap")
    val_mask = torch.load("cached_data/val_mask")
    text_net = TextNet("cpu")
    print(text_net.forward(val_cap[[0, 0]], val_mask[[0, 0]])[0])
    print()
    print(text_net.forward(val_cap[[0, 0]], val_mask[[0, 0]])[0])
    print()

    text_net.model.eval()
    print(text_net.forward(val_cap[[0, 0]], val_mask[[0, 0]])[0])
    print()

    text_net.no_dropout = True
    print(text_net.forward(val_cap[[0, 0]], val_mask[[0, 0]])[0])


        


