from transformers.modeling_distilbert import DistilBertForSequenceClassification
from transformers import DistilBertTokenizer, DistilBertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

import types
import utils
import torch
import torch.nn as nn


def forward_layer(self, input_ids, attention_mask=None, head_mask=None):
    """
    used to forward only first components
    :param self:
    :param input_ids:
    :param attention_mask:
    :param head_mask:
    :return:
    """
    distilbert_output = self.distilbert(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        head_mask=head_mask)
    hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
    hidden_state = hidden_state.permute(1, 0, 2)
    pooled_output, hidden_cell = self.lstm(hidden_state)
    return hidden_cell[0].squeeze()


class TextNet:
    def __init__(self, dev="cpu"):
        config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        config.output_hidden_states = True
        self.no_dropout = False
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", config=config)
        self.model.forward_layer = types.MethodType(forward_layer, self.model)
        self.model.lstm = nn.LSTM(768, 768)

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
    text_net.forward(val_cap[[0, 0]], val_mask[[0, 0]])
    print()


        


