import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# from bertviz.pytorch_pretrained_bert import BertModel
from fastNLP.modules.encoder import LSTM
from fastNLP.modules.decoder import MLP
from fastNLP.io.model_io import ModelSaver

bert_base_dir = '/remote-home/ygxu/workspace/BERT/BERT_English_uncased_L-12_H-768_A_12'
bert_large_dir = '/remote-home/ygxu/workspace/BERT/BERT_English_uncased_L-24_H-1024_A_16'


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def func(x):
    return x


class TextClassificationModel(nn.Module):

    def __init__(
            self,
            n_labels=2,
            dropout=0.0,
            fine_tune_bert=False,
            max_len=100,
            bert_large=False,
            use_cls=True,
            layer=0,
            use_lstm=True,
    ):
        super(TextClassificationModel, self).__init__()
        self.bert_encoder = BertModel.from_pretrained(bert_large_dir if bert_large else bert_base_dir)
        print(f"type of bert encoder = {type(self.bert_encoder)}")
        self.bert_encoder.eval()
        self.n_labels = n_labels
        self.dropout_p = dropout
        self.layer = layer
        self.dropout = nn.Dropout(self.dropout_p)
        self.use_lstm = use_lstm
        if not self.use_lstm:
            self.mlp = MLP([(1024 if bert_large else 768) * (1 if self.layer != 0 else (24 if bert_large else 12)), self.n_labels], func, dropout=self.dropout_p)
        else:
            self.mlp = MLP([(1024 if bert_large else 768) * 2, self.n_labels], func, dropout=self.dropout_p)
        self.max_len = max_len
        self.use_cls = use_cls
        self.lstm = LSTM((1024 if bert_large else 768) * (1 if self.layer != 0 else (24 if bert_large else 12)), (1024 if bert_large else 768), bidirectional=True, dropout=dropout)
        if not fine_tune_bert:
            self._clear_bert_encoder_grad()

    def _clear_bert_encoder_grad(self):
        for p in self.bert_encoder.parameters():
            p.requires_grad = False

    def forward(self, tokens, masks):
        """
        :param tokens: [batch, seq_len]
        :param masks: [batch, seq_len]
        :return:
        """
        if tokens.size(1) > self.max_len:
            tokens = tokens[:, : self.max_len]
            masks = masks[:, : self.max_len]
        mask = masks.view(masks.size(0), masks.size(1), -1)
        # mask: [batch, seq_len, 1]
        attention_mask = masks

        tokens_device = tokens.device
        if tokens.device != self.parameters().__next__().device:
            tokens = tokens.to(self.parameters().__next__().device)
            mask = mask.to(self.parameters().__next__().device)
            attention_mask = attention_mask.to(self.parameters().__next__().device)
        x, _, __ = self.bert_encoder(tokens, attention_mask=attention_mask, output_all_encoded_layers=True)
        # x: list of [batch, seq_len, H]
        # x = torch.cat(x, dim=-1)
        # x = torch.sum(x.view(x.size(0), x.size(1), -1, 12), dim=-1)
        if self.layer == 0:
            hidden = torch.cat(x, dim=-1)
        else:
            hidden = x[self.layer - 1]
        # hidden: [batch, seq_len, H * 12] if all layer else [batch, seq_len, H]
        if self.use_lstm:
            hidden = self.lstm(hidden)
        # hidden: [batch, seq_len, H * 24] if all layer else [batch, seq_len, H * 2]

        if self.use_cls:
            encode = hidden[:, 0, :]
        else:
            encode = torch.sum(hidden * mask, dim=1) / torch.sum(mask, dim=1)
        # encode: [batch, 1, H * 12] <- get the [CLS] token for next layer.
        prediction = self.mlp(self.dropout(encode))
        return {'predict': prediction.to(tokens_device)}

    def save_bert_model(self, path):
        # torch.save(self.bert_encoder.state_dict(), path)
        torch.save(self.bert_encoder, path)



