import torch
from transformers import BertForTokenClassification, BertModel
from torch.nn import  CrossEntropyLoss
from collections import namedtuple

BertOutput = namedtuple("BertOutput",["loss","logits"])

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = BertForTokenClassification.from_pretrained(
            'bert-base-cased', num_labels=2)

    def forward(self, ids, mask, labels=None):
        output_1 = self.l1(ids, mask, labels=labels)
        return output_1


class DualAttentiveBert(torch.nn.Module):
    def __init__(self, max_len: int, hidden_size: int = 256) -> None:
        super(DualAttentiveBert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.layer1 = torch.nn.Linear(768, hidden_size)
        self.encode = torch.nn.Sequential(
            torch.nn.Linear(max_len, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
        )
        self.merge = torch.nn.Linear(hidden_size, 2)

    def forward(self, ids, mask, defs, labels=None):
        last_hidden = self.bert(ids, mask)[0]
        src1 = self.layer1(last_hidden)
        src2 = self.encode(defs)
        # Element-wise multiplication
        multi = src1 + src2.unsqueeze(1)
        logits = self.merge(multi)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return BertOutput(loss=loss, logits=logits)
