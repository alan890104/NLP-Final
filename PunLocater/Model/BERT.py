import torch
from transformers import BertForTokenClassification, BertModel
import  torch.nn.functional  as F
from collections import namedtuple

BertOutput = namedtuple("BertOutput",["loss","logits"])

class BERTTokenClassification(torch.nn.Module):
    def __init__(self, pretrained_name: str, num_labels: int, **kwargs) -> None:
        super(BERTTokenClassification, self).__init__()
        self.bert = BertForTokenClassification.from_pretrained(
            pretrained_name,
            num_labels=num_labels,
        )

    def forward(self, ids, mask, labels):
        out1 = self.bert(
            ids,
            mask,
            labels=labels)
        return out1


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
        print(ids.size(),mask.size(),defs.size())
        last_hidden = self.bert(ids, mask)
        src1 = self.layer1(last_hidden)
        src2 = self.encode(defs)
        # Element-wise multiplication
        multi = src1 * src2
        logits = self.merge(multi)
        loss = F.cross_entropy(logits, labels)
        return BertOutput(loss=loss, logits=logits)
