import torch
from transformers import BertForTokenClassification, BertForSequenceClassification


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


class BERTSequenceClassification(torch.nn.Module):
    def __init__(self, pretrained_name: str, num_labels: int, **kwargs) -> None:
        super(BERTSequenceClassification, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            pretrained_name,
            num_labels=num_labels,
        )
    def forward(self, ids,mask,labels):
        out1 = self.bert(
            ids,
            mask,
            labels=labels
        )
        return out1
