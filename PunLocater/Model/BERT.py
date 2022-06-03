import torch
from transformers import BertForTokenClassification, BertForSequenceClassification, BertModel


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
    def __init__(self,pretrained_name:str):
        super(BERTClass, self).__init__()
        self.l1 = BertForTokenClassification.from_pretrained(
            pretrained_name, num_labels=2)

    def forward(self, ids, mask, labels=None):
        output_1 = self.l1(ids, mask, labels=labels)
        return output_1


class DualAttentiveBert(torch.nn.Module):
    def __init__(self) -> None:
        super(DualAttentiveBert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')