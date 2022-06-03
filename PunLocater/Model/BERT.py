import torch
from transformers import BertForTokenClassification, BertForSequenceClassification, BertModel
from transformers import RobertaForTokenClassification, RobertaTokenizer


class BERTClass(torch.nn.Module):
    def __init__(self, num_labels: int = 2):
        super(BERTClass, self).__init__()
        self.l1 = BertForTokenClassification.from_pretrained(
            'bert-base-cased', num_labels=num_labels)

    def forward(self, ids, mask, labels=None):
        output_1 = self.l1(ids, mask, labels=labels)
        return output_1


class RobertaClass(torch.nn.Module):
    def __init__(self, num_labels: int = 2):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaForTokenClassification.from_pretrained(
            'roberta-base', num_labels=num_labels)

    def forward(self, ids, mask, labels=None):
        output_1 = self.l1(ids, mask, labels=labels)
        return output_1


class DualAttentiveBert(torch.nn.Module):
    def __init__(self, num_labels: int = 2) -> None:
        super(DualAttentiveBert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
