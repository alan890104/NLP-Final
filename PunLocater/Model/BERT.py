import torch
from transformers import BertForTokenClassification


class BERTClass(torch.nn.Module):
    def __init__(self,pretrained_name:str):
        super(BERTClass, self).__init__()
        self.l1 = BertForTokenClassification.from_pretrained(
            pretrained_name, num_labels=2)

    def forward(self, ids, mask, labels=None):
        output_1 = self.l1(ids, mask, labels=labels)
        return output_1
