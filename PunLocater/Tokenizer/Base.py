from ctypes import Union
from typing import Any, List

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer


class CustomDataset(Dataset):
    def __init__(self, tokenizer, sentences:List[str], labels:List[str], max_len:int, mode:str):
        assert mode in ["train", "valid", "test"]
        self.mode = mode
        self.len = len(sentences)
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        sentence = str(self.sentences[index])
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        if self.mode == "test":
            label_tensor = None
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
            } 
        else:
            label = self.labels[index]
            label.extend([0]*self.max_len)
            label_tensor = torch.tensor(label[:self.max_len], dtype=torch.long)
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'labels': label_tensor
            } 
    
    def __len__(self):
        return self.len
        