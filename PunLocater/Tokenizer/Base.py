import torch
from torch.utils.data import Dataset
from nltk.corpus import stopwords


class CustomDataset(Dataset):
    def __init__(self, tokenizer, sentences, labels, max_len, mode, config=None):
        assert mode in ["train", "valid", "test"]
        self.mode = mode
        self.len = len(sentences)
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.config = config

    def __getitem__(self, index):
        sentence = str(self.sentences[index])

        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        # 將 tokens_tensor 還原成文本
        tokens = [self.tokenizer.convert_ids_to_tokens(id) for id in ids]
        # 建立位置索引
        locations = []
        location_dict = {}
        loc = 0
        for idx, token in enumerate(tokens):

            if token in ['[CLS]', '[SEP]', '[PAD]']:
                locations.append(0)
            else:
                if '##' not in token:
                    loc += 1
                locations.append(loc)
                if location_dict.get(loc) == None:
                    location_dict[loc] = []
                location_dict[loc].append(idx)

        if self.mode == "test":
            label_tensor = None
            return {
                'token': tokens,
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'location': torch.tensor(locations, dtype=torch.long)
            }
        else:
            origin_label = self.labels[index].index(1)+1
            new_label = location_dict[origin_label][0]
            label_vec = [0]*new_label + [1]
            label_vec.extend([0]*self.max_len)
            label_tensor = torch.tensor(
                label_vec[:self.max_len], dtype=torch.long)
            return {
                'token': tokens,
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'tags': label_tensor,
                'location': torch.tensor(locations, dtype=torch.long)
            }

    def __len__(self):
        return self.len


class SynsetDataset(Dataset):
    def __init__(self, Synsets, tokenizer, sentences, labels, max_len, mode, config=None):
        assert mode in ["train", "valid", "test"]
        self.mode = mode
        self.len = len(sentences)
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.config = config
        self.Synsets = Synsets
        self.threshold = 4 # Threshold of definition count

    def __getitem__(self, index):
        sentence = str(self.sentences[index])

        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        # 將 tokens_tensor 還原成文本
        tokens = [self.tokenizer.convert_ids_to_tokens(id) for id in ids]
        # 建立位置索引
        # 取得同義詞集合數量，
        defs = []
        locations = []
        location_dict = {}
        loc = 0
        for idx, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                locations.append(0)
                defs.append(-1)
            else:
                if '##' not in token:
                    loc += 1
                    # Count definitions
                    syn_length = len(self.Synsets(token))
                    defs.append(syn_length if syn_length>self.threshold else 0)
                else:
                    defs.append(-1)             
                    
                locations.append(loc)
                if location_dict.get(loc) == None:
                    location_dict[loc] = []
                location_dict[loc].append(idx)
        defs.extend([0]*self.max_len)
        defs = torch.tensor(defs[:self.max_len], dtype=torch.float)

        if self.mode == "test":
            label_tensor = None
            return {
                'token': tokens,
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'location': torch.tensor(locations, dtype=torch.long),
                'defs': defs
            }
        else:
            origin_label = self.labels[index].index(1)+1
            new_label = location_dict[origin_label][0]
            label_vec = [0]*new_label + [1]
            label_vec.extend([0]*self.max_len)
            label_tensor = torch.tensor(
                label_vec[:self.max_len], dtype=torch.long)
            return {
                'token': tokens,
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'tags': label_tensor,
                'location': torch.tensor(locations, dtype=torch.long),
                'defs': defs,
            }

    def __len__(self):
        return self.len