#Author: Yu-Lun Hsu
#Student ID: 0716235
#HW ID: final_project
#Due Date: 06/06/2022

# %%
# Importing the libraries needed

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from Tokenizer import SynsetDataset
from Trainer import BaseTrainer, TrainConfig
from Model import DualAttentiveBert
from Trainer.Base import TrainConfig
from Resolver import XMLResolver, testloader
from config import Config
from nltk.corpus import wordnet
import numpy as np
import random

# %% 
# Setting Random seed 
seed = 890104
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# %%
# Start Load Config file
config = Config("./config/submit/dual.yaml")

# %%
# Start resolve training set
data = XMLResolver(config.Global.training_path, config.Global.answer_path)

# %%
# Start loading tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.Global.pretrain_name)

# %%
# Start Train Test Split
train_size = int(config.Global.train_percent*len(data))
train_sent, train_label = data[:train_size]
test_sent, test_label = data[train_size:]

synsets = wordnet.synsets

training_set = SynsetDataset(
    synsets, tokenizer, train_sent, train_label, config.Global.max_length, 'train', config)
validing_set = SynsetDataset(
    synsets, tokenizer, test_sent, test_label, config.Global.max_length, 'valid', config)

# %%
# Start generate dataloader
training_loader = DataLoader(
    training_set, **config.Dataloader.Train.to_dict())
validing_loader = DataLoader(
    validing_set, **config.Dataloader.Validate.to_dict())

# %%
# Start Building Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DualAttentiveBert(max_len=config.Global.max_length)
train_config = TrainConfig(
    Epoch=config.Model.epoch,
    LearningRate=config.Model.lr,
)
trainer = BaseTrainer(
    model=model,
    train_loader=training_loader,
    test_loader=validing_loader,
    train_config=train_config,
)

# %%
# Start Training
trainer.train()

# %%
# Start Validating
trainer.validate()
trainer.save()

# %%
# Generating Result
test_sent_ids, test_sent, test_labels = testloader(
    config.Global.testing_path)

testing_set = SynsetDataset(
    synsets, tokenizer, test_sent, test_labels, config.Global.max_length, 'test',config)

testing_loader = DataLoader(
    testing_set, **config.Dataloader.Test.to_dict())

def test(model, testing_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            defs = data['defs'].to(device, dtype=torch.float)
            outputs = model(ids, mask,defs, labels=None)
            preds = outputs.logits[:, :, 1]
            _, big_idx = torch.max(preds.data, dim=1)
            ans = data['location'][0][big_idx[0]]
            predictions.append(ans.item())
    return predictions

preds = test(model, testing_loader)

ans_df = pd.DataFrame(
    {
        "text_id": test_sent_ids,
        "word_id": preds
    }
)
ans_df["word_id"] = ans_df.apply(lambda x: str(x[0])+"_"+str(x[1]), axis=1)
ans_df.to_csv("sub.csv", index=False)
