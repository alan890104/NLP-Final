# %%
# Importing the libraries needed

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from Tokenizer import CustomDataset
from Trainer import BaseTrainer, TrainConfig
from Model import BERTClass
from Trainer.Base import TrainConfig
from Resolver import XMLResolver, testloader
from config import Config
import argparse
import random
import numpy as np

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

if __name__ == "__main__":
    # %%
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="relative path to config yaml file",
                        type=str, default="./config/submit/main.yaml")
    args = parser.parse_args()
    # %%
    # Start Load Config file
    config = Config(args.config)

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

    training_set = CustomDataset(
        tokenizer, train_sent, train_label, config.Global.max_length, 'train', config)
    validing_set = CustomDataset(
        tokenizer, test_sent, test_label, config.Global.max_length, 'valid', config)

    # %%
    # Start generate dataloader
    training_loader = DataLoader(
        training_set, **config.Dataloader.Train.to_dict())
    validing_loader = DataLoader(
        validing_set, **config.Dataloader.Validate.to_dict())

    # %%
    # Start Building Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BERTClass(config.Global.pretrain_name)
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

    testing_set = CustomDataset(
        tokenizer, test_sent, test_labels, config.Global.max_length, 'test')

    testing_loader = DataLoader(
        testing_set, **config.Dataloader.Test.to_dict())

    def test(model, testing_loader):
        model.eval()
        predictions = []
        with torch.no_grad():
            for _, data in enumerate(testing_loader, 0):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                outputs = model(ids, mask, labels=None)
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
