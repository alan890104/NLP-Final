from dataclasses import dataclass
import os
from typing import List

import numpy as np
import torch
from sqlalchemy import true
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

@dataclass
class TrainConfig:
    '''
    Configuration to train and test your nn Modules

    Default Params
    -----
    ```
    Epoch: int = 5
    LearningRate: float = 2e-5
    LoggingInterval: int = 10
    ```
    '''

    def __init__(
        self,
        Epoch: int = 5,
        LearningRate: float = 2e-5,
        LoggingInterval: int = 10,
        **kwargs
    ) -> None:
        self.Epoch = Epoch
        self.LearningRate = LearningRate
        self.LoggingInterval = LoggingInterval
        for k, v in kwargs.items():
            setattr(self, k, v)


class BaseTrainer:
    '''
    Default Trainer with simple train and test
    '''

    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        train_config: TrainConfig = None,
        use_gpu: bool = true,
        **kwargs
    ) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.model: Module = model.to(self.device)
        self.train_loader: DataLoader = train_loader
        self.test_loader: DataLoader = test_loader
        self.train_config: TrainConfig = train_config if train_config != None else TrainConfig()
        self.optimizer: AdamW = AdamW(
            params=self.model.parameters(), lr=train_config.LearningRate, eps=1e-8)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*self.train_config.Epoch)
        print(self.model)

    def train(self):
        # Turn on train mode
        self.model.train()

        # Start Training
        with tqdm(total=self.train_config.Epoch) as pbar:
            for epoch in range(self.train_config.Epoch):
                for batchnum, batch in enumerate(self.train_loader, 0):

                    batch_input_ids = batch["ids"].to(self.device)
                    batch_input_mask = batch["mask"].to(self.device)
                    batch_labels = batch["labels"].to(self.device)

                    output = self.model(
                        batch_input_ids,
                        batch_input_mask,
                        labels=batch_labels,
                    )
                    writer.add_scalar("Loss/train", output.loss, epoch)
                    self.optimizer.zero_grad()
                    output.loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=1)
                    self.optimizer.step()
                    self.scheduler.step()
                pbar.update()

    def validate(self):
        # Turn on testing mode
        self.model.eval()
        y_true = []
        y_pred = []
        # testing and get classification report
        with torch.no_grad():
            for _, data in enumerate(self.test_loader):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                targets = data['labels'].to(self.device, dtype=torch.long)
                output = self.model(ids, mask, labels=targets)
                preds = output.logits[:, :, 1]
                _, pred_idx = torch.max(preds, dim=1)
                _, true_idx = torch.max(targets, dim=1)
                y_pred.extend(pred_idx.tolist())
                y_true.extend(true_idx.tolist())
        print("Accuracy: ", accuracy_score(y_true, y_pred))

    def save(self, path: str = None):
        '''
        default save to models/{classname}
        '''
        if path == None:
            model_name = self.model.__class__.__name__
            path = "models/{}".format(model_name)
        if not os.path.exists("./models"):
            os.makedirs("./models")
        torch.save(self.model, path)
