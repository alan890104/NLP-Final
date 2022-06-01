from dataclasses import dataclass
import os
from typing import List

import numpy as np
import torch
from seqeval.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sqlalchemy import true
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


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
        for epoch in range(self.train_config.Epoch):
            for _, batch in enumerate(self.train_loader,0):

                batch_input_ids = batch["ids"].to(self.device)
                batch_input_mask = batch["mask"].to(self.device)
                batch_labels = batch["labels"].to(self.device)

                output = self.model(
                    batch_input_ids,
                    batch_input_mask,
                    labels=batch_labels,
                )

                self.optimizer.zero_grad()
                output.loss.backward()
                self.optimizer.step()
                self.scheduler.step()

    def validate(self):
        # Turn on testing mode
        self.model.eval()
        eval_loss: float = 0
        y_pred: List[List[str]] = []
        y_true: List[List[str]] = []
        # testing and get classification report
        with torch.no_grad():
            for _, data in enumerate(self.test_loader):
                ids = data["ids"].to(self.device, dtype=torch.long)
                mask = data["mask"].to(self.device, dtype=torch.long)
                labels = data["labels"].to(self.device, dtype=torch.long)

                loss, logits = self.model(ids, mask, labels=labels)
                logits = logits.detach().cpu().numpy()
                label_ids = labels.to("cpu").numpy()
                y_true.append(label_ids)
                y_pred.extend([list(p) for p in np.argmax(logits, axis=2)])

                eval_loss += loss.mean().item()
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, digits=4)
        print("Eval loss: {}".format(eval_loss))
        print("Accuracy : {:<20s}".format(accuracy))
        print("Precision: {:<20s}".format(precision))
        print("Recall   : {:<20s}".format(recall))
        print("F1-score : {:<20s}".format(f1))
        print("Report")
        print(report)

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
