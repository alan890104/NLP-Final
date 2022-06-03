import os
from abc import abstractmethod
from dataclasses import dataclass
import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

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
        Epsilon: float = 1e-8,
        **kwargs
    ) -> None:
        self.Epoch = Epoch
        self.LearningRate = LearningRate
        self.Epsilon = Epsilon
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def train(self):
        return NotImplementedError

    @abstractmethod
    def validate(self):
        return NotImplementedError

    @abstractmethod
    def save(self):
        return NotImplementedError


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
        use_gpu: bool = True,
        **kwargs
    ) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.model: Module = model.to(self.device)
        self.train_loader: DataLoader = train_loader
        self.test_loader: DataLoader = test_loader
        self.train_config: TrainConfig = train_config if train_config != None else TrainConfig()
        self.optimizer: AdamW = AdamW(
            params=self.model.parameters(), lr=train_config.LearningRate, eps=self.train_config.Epsilon)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*self.train_config.Epoch)
        print(self.model)

    def train(self):
        self.model.train()
        nb_eval_steps, nb_eval_examples = 0, 0
        eval_loss = 0
        eval_accuracy = 0
        for epoch in tqdm(range(self.train_config.Epoch)):
            for _, data in enumerate(self.train_loader, 0):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                targets = data['tags'].to(self.device, dtype=torch.long)

                outputs = self.model(ids, mask, labels=targets)
                loss = outputs.loss
                preds = outputs.logits[:, :, 1]
                t, big_idx = torch.max(preds, dim=1)
                t, target_idx = torch.max(targets, dim=1)

                accuracy = (big_idx == target_idx).sum().item()
                eval_loss += loss.mean().item()
                eval_accuracy += accuracy
                nb_eval_examples += ids.size(0)
                nb_eval_steps += 1

                writer.add_scalar("Loss/train", eval_loss /
                                  nb_eval_steps, nb_eval_steps)
                writer.add_scalar("Accuracy/train", eval_accuracy *
                                  100./nb_eval_examples, nb_eval_steps)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

    def validate(self):
        self.model.eval()
        eval_loss = 0
        eval_accuracy = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        with torch.no_grad():
            for _, data in enumerate(self.test_loader, 0):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                targets = data['tags'].to(self.device, dtype=torch.long)
                outputs = self.model(ids, mask, labels=targets)
                preds = outputs.logits[:, :, 1]
                big_val, big_idx = torch.max(preds, dim=1)
                target_val, target_idx = torch.max(targets, dim=1)
                accuracy = (big_idx == target_idx).sum().item()
                eval_loss += outputs.loss.mean().item()
                eval_accuracy += accuracy
                nb_eval_examples += ids.size(0)
                nb_eval_steps += 1
            print("Validation loss: {}".format(eval_loss/nb_eval_steps))
            print("Validation Accuracy: {}".format(
                eval_accuracy/nb_eval_examples))

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
