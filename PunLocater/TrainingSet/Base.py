from torch.utils.data import Dataset,DataLoader
from torch.nn import Module
from Resolver import Resolver
from Tokenizer import CustomDataset
from Trainer import TrainConfig,Trainer
from transformers import AutoTokenizer

def  BaseTrainigSet(
    resolver: Resolver,
    model:Module,
    trainer: Trainer,
    trainConfig: TrainConfig,
    **kwargs
):
    TRAIN_PATH = kwargs["TRAIN_PATH"]
    ANSWER_PATH = kwargs["ANSWER_PATH"]
    PRETRAIN_NAME = kwargs["PRETRAIN_NAME"]
    TRAIN_PERCENT = kwargs["TRAIN_PERCENT"]
    MAX_LEN = kwargs["MAX_LEN"]
    TRAIN_BATCH_SIZE = kwargs["TRAIN_BATCH_SIZE"]
    VALID_BATCH_SIZE = kwargs["VALID_BATCH_SIZE"]

    print("- Start resolve training set")
    data = resolver(TRAIN_PATH, ANSWER_PATH)

    print("- Start loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAIN_NAME)

    print("- Start Train Test Split")
    train_size = int(TRAIN_PERCENT*len(data))
    train_sent, train_label = data[:train_size]
    test_sent, test_label = data[train_size:]

    print("- Start Generate Dataset")
    train_dataset = CustomDataset(
    tokenizer,train_sent, train_label, max_len=MAX_LEN,mode="train")
    test_dataset = CustomDataset(
        tokenizer,test_sent, test_label, max_len=MAX_LEN,mode="valid")

    print("- Start Generate DataLoader")
    train_params = {
        'batch_size': TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0, }
    test_params = {
        'batch_size': VALID_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0, }

    train_loader = DataLoader(train_dataset, **train_params)
    test_loader = DataLoader(test_dataset, **test_params)

    print("- Start Modeling")
    model = model(PRETRAIN_NAME, num_labels=2, **kwargs)

    trainer = trainer(model, train_loader, test_loader,
                        train_config=trainConfig)
    
    print("- Start Training")
    trainer.train()
    print("- Start Validating")
    trainer.validate()
    trainer.save()