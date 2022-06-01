# %%
# Importing
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from Model.BERT import BERTTokenClassification
from Resolver import XMLResolver
from Tokenizer import CustomDataset
from Trainer import BaseTrainer, TrainConfig


TRAIN_PATH = "./data/training_set/data_homo_train.xml"
ANSWER_PATH = "./data/training_set/benchmark_homo_train.csv"
PRETRAIN_NAME = "bert-base-cased"
TRAIN_PERCENT = 0.7
MAX_LEN = 80
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 16


# %%
# Step1 : get data from files
data = XMLResolver(TRAIN_PATH, ANSWER_PATH)

# %%
# Step2 : specify tokenizers
tokenizer = BertTokenizer.from_pretrained(PRETRAIN_NAME)

# %%
# Step3 : split into train and test
train_size = int(TRAIN_PERCENT*len(data))
train_sent, train_label = data[:train_size]
test_sent, test_label = data[train_size:]

# %%
# Step4 : get train & test dataset
train_dataset = CustomDataset(
    tokenizer,train_sent, train_label, max_len=MAX_LEN,mode="train")
test_dataset = CustomDataset(
    tokenizer,test_sent, test_label, max_len=MAX_LEN,mode="valid")

# %%
# Step5 : set config for train and test dataloader
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


# %%
# Step6 : select model to train
model = BERTTokenClassification(PRETRAIN_NAME, num_labels=2)

# %%
# Step7 : set config for trainer
train_config = TrainConfig(LearningRate=2e-5, LoggingInterval=2, Epoch=100)
trainer = BaseTrainer(model, train_loader, test_loader,
                      train_config=train_config)

# %%
# Step 8 : train and validate
trainer.train()
trainer.validate()
trainer.save()

# %%
