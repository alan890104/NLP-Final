from Model.BERT import BERTTokenClassification
from Resolver import XMLResolver
from Trainer import TrainConfig, BaseTrainer
from TrainingSet import BaseTrainigSet


TRAIN_PATH = "./data/training_set/data_homo_train.xml"
ANSWER_PATH = "./data/training_set/benchmark_homo_train.csv"
PRETRAIN_NAME = "bert-base-cased"
TRAIN_PERCENT = 0.7
MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8


trainer = BaseTrainigSet(
    resolver=XMLResolver,
    model=BERTTokenClassification,
    trainer=BaseTrainer,
    trainConfig=TrainConfig(Epoch=20),
    TRAIN_PATH=TRAIN_PATH,
    ANSWER_PATH=ANSWER_PATH,
    PRETRAIN_NAME=PRETRAIN_NAME,
    TRAIN_PERCENT=TRAIN_PERCENT,
    MAX_LEN=MAX_LEN,
    TRAIN_BATCH_SIZE=TRAIN_BATCH_SIZE,
    VALID_BATCH_SIZE=VALID_BATCH_SIZE,
)
