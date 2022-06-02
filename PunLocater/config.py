import yaml
from dataclasses import dataclass
from pprint import pformat


@dataclass
class Config:
    @dataclass
    class __Global__:
        def __init__(
                self,
                training_path: str,
                answer_path: str,
                testing_path: str,
                pretrain_name: str,
                max_length: int,
                train_percent: float,
        ) -> None:
            self.training_path = training_path
            self.answer_path = answer_path
            self.pretrain_name = pretrain_name
            self.max_length = max_length
            self.train_percent = train_percent
            self.testing_path = testing_path

        def to_dict(self)->dict:
            return {
                "training_path": self.training_path,
                "answer_path": self.answer_path,
                "pretrain_name": self.pretrain_name,
                "max_length": self.max_length,
                "train_percent": self.train_percent,
            }

        def __repr__(self) -> str:
            return pformat(self.to_dict())

    @dataclass
    class __Dataloader__:
        @dataclass
        class __StageConfig__:
            def __init__(self, batch_size: int, num_workers: int, shuffle: bool) -> None:
                self.batch_size = batch_size
                self.num_workers = num_workers
                self.shuffle = shuffle
            
            def to_dict(self)->dict:
                return {
                    "batch_size": self.batch_size,
                    "num_workers": self.num_workers,
                    "shuffle": self.shuffle,
                }

            def __repr__(self) -> str:
                return pformat(self.to_dict())

        def __init__(self, dataloader: dict) -> None:
            self.Train = self.__StageConfig__(**dataloader["Train"])
            self.Validate = self.__StageConfig__(**dataloader["Validate"])
            self.Test = self.__StageConfig__(**dataloader["Test"])
        
        def to_dict(self)->dict:
            return {
                "Train": self.Train,
                "Validate": self.Validate,
                "Test": self.Test,
            }

        def __repr__(self) -> str:
            return pformat(self.to_dict())

    @dataclass
    class __Model__:
        def __init__(
                self,
                lr: float,
                epsilon: float,
                epoch: int,
        ) -> None:
            self.lr = lr
            self.epsilon = epsilon
            self.epoch = epoch
        def to_dict(self)->dict:
            return {
                "lr": self.lr,
                "epsilon": self.epsilon,
                "epoch": self.epoch,
            }
        def __repr__(self) -> str:
            return pformat(self.to_dict())

    def __init__(self, path: str) -> None:
        assert path.endswith(".yaml") or path.endswith(
            ".yml"), "expected to be yml/yaml file"
        self.Global = None
        self.Dataloader = None
        self.Model = None
        with open(path, 'r') as stream:
            dic: dict = yaml.load(stream, Loader=yaml.CLoader)
        for k, v in dic.items():
            if k == "Global":
                self.Global = self.__Global__(**v)
            elif k == "Dataloader":
                self.Dataloader = self.__Dataloader__(v)
            elif k == "Model":
                self.Model = self.__Model__(**v)
    def to_dict(self)->dict:
        return {
            "Global": self.Global,
            "Dataloader": self.Dataloader,
            "Model": self.Model,
        }
    def __repr__(self) -> str:
        return pformat(self.to_dict())


if __name__ == "__main__":
    c = Config('./config/submit/main.yaml')
    print(c)
    print(c.Global)
    print(c.Dataloader)
    print(c.Model)
