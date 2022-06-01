import os
import xml.etree.ElementTree as ET
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Dict, List, Tuple

import pandas as pd


class Resolver(metaclass=ABCMeta):
    '''
    An resolver which has two getters sentences and answers.
    '''

    def __answer_Resolver__(self, answer_path: str=None) -> Dict[str, str]:
        if answer_path==None:
            return None
        return pd.read_csv(answer_path, index_col=0, squeeze=True).to_dict()

    @abstractmethod
    def __question_Resolver__(self, train_path: str) -> Dict[str, Dict[str, str]]:
        return NotImplementedError

    @abstractmethod
    def __len__(self):
        return NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        return NotImplementedError

    @abstractproperty
    def sentences(self) -> List[str]:
        return NotImplementedError

    @abstractproperty
    def answers(self) -> List[List[str]]:
        return NotImplementedError


class XMLResolver(Resolver):
    '''
    Turn XML training set into Data class
    '''

    def __init__(self, train_path: str, answer_path: str=None) -> None:
        self.ques = self.__question_Resolver__(train_path)
        self.ans = self.__answer_Resolver__(answer_path)

        self.__sentences__: List[str] = self.__gen_sent__()
        self.__answers__: List[List[str]] = self.__gen_answers__() if self.ans != None else []

    def __question_Resolver__(self, train_path: str) -> Dict[str, Dict[str, str]]:
        tree = ET.parse(train_path)
        root = tree.getroot()
        Questions: Dict[str, Dict[str, str]] = {}
        for child in root:
            Questions[child.attrib["id"]] = {}
            for row in child:
                Questions[child.attrib["id"]][row.attrib["id"]] = row.text
        return Questions

    def __gen_sent__(self) -> List[str]:
        return [' '.join([s for s in sent.values()]) for _, sent in self.ques.items()]

    def __gen_answers__(self) -> List[List[str]]:
        ans: List[str] = []
        for sentID in self.ques:
            row: List[str] = []
            for wordID in self.ques[sentID]:
                if wordID == self.ans[sentID]:
                    row.append(1)
                else:
                    row.append(0)
            ans.append(row)
        return ans

    def __len__(self):
        return len(self.__sentences__)

    def __getitem__(self, index) -> Tuple[List[str], List[List[str]]]:
        if self.__answers__==None:
            return self.__sentences__[index], [] 
        return self.__sentences__[index], self.__answers__[index]

    def sentences(self) -> List[str]:
        return self.__sentences__

    def answers(self) -> List[List[str]]:
        return self.__answers__
