# NLP Final
徐煜倫、蔡育呈、蔡書維、王耀德


## How to execute

1. clone the repo
```console
git clone https://github.com/alan890104/NLP-Final.git
```

2. go into PunLocater
```console
cd ./NLP-Final/PunLocater
```

3. prepare requirements
```console
pip install -r requirements.txt
```

4. run main.py with yaml settings, for example:
```console
python main.py -c ./config/test/roberta.yaml
```


## Result
![Model Design on Dual Attentive Network](https://i.imgur.com/Vc77yHV.png)

## Reference

[A Dual-Attention Neural Network for Pun
Location and Using Pun-Gloss Pairs for
Interpretation](https://arxiv.org/pdf/2110.07209.pdf)
Reach the highest F1



[The system analysis and Research based on pun recognition](https://iopscience.iop.org/article/10.1088/1742-6596/2044/1/012190/pdf)

[Supervised Pun Detection and Location
with Feature Engineering and Logistic Regression](http://ceur-ws.org/Vol-2624/paper3.pdf)


[SemEval-2017 Task 7: Detection and Interpretation of English Puns](https://aclanthology.org/S17-2005/)


[Introduction to Pun](https://yz-joey.github.io/files/Pun.pdf)

[Pun Recongnition](https://github.com/joey1993/pun-recognition)