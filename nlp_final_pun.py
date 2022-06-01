# %%
import xml.etree.ElementTree as ET
from pprint import pprint
# Importing the libraries needed
import pandas as pd
import torch
import seaborn as sns
import transformers
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
import logging
import numpy as np
from sklearn.model_selection import train_test_split

# %%
def read_ques(filename):
    ques_dict = {}
    ques_list = []
    tree = ET.parse(filename)
    root = tree.getroot()
    for child in root:
        temp_list =[]
        ques_dict[child.attrib['id']] = {}
        # print(child.attrib['id'], child.text)
        for c in child:
            ques_dict[child.attrib['id']][c.attrib['id']] = c.text
            temp_list.append(c.text)
            # print(c.attrib['id'], c.text)
        ques_list.append(temp_list)
    return ques_dict, ques_list

# %%
ques_dict, ques_list = read_ques('./training_set/data_homo_train.xml')
sub = pd.read_csv('./training_set/benchmark_homo_train.csv')
test_ques_dict, test_ques_list = read_ques('./testing_set/data_homo_test.xml')

# %%
def get_dataset(ques_dict, sub):
    ans_dict = {}
    for a in sub.iterrows():
        i = int(a[1].text_id.split("_")[-1])
        s = int(a[1].word_id.split("_")[-1])
        ans_dict[i] = s

    tokens, sent_ids, labels = [], [], []
    for i,v in ques_dict.items():
        sent_index = int(i.split("_")[-1])
        for ii, vv in v.items():
            token_index = int(ii.split("_")[-1])
            sent_ids.append(sent_index)
            tokens.append(vv)
            labels.append('P' if token_index==ans_dict[sent_index] else 'O')

    assert len(tokens)==len(sent_ids)==len(labels), "length must be same"
    dataset = pd.DataFrame(data={
        'tokens': tokens,
        'sentence_id': sent_ids,
        'label': labels
    })
    return dataset
dataset = get_dataset(ques_dict, sub)
dataset[dataset.sentence_id==15]

# %%
class SentenceGetter(object):
    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w,p) for w,p in zip(s["tokens"].values.tolist(),
                                                       s['label'].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_id").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(dataset)

# %%
labels_vals = list(set(dataset["label"].values))
label2idx = {t: i for i, t in enumerate(labels_vals)}
sentences = [' '.join([s[0] for s in sent]) for sent in getter.sentences]
labels = [[s[1] for s in sent] for sent in getter.sentences]
labels = [[label2idx.get(l) for l in lab] for lab in labels]

# %%
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForTokenClassification, BertTokenizer, BertConfig, BertModel


# %%
# Defining some key variables that will be used later on in the training
MAX_LEN = 80
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 2
TEST_BATCH_SIZE = 1
EPOCHS = 6
LEARNING_RATE = 2e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# %%
class CustomDataset(Dataset):
    def __init__(self, tokenizer, sentences, labels, max_len, mode):
        assert mode in ["train", "valid", "test"]
        self.mode = mode
        self.len = len(sentences)
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        sentence = str(self.sentences[index])
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        if self.mode == "test":
            label_tensor = None
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
            } 
        else:
            label = self.labels[index]
            label.extend([0]*MAX_LEN)
            label_tensor = torch.tensor(label[:MAX_LEN], dtype=torch.long)
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'tags': label_tensor
            } 
    
    def __len__(self):
        return self.len

# %%
# Creating the dataset and dataloader for the neural network
train_percent = 0.8
train_size = int(train_percent*len(sentences))
# train_dataset=df.sample(frac=train_size,random_state=200).reset_index(drop=True)
# valid_dataset=df.drop(train_dataset.index).reset_index(drop=True)
train_sentences = sentences[0:train_size]
train_labels = labels[0:train_size]

valid_sentences = sentences[train_size:]
valid_labels = labels[train_size:]

print("FULL Dataset: {}".format(len(sentences)))
print("TRAIN Dataset: {}".format(len(train_sentences)))
print("VALID Dataset: {}".format(len(valid_sentences)))

training_set = CustomDataset(tokenizer, train_sentences, train_labels, MAX_LEN, 'train')
validing_set = CustomDataset(tokenizer, valid_sentences, valid_labels, MAX_LEN, 'valid')


# %%
training_set[16]

# %%
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

valid_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
validing_loader = DataLoader(validing_set, **valid_params)

# %%
# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model. 

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=2)
        # self.l2 = torch.nn.Dropout(0.3)
        # self.l3 = torch.nn.Linear(768, 1)
    
    def forward(self, ids, mask, labels=None):
        output_1= self.l1(ids, mask, labels = labels)
        # output_2 = self.l2(output_1[0])
        # output = self.l3(output_2)
        return output_1

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)
model = BERTClass()
model.to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

# %%
def train(epoch):
    model.train()
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['tags'].to(device, dtype = torch.long)
        # print(ids, targets)
        # tokened_words = [tokenizer.convert_ids_to_tokens(id) for id in ids]
        # tokened_words
        outputs = model(ids, mask, labels = targets)
        print(outputs)
        preds = outputs.logits[:,:,1]
        loss = loss_function(preds, torch.tensor(targets, dtype=float, requires_grad=True))
        # optimizer.zero_grad()
        if _%100==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# %%
for epoch in range(EPOCHS):
    train(epoch)

# %%
def flat_accuracy(preds, labels):
    flat_preds = np.argmax(preds, axis=2).flatten()
    flat_labels = labels.flatten()
    return np.sum(flat_preds == flat_labels)/len(flat_labels)

# %%
def valid(model, validing_loader):
    model.eval()
    eval_loss = 0; eval_accuracy = 0
    n_correct = 0; n_wrong = 0; total = 0
    predictions , true_labels = [], []
    nb_eval_steps, nb_eval_examples = 0, 0
    with torch.no_grad():
        for _, data in enumerate(validing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['tags'].to(device, dtype = torch.long)
            outputs = model(ids, mask, labels=targets)
            preds = outputs.logits[:,:,1]
            loss = loss_function(preds, torch.tensor(targets, dtype=float))
            _ , big_idx = torch.max(preds, dim=1)
            print(big_idx, [tokenizer.convert_ids_to_tokens(id) for id in ids])
            _ , target_idx = torch.max(targets, dim=1)
            print(big_idx, target_idx)
            accuracy = (big_idx==target_idx).sum().item()
            eval_loss += loss.mean().item()
            eval_accuracy += accuracy
            nb_eval_examples += ids.size(0)
            nb_eval_steps += 1
        # print(np.array(predictions).shape, np.array(true_labels).shape)
        eval_loss = eval_loss/nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

# %%
# To get the results on the validation set. This data is not seen by the model
valid(model, validing_loader)

# %%
def get_test_data(ques_dict):
    tokens, sent_ids = [], []
    for i,v in ques_dict.items():
        sent_index = int(i.split("_")[-1])
        
        for ii, vv in v.items():
            sent_ids.append(sent_index)
            tokens.append(vv)

    assert len(tokens)==len(sent_ids), "length must be same"
    dataset = pd.DataFrame(data={
        'tokens': tokens,
        'sentence_id': sent_ids,
        'label': None
    })
    return dataset
test_data = get_test_data(test_ques_dict)
test_data[test_data.sentence_id==1885]

# %%
test_getter = SentenceGetter(test_data)
test_sentences = [' '.join([s[0] for s in sent]) for sent in test_getter.sentences]
test_labels = None

# %%
testing_set = CustomDataset(tokenizer, test_sentences, test_labels, MAX_LEN, 'test')

# %%
testing_set[1]

# %%
test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': False,
                'num_workers': 0
                }

testing_loader = DataLoader(testing_set, **test_params)

# %%
def test(model, testing_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            print(data['ids'][0].tolist())
            print([tokenizer.convert_ids_to_tokens(d) for d in data['ids'].tolist()])

            outputs = model(ids, mask, labels=None)
            preds = outputs.logits[:,:,1]
            print(preds)
            _ , big_idx = torch.max(preds.data, dim=1)
            predictions.append(list(big_idx.cpu().numpy()))
    return predictions

# %%
preds = test(model, testing_loader)
preds = np.squeeze(preds).tolist()

# %%


# %%
sentsId = list(set(test_data.sentence_id.to_list()))
sentsId.sort()

# %%
ans_df = pd.DataFrame(
    {
        "text_id": sentsId,
        "word_id": preds
    }
)
ans_df["text_id"] = ans_df["text_id"].apply(lambda x: "hom_"+str(x))
ans_df["word_id"] = ans_df.apply(lambda x: str(x[0])+"_"+str(x[1]+1), axis=1)

# %%
ans_df.to_csv("sub.csv", index=False)

# %%
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
sentence = "Tokyo to report nearly 370 new coronavirus cases, setting new single-day record"



# %%
encodings = tokenizer(sentence, return_offsets_mapping=True)
print(encodings.char_to_token())
for token_id, pos in zip(encodings['input_ids'], encodings['offset_mapping']):
    print(token_id, pos, sentence[pos[0]:pos[1]])

# %%



