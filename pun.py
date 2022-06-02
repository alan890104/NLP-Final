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
import transformers
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForTokenClassification, BertTokenizer, BertConfig, BertModel

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


# %%
ques_dict, ques_list = read_ques('./training_set/data_homo_train.xml')
sub = pd.read_csv('.//training_set/benchmark_homo_train.csv')
test_ques_dict, test_ques_list = read_ques('./testing_set/data_homo_test.xml')

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
label2idx = {'O':0,'P': 1}
sentences = [' '.join([s[0] for s in sent]) for sent in getter.sentences]
labels = [[s[1] for s in sent] for sent in getter.sentences]
labels = [[label2idx.get(l) for l in lab] for lab in labels]


# %%
labels

# %%
MAX_LEN = 80
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 32
TEST_BATCH_SIZE = 1
EPOCHS = 5
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
        # 將 tokens_tensor 還原成文本
        tokens = [tokenizer.convert_ids_to_tokens(id) for id in ids]
        # 建立位置索引 
        locations = []
        location_dict = {}
        loc = 0
        for idx, token in enumerate(tokens):
            
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                locations.append(0)
            else:
                if '##' not in token:
                    loc += 1
                locations.append(loc)
                if location_dict.get(loc)==None:
                    location_dict[loc] = []
                location_dict[loc].append(idx)
        
        if self.mode == "test":
            label_tensor = None
            return {
                'token': tokens,
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'location': torch.tensor(locations, dtype=torch.long)
            } 
        else:
            origin_label = self.labels[index].index(1)+1
            new_label = location_dict[origin_label][0]
            label_vec = [0]*new_label + [1]
            label_vec.extend([0]*MAX_LEN)
            label_tensor = torch.tensor(label_vec[:MAX_LEN], dtype=torch.long)
            return {
                'token': tokens,
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'tags': label_tensor, 
                'location': torch.tensor(locations, dtype=torch.long)
            } 
    
    def __len__(self):
        return self.len


# %%
# Creating the dataset and dataloader for the neural network
train_percent = 1
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
training_set[8]

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
data = next(iter(training_loader))
tokens_tensors, masks_tensors, label_ids, locations = data['ids'], data['mask'], data['tags'], data['location']
tokens = np.array([' '.join([tokenizer.convert_ids_to_tokens(i.item()) for i in t]) for t in tokens_tensors])
print(f""" 
tokens.shape   = {tokens.shape} 
{tokens}
------------------------
tokens_tensors.shape   = {tokens_tensors.shape} 
{tokens_tensors}
------------------------
masks_tensors.shape    = {masks_tensors.shape}
{masks_tensors}
------------------------
label_ids.shape        = {label_ids.shape}
{label_ids}
------------------------
locations.shape        = {locations.shape}
{locations}
""")

# %%
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=2)

    def forward(self, ids, mask, labels=None):
        output_1= self.l1(ids, mask, labels = labels)
        return output_1

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = BERTClass()
model.to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


# %%
def train(epoch):
    model.train()
    nb_eval_steps, nb_eval_examples = 0, 0
    eval_loss = 0; eval_accuracy = 0
    n_correct = 0; n_wrong = 0; total = 0
    for _,data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['tags'].to(device, dtype = torch.long)
        # print(ids, targets)
        # tokened_words = [tokenizer.convert_ids_to_tokens(id) for id in ids]
        # tokened_words
        outputs = model(ids, mask, labels = targets)
        preds = outputs.logits[:,:,1]
        loss = loss_function(preds, torch.tensor(targets, dtype=float, requires_grad=True))
        big_val , big_idx = torch.max(preds, dim=1)
        target_val , target_idx = torch.max(targets, dim=1)
        # print(big_idx, target_idx)
        accuracy = (big_idx==target_idx).sum().item()
        print(accuracy,ids.size(0))
        eval_loss += loss.mean().item()
        eval_accuracy += accuracy
        nb_eval_examples += ids.size(0)
        nb_eval_steps += 1
        # optimizer.zero_grad()
        if _%2==1:
            print(f'Epoch: {epoch}, Loss:  {eval_loss/nb_eval_steps}, Acc: {eval_accuracy*100./nb_eval_examples} %')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# %%
for epoch in range(5):
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
            big_val , big_idx = torch.max(preds, dim=1)
            target_val , target_idx = torch.max(targets, dim=1)
            accuracy = (big_idx==target_idx).sum().item()
            eval_loss += loss.mean().item()
            eval_accuracy += accuracy
            nb_eval_examples += ids.size(0)
            nb_eval_steps += 1
        # print(np.array(predictions).shape, np.array(true_labels).shape)
        print("Validation loss: {}".format(eval_loss/nb_eval_steps))
        print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_examples))


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
            
            # print([tokenizer.convert_ids_to_tokens(d) for d in data['ids'].tolist()])
            outputs = model(ids, mask, labels=None)
            preds = outputs.logits[:,:,1]
            _ , big_idx = torch.max(preds.data, dim=1)
            # print(big_idx)
            ans = data['location'][0][big_idx[0]]
            predictions.append(ans.item())
    return predictions

# %%
preds = test(model, testing_loader)
preds

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
ans_df["word_id"] = ans_df.apply(lambda x: str(x[0])+"_"+str(x[1]), axis=1)


# %%
ans_df.to_csv("sub.csv", index=False)


