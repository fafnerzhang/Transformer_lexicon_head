import torchtext
import torch
from model import Transformer
from config import Config
from utils import evaluate_model
import torch.optim as optim
from torchtext import data
from torch import nn
import spacy
import pandas as pd
import numpy as np
import time

def parse_label(label):

    return int(label.replace("__label__", ""))

def str_to_float(x):
    return float(x) *1000

def Lexicon_tokenizer(lexicon):
    return list(map(str_to_float,lexicon.replace(' ','').split(",")))

def process_csv(name,function):
    df = pd.read_csv(name,header=0)
    df['lexicon'] = df['lexicon'].apply(function)
    df.to_csv("processed"+name,index=False)
    print("process complete")
def padding(lexicon):
    pad_len = 400
    seq_len = lexicon.count(',') + 1
    lexicon = lexicon + " ,0.001" * (pad_len - seq_len)
    return lexicon


#create text tokenizer
NLP = spacy.load('en_core_web_sm')
tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]

device = torch.device('cuda')

# Creating Field for data
TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=Config.max_sen_len, batch_first=True)
LABEL = data.Field(sequential=False,use_vocab=False)
Lexicon = data.Field(sequential=True, tokenize=Lexicon_tokenizer,use_vocab=False,fix_length=Config.max_sen_len,batch_first=True)

time_start=time.time()

#building itreator
train_data=torchtext.data.TabularDataset(path="twitter_train_lexicon.csv",fields=[('index',None),('label',LABEL),('text',TEXT),('lexicon',Lexicon)],format='csv',skip_header=True)
valid_data=torchtext.data.TabularDataset(path="twitter_test_lexicon.csv",fields=[('index',None),('label',LABEL),('text',TEXT),('lexicon',Lexicon)],format='csv',skip_header=True)

TEXT.build_vocab(train_data)
train_iter , valid_iter = data.BucketIterator.splits(datasets=(train_data,valid_data),batch_size=Config.batch_size,
                                                    sort_key=lambda x: len(x.text),
                                                    repeat=False,
                                                    shuffle=True)

time_end=time.time()
print(time_end - time_start,'s complete the processed')

print("vocab size",len(TEXT.vocab))
model = Transformer(Config, len(TEXT.vocab))
print(model)
model.cuda()
model.train()
model.load_state_dict(torch.load('epoch_0_0.4479.pt'))
optimizer = optim.Adam(model.parameters(), lr=Config.lr)
loss = nn.CrossEntropyLoss()
model.add_optimizer(optimizer)
model.add_loss_op(loss)
train_losses = []
val_accuracies = []
#    val_accuracy, F1_score = evaluate_model(model, dataset.val_iterator)
#    print("\tVal Accuracy: {:.4f}".format(val_accuracy))

for i in range(Config.max_epochs):
    print ("\nEpoch: {}".format(i))
    train_loss,val_accuracy = model.run_epoch(train_iter, valid_iter, i)
    train_losses.append(train_loss)
    val_accuracies.append(val_accuracy)

train_acc,train_F1 = evaluate_model(model, train_iter)
val_acc ,val_F1= evaluate_model(model, valid_iter)


print ('Final Training Accuracy: {:.4f}'.format(train_acc))
print('Final Training F1: {:.4f}'.format(train_F1))
print ('Final Validation Accuracy: {:.4f}'.format(val_acc))
print ('Final Validation F1: {:.4f}'.format(val_F1_acc))

