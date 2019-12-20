# utils.py

import torch
from torchtext import data
import spacy
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def create_masks(src):
    src_mask = (src == 1).unsqueeze(-2)
    return src_mask

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}
    
    def parse_label(self, label):
        '''
        Get the actual labels from label string
        Input:
            label (string) : labels of the form '__label__2'
        Returns:
            label (int) : integer value corresponding to label string
        '''
        return int(label.replace("__label__",""))

    def get_pandas_df(self, filename):
        '''
        Load the data into Pandas.DataFrame object
        This will be used to convert data to torchtext object
        '''
        with open(filename, 'r',encoding="utf-8") as datafile:
            data = [line.strip().split(',', maxsplit=1) for line in datafile]
            data_text = list(map(lambda x: x[1], data))
            data_label = list(map(lambda x: self.parse_label(x[0]), data))
            data_lexicon = list(map(lambda x:x[2],data))
        full_df = pd.DataFrame({"text":data_text, "label":data_label,"lexicon":data_lexicon})
        return full_df

    def str_to_float(self,x):
        return float(x)
    def Lexicon_tokenizer(self,x):
        result = x.replace("tensor([", '').replace(',', '').replace("])", '').split(" ")
        result = list(map(self.str_to_float, result))
        return result

    def load_data(self, train_file, test_file=None, val_file=None):
        '''
        Loads the data from files
        Sets up iterators for training, validation and test data
        Also create vocabulary and word embeddings based on the data
        
        Inputs:
            train_file (String): path to training file
            test_file (String): path to test file
            val_file (String): path to validation file
        '''

        NLP = spacy.load('en_core_web_sm')
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
        
        # Creating Field for data
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len,batch_first=True)
        LABEL = data.Field(sequential=False, use_vocab=False)
        Lexicon = data.Field(sequential=True, tokenize=self.Lexicon_tokenizer,lower=True)
        datafields = [("text",TEXT),("label",LABEL),("lexicon",Lexicon)]
        
        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = self.get_pandas_df(train_file)
        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)
        
        test_df = self.get_pandas_df(test_file)
        test_examples = [data.Example.fromlist(i, datafields) for i in test_df.values.tolist()]
        test_data = data.Dataset(test_examples, datafields)
        
        # If validation file exists, load it. Otherwise get validation data from training data
        if val_file:
            val_df = self.get_pandas_df(val_file)
            val_examples = [data.Example.fromlist(i, datafields) for i in val_df.values.tolist()]
            val_data = data.Dataset(val_examples, datafields)
        else:
            train_data, val_data = train_data,train_data
        
        TEXT.build_vocab(train_data)
        self.vocab = TEXT.vocab
        
        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)
        
        self.val_iterator, self.test_iterator = data.BucketIterator.splits(
            (val_data, test_data),
            batch_size=self.config.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)
        
        print ("Loaded {} training examples".format(len(train_data)))
        print ("Loaded {} test examples".format(len(test_data)))
        print ("Loaded {} validation examples".format(len(val_data)))

def evaluate_model(model, iterator):
    all_preds = []
    all_y = []
    torch.cuda.empty_cache()
    model.eval()
    with torch.no_grad():
        for idx,batch in enumerate(iterator):
            if torch.cuda.is_available():
                x = batch.text.cuda()
            else:
                x = batch.text
            #print(x)
            mask = create_masks(x)
            lexicon = (batch.lexicon).type(torch.cuda.FloatTensor)
            y_pred = model(x,mask,lexicon)
            #print(y_pred)
            predicted = torch.max(y_pred.cpu().data, 1)[1]

            all_preds.extend(predicted.numpy())
            all_y.extend(batch.label.numpy())
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    f1 = f1_score(all_y,np.array(all_preds).flatten(),average="macro")
    return score,f1