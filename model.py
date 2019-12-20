# Model.py

import torch
import torch.nn as nn
from copy import deepcopy
from train_utils import Embeddings,PositionalEncoding
from attention import MultiHeadedAttention
from encoder import EncoderLayer, Encoder
from feed_forward import PositionwiseFeedForward
import numpy as np
from utils import *
from tqdm import tqdm

def create_masks(src):
    src_mask = (src == 1).unsqueeze(-2)
    return src_mask

class Transformer(nn.Module):
    def __init__(self, config, src_vocab):
        super(Transformer, self).__init__()
        self.config = config
        
        h, N, dropout = self.config.h, self.config.N, self.config.dropout
        d_model, d_ff = self.config.d_model, self.config.d_ff
        sent_len = self.config.max_sen_len
        attn = MultiHeadedAttention(h, d_model,sent_len)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        self.encoder = Encoder(EncoderLayer(config.d_model, deepcopy(attn), deepcopy(ff), dropout), N)
        self.src_embed = nn.Sequential(Embeddings(config.d_model, src_vocab), deepcopy(position)) #Embeddings followed by PE

        # Fully-Connected Layer
        self.fc = nn.Linear(
            self.config.d_model,
            self.config.output_size
        )
        
        # Softmax non-linearity
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x,mask=None,Lexicon=None):
        #embedded_sents = self.src_embed(x.permute(1,0)) # shape = (batch_size, sen_len, d_model)
        embedded_sents = self.src_embed(x)
        encoded_sents = self.encoder(embedded_sents,mask,Lexicon)
        
        # Convert input to (batch_size, d_model) for linear layer
        final_feature_map = encoded_sents[:,-1,:]

        final_out = self.fc(final_feature_map)
        return final_out
    
    def add_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def add_loss_op(self, loss_op):
        self.loss_op = loss_op
    
    def reduce_lr(self):
        print("Reducing LR")
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / 2
                
    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []
        
        # Reduce learning rate as number of epochs increase
        if (epoch == int(self.config.max_epochs/3)) or (epoch == int(2*self.config.max_epochs/3)):
            self.reduce_lr()

        for i, batch in enumerate(tqdm(train_iterator)):
            self.optimizer.zero_grad()
            mask = create_masks(batch.text)
            if torch.cuda.is_available():
                x = batch.text.cuda()
                y = (batch.label).type(torch.cuda.LongTensor)
            else:
                x = batch.text
                y = (batch.label).type(torch.LongTensor)
            Lexicon = (batch.lexicon).type(torch.cuda.FloatTensor)
            y_pred = self.__call__(x,mask.cuda(),Lexicon)

            loss = self.loss_op(y_pred, y)
            #print("prediction:",y_pred,"label:",y,"loss",loss)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.data.cpu().numpy())

    
            """if i % 10000 == 0:
                print("Iter: {}".format(i+1))
                avg_train_loss = np.mean(losses)
                train_losses.append(avg_train_loss)
                print("\tAverage training loss: {:.5f}".format(avg_train_loss))
                losses = []
                
                # Evalute Accuracy on validation set
                val_accuracy = evaluate_model(self, val_iterator)
                print("\tVal Accuracy: {:.4f}".format(val_accuracy))
                self.train()"""
        self.eval()
        avg_train_loss = np.mean(losses)
        train_losses.append(avg_train_loss)
        print("\n\tAverage training loss: {:.5f}".format(avg_train_loss))
        # Evalute Accuracy on validation set
        val_accuracy ,F1_score= evaluate_model(self, val_iterator)
        print("\tVal Accuracy: {:.4f}".format(val_accuracy))
        print("\tVal F1: {:.4f}".format(F1_score))
        self.train()
        torch.save(self.state_dict(),"epoch_{}_{:.4f}.pt".format(epoch,val_accuracy))
        return train_losses, val_accuracies