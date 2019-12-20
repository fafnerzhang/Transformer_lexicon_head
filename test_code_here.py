import torch

from model import Transformer
from config import Config
import pandas as pd
def str_to_float(x):
    return float(x)
def Lexicon_tokenizer( x):
    result = x.replace("tensor([", '').replace(',', '').replace("])", '').split(" ")
    result = list(map(str_to_float,result))
    return result
def create_masks(src):
    src_mask = (src == 1).unsqueeze(-2)
    return src_mask

def padding(lexicon):
    pad_len = 400
    seq_len = lexicon.count(',')+1
    lexicon = lexicon + " ,0.001" * (pad_len-seq_len)
    return lexicon

model = Transformer(config=Config,src_vocab=8)

Tensor = torch.tensor([[2,3,4,5,6,7,1,1,1,1,1,1],[4,5,4,2,3,4,5,1,1,1,1,1]])
Lexicon = torch.tensor([[1,0.1,0.1,0.1,0.1,0.1,0.1,0,1,1,1,1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,1,1,1,1]])
mask = create_masks(Tensor)
#print(mask)
model(Tensor,mask,Lexicon)
