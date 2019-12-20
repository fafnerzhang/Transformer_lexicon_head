# config.py

class Config(object):
    N = 6 #6 in Transformer Paper
    d_model = 512 #512 in Transformer Paper
    d_ff = 1024 #2048 in Transformer Paper
    h = 8
    dropout = 0.1
    output_size = 20
    lr = 0.0003
    max_epochs = 300
    batch_size = 64
    max_sen_len = 128