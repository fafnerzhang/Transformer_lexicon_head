# train.py

from utils import *
from model import *
from config import Config
import sys
import torch.optim as optim
from torch import nn
import torch

if __name__=='__main__':
    config = Config()
    train_file = '../data/IMDB_extracted.csv'
    if len(sys.argv) > 2:
        train_file = sys.argv[1]
    test_file  = '../data/IMDB_extracted.csv'
    if len(sys.argv) > 3:
        test_file = sys.argv[2]
    
    dataset = Dataset(config)
    dataset.load_data(train_file, test_file)
    
    # Create Model with specified optimizer and loss function
    ##############################################################
    model = Transformer(config, len(dataset.vocab))
    print(model)
    if torch.cuda.is_available():
        model.cuda()
    #model.load_state_dict(torch.load("epoch_0_0.2160.pt"))
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    NLLLoss = nn.CrossEntropyLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(NLLLoss)
    ##############################################################
    
    train_losses = []
    val_accuracies = []
#    val_accuracy, F1_score = evaluate_model(model, dataset.val_iterator)
#    print("\tVal Accuracy: {:.4f}".format(val_accuracy))

    for i in range(config.max_epochs):
        print ("Epoch: {}".format(i))
        train_loss,val_accuracy = model.run_epoch(dataset.train_iterator, dataset.test_iterator, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)

    train_acc,train_F1 = evaluate_model(model, dataset.train_iterator)
    val_acc ,val_F1= evaluate_model(model, dataset.val_iterator)
    test_acc ,test_F1= evaluate_model(model, dataset.test_iterator)

    print ('Final Training Accuracy: {:.4f}'.format(train_acc))
    print('Final Training F1: {:.4f}'.format(train_F1))
    print ('Final Validation Accuracy: {:.4f}'.format(val_acc))
    print('Final Training F1: {:.4f}'.format(val_F1))
    print ('Final Test Accuracy: {:.4f}'.format(test_acc))
    print('Final Training F1: {:.4f}'.format(test_F1))