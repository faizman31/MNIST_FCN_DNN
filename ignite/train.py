import argparse 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from trainer import *
from model import *
from utils import *
from data_loader import *

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn',required=True)
    p.add_argument('--gpu_id',type=int,default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio',default=.8,type=float)
    
    p.add_argument('--batch_size',type=int,default=256)
    p.add_argument('--n_epochs',type=int,default=5)
    
    p.add_argument('--n_layers',type=int,default=5)
    p.add_argument('--use_dropout',action='store_true')
    p.add_argument('--dropout_p',type=float,default=.3)

    p.add_argument('--verbose',type=int,default=1)

    config = p.parse_args()

    return config

def main(config):
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d'%(config.gpu_id))

    train_loader ,valid_loader , test_loader = get_loaders(config)

    print('Train :',len(train_loader.dataset))
    print('Valid :',len(valid_loader.dataset))
    print('Test :',len(test_loader.dataset))

    model = ImageClassifier(28**2,10).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.CrossEntropyLoss()

    trainer = Trainer(config)
    trainer.train(model,crit,optimizer,train_loader,valid_loader)

if __name__ == "__main__":
    config = define_argparser()
    main(config)
