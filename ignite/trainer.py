from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as torch_utils

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from utils import get_grad_norm,get_parameter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

class MyEngine(Engine):
    
    def __init__(self,func,model,crit,optimizer,config):
        self.model=model
        self.crit=crit
        self.optimizer=optimizer
        self.config=config

        super().__init__(func)

        self.best_loss = np.inf
        self.best_model = None
        
        self.device = next(model.parameters()).device

    @staticmethod
    def train(engine,mini_batch):
        engine.model.train()
        engine.optimizer.zero_grad()

        x,y=mini_batch
        x,y = x.to(engine.device) , y.to(engine.device)

        y_hat = engine.model(x)

        loss = engine.crit(y_hat,y)
        loss.backward()

        if isinstance(y,torch.LongTensor) or isinstance(y,torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat,dim=-1)==y).sum() / float(y.shape[0])
        else:
            accuracy = 0
        
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        g_norm = float(get_grad_norm(engine.model.parameters()))

        engine.optimizer.step()

        return {
            'loss' : float(loss),
            'accuracy' : accuracy,
            '|param|' : p_norm,
            '|g_param|': g_norm
        }

    @staticmethod
    def validate(engine,mini_batch):
        engine.model.eval()

        with torch.no_grad():
            x,y = mini_batch

            y_hat = engine.model(y)

            loss = engine.crit(y_hat,y)

        if isinstance(y,torch.LongTensor) or isinstance(y,torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat,dim=-1)==y).sum() / float(y.shape[0])
        else:
            accuracy = 0

        return {
            'loss' : float(loss),
            'accuracy' : float(accuracy)
        }
    
    @staticmethod
    def attach(train_engine,validation_engine,verbose=VEROBSE_BATCH_WISE):
        def attach_running_average(engine,metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name
            )
        
        training_metric_names = ['loss','accuracy','|param|','|g_param|']

        for metric_name in training_metric_names:
            attach_running_average(train_engine,metric_name)




class Trainer():
    def __init__(self,config):
        self.config=config

    def train(self,model,crit,optimizer,train_loader,valid_loader):
        
        train_engine = MyEngine(
            MyEngine.train,
            model,crit,optimizer,self.config
        )

        validation_engine = MyEngine(
            MyEngine.validate,
            model,crit,optimizer,self.config
        )

        MyEngine.attach(
            train_engine,
            validation_engine,
            verbose = self.config.verbose
        )
