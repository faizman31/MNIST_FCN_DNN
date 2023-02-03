from copy import deepcopy
import numpy as np

import torch

class Trainer():
    
    def __init__(self,model,optimizer,crit):
        self.model=model
        self.optimizer=optimizer
        self.crit = crit

        super().__init__()

    def _train(self,train_loader,config):
        self.model.train()

        total_loss = 0

        for i , (x_i,y_i) in enumerate(train_loader):
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i,y_i.squeeze())

            self.optimizer.zero_grad()
            loss_i.backward()

            self.optimizer.step()

            if config.verbose >=2:
                print('Train Iteration(%d/%d) : loss=%.4e'%(i+1,len(train_loader),float(loss_i)))
            
            total_loss += float(loss_i)

        return total_loss / len(train_loader)
            
    def _validate(self,valid_loader,config):
        self.model.eval()

        with torch.no_grad():
            total_loss = 0 

            for i,(x_i,y_i) in enumerate(valid_loader):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i,y_i.squeeze())

                if config.verbose >= 2:
                    print('Valid Iteration(%d/%d) : loss=%.4e'%(i+1,len(valid_loader),float(loss_i)))

                total_loss += float(loss_i)

            return total_loss / len(valid_loader)

    def train(self,train_loader,valid_loader,config):
        lowest_loss = np.inf
        best_model = None

        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_loader,config)
            valid_loss = self._validate(valid_loader,config)

            if valid_loss < lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            print('Epoch(%d/%d) : train_loss=%.4e valid_loss=%.4e lowest_loss=%.4e'%(
                epoch_index+1,
                config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss
            ))

        
        self.model.load_state_dict(best_model)
