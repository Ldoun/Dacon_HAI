import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from lr_scheduler import base

class Trainer():
    def __init__(self, train_loader, valid_loader, model, loss_fn, optimizer, scheduler, device, patience, epochs, result_path, fold_logger, epoch=1):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.patience = patience
        self.epochs = epochs
        self.logger = fold_logger
        self.best_model_path = os.path.join(result_path, 'best_model')
        self.last_model_path = os.path.join(result_path, 'last_model')
        self.epoch = epoch
    
    def train(self):
        best = np.inf
        for epoch in range(1,self.epochs+1):
            print(f'lr: {self.scheduler.get_last_lr()}')
            loss_train, score_train = self.train_step()
            loss_val, score_val = self.valid_step()
            self.scheduler.step()

            self.logger.info(f'Epoch {str(epoch).zfill(5)}: t_loss:{loss_train:.3f} t_score:{score_train:.3f} v_loss:{loss_val:.3f} v_score:{score_val:.3f}')

            if loss_val < best:
                best = loss_val
                self.model.save_pretrained(self.best_model_path)
                torch.save({
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': [] if isinstance(self.scheduler, base) else self.scheduler.state_dict(),
                    'epoch': epoch,
                    'score_val': score_val,
                    'loss_val': loss_val, 
                }, os.path.join(self.best_model_path, 'state.pt'))
                bad_counter = 0

            else:
                bad_counter += 1

            if bad_counter == self.patience:
                break
            
            self.model.save_pretrained(self.last_model_path)
            torch.save({
                'optimizer': self.optimizer.state_dict(),
                'scheduler': [] if isinstance(self.scheduler, base) else self.scheduler.state_dict(),
                'epoch': epoch,
                'score_val': score_val,
                'loss_val': loss_val, 
            }, os.path.join(self.last_model_path, 'state.pt'))

    def train_step(self):
        self.model.train()

        total_loss = 0
        correct = 0
        for x, y in tqdm(self.train_loader, file=sys.stdout): #tqdm output will not be written to logger file(will only written to stdout)
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x, labels=y)
            loss = output.loss
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.shape[0]
            correct += sum(output.logits.argmax(dim=1) == y).item() # classification task
        
        return total_loss/len(self.train_loader.dataset), correct/len(self.train_loader.dataset)
    
    def valid_step(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            correct = 0
            for x, y in self.valid_loader:
                x, y = x.to(self.device), y.to(self.device)

                output = self.model(x, labels=y)
                loss = output.loss

                total_loss += loss.item() * x.shape[0]
                correct += sum(output.logits.argmax(dim=1) == y).item() # classification task
                
        return total_loss/len(self.valid_loader.dataset), correct/len(self.valid_loader.dataset)
    
    def test(self, test_loader):
        self.model.from_pretrained(self.best_model_path)
        self.model.eval()
        with torch.no_grad():
            result = []
            for batch in test_loader:
                x = x.to(self.device)
                output = self.model(x).logits.detach().cpu().numpy()
                result.append(output)

        return np.concatenate(result,axis=0)