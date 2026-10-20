import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from utils.model_utils import LargeMargin
import numpy as np
import copy

class Userbase:
    """
    Base class for users in federated learning.
    """
    def __init__(self, device, id, train_data, model, batch_size = 0, learning_rate = 0, local_epochs = 0,localstep=5):

        self.device = device
        self.model = copy.deepcopy(model)
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.trainloader = DataLoader(train_data, self.batch_size,shuffle=True)
        self.trainloaderfull = DataLoader(train_data, self.batch_size)
        self.iter_trainloader = iter(self.trainloader)
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.localstep=localstep
    def settest(self,test_data):
        self.test_samples = len(test_data)
        self.testloader =  DataLoader(test_data, self.batch_size)
        self.testloaderfull = DataLoader(test_data, self.test_samples)
    
    def set_parameters(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()
        '''
        self.model=copy.deepcopy(model)
        self.local_model=copy.deepcopy(self.model)
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.detach()
        for local_param, new_param in zip(self.local_model.parameters(), model.parameters()):
            local_param.data = new_param.data.detach()
        '''
        #self.local_weight_updated = copy.deepcopy(self.optimizer.param_groups[0]['params'])

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()
    
    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param
    
    def get_updated_parameters(self):
        return self.local_weight_updated
    
    def update_parameters(self, new_params):
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grad=[]
        for param_1,param_0 in zip(self.model.parameters(),self.local_model):
            param=param_0.data-param_1.data
            #param=param_0.data-param_0.data
            grad=param.data.view(-1) if not len(grad) else torch.cat((grad,param.view(-1)))
        return grad
    def get_param(self):
        param=[]
        for param_temp in self.model.parameters():
            param=param_temp.data.view(-1) if not len(param) else torch.cat((param,param_temp.view(-1)))
        return param
    def get_params(self):
        param=[]
        local_dict=self.model.cpu().state_dict()
        for key in local_dict:
            param = local_dict[key].data.view(-1) if not len(param) else torch.cat((param, local_dict[key].data.view(-1)))
        return param


    def test(self):
        self.model.eval()
        test_acc = 0
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            #@loss += self.loss(output, y)
            #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            #print(self.id + ", Test Loss:", loss)
        return test_acc, y.shape[0]

    def train_error_loss_new(self, epochs):
        LOSS = 0
        train_acc=0
        self.model.eval()
        self.model.to(self.device)
        self.model.eval()
        for batchidx, (inputs, targets) in enumerate(self.trainloaderfull):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            #loss = self.loss(outputs, targets)
            train_acc += (torch.sum(torch.argmax(outputs, dim=1) == targets)).item()
            LOSS += self.loss(outputs, targets).item()
            # LOSS+= loss.item()

                # dis = LargeMargin(self.model, targets, outputs)
                # batch_size = len(dis)
                # batch_sum = torch.sum(dis, axis=0).item()
                #
                # batch_LargeMargin.append(batch_sum)

            '''
            if self.id==1:
                self.model.eval()
                for batchidx,(inputs,targets) in enumerate(self.trainloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss=self.loss(outputs, targets)
                    LOSS+= loss.item()
                print('id 1 model loss')
                print(LOSS)
                self.local_model.eval()
                LOSS=0
                for batchidx,(inputs,targets) in enumerate(self.trainloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.local_model(inputs)
                    loss=self.loss(outputs, targets)
                    LOSS+= loss.item()
                print('id 1 local model loss')
                print(LOSS)
            '''
        # sum_LargeMargin = np.sum(batch_LargeMargin, axis=0)
        # num_largeMargin = len(batch_LargeMargin) * batch_size
        # avg_LargeMargin = sum_LargeMargin/num_largeMargin

        return train_acc, LOSS , self.train_samples


    def train_error_and_loss(self):
        self.model.eval()
        self.model.to(self.device)
        train_acc = 0
        loss = 0

        for x, y in self.trainloader:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)

            #print(self.id + ", Train Accuracy:", train_acc)
            #print(self.id + ", Train Loss:", loss)

        return train_acc, loss , self.train_samples
    
    
    def get_next_train_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_trainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_trainloader = iter(self.trainloader)
            (X, y) = next(self.iter_trainloader)
        return (X.to(self.device), y.to(self.device))
    
    def get_next_test_batch(self):
        try:
            # Samples a new batch for persionalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X.to(self.device), y.to(self.device))

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))



    
    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))
