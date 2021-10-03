
# Copyright (c) 2020, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# ---------------------------------------------------------
# Base classes for feedforward, convolutional and recurrent 
# neural network (DNN, CNN, RNN) dj_models in pytorch
# ---------------------------------------------------------


from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
import sys
import ipdb
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

import torch
from torch.autograd import Variable 
import torch.nn.functional as nnf
from torch.utils.data import random_split
from torch.optim import SGD 
from torch.distributions import constraints
import torchvision as torchv
import torchvision.transforms as torchvt
from torch import nn
import torchvision.transforms as transforms
from torch.autograd import grad
import scipy.stats as st

from sklearn.preprocessing import StandardScaler
from copy import deepcopy
import time

from models.baselines.djkp.dj_utils.parameters import *
torch.manual_seed(1) 

from models.baselines.djkp.influence.influence_computation import *
from models.baselines.djkp.influence.influence_utils import *


class linearRegression(torch.nn.Module):
    
    def __init__(self, inputSize, outputSize):
        
        super(linearRegression, self).__init__()
        
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        
        out         = self.linear(x)
        
        return out


class LinearRegression(torch.nn.Module):
    
    def __init__(self, inputDim=1, outputDim=1, learningRate=0.01, epochs=1000):
        
        super(LinearRegression, self).__init__()
        
        self.model         = linearRegression(inputSize=inputDim, outputSize=outputDim) 
        
        if torch.cuda.is_available():
            
            self.model.cuda()
        
        self.inputDim      = inputDim   # takes variable 'x' 
        self.outputDim     = outputDim  # takes variable 'y'
        self.learningRate  = learningRate 
        self.epochs        = epochs
        self.loss_fn       = torch.nn.MSELoss() 
        self.optimizer     = torch.optim.SGD(self.model.parameters(), lr=self.learningRate)

    def forward(self, x):
        
        out         = self.model(x)
        
        return out
    
    def fit(self, x_train, y_train, verbosity=True):
        
        self.X         = torch.tensor(x_train.reshape((-1, self.inputDim))).float()
        self.y         = torch.tensor(y_train).float()
        self.losses    = []
        
        for epoch in range(self.epochs):
            
            # Converting inputs and labels to Variable
            
            if torch.cuda.is_available():
                
                inputs = Variable(torch.from_numpy(x_train).cuda()).float()
                labels = Variable(torch.from_numpy(y_train).cuda()).float()
            
            else:
                
                inputs = Variable(torch.from_numpy(x_train)).float()
                labels = Variable(torch.from_numpy(y_train)).float()

            
            self.model.zero_grad()

            self.optimizer.zero_grad()

            # get output from the model, given the inputs
            outputs = self.model(inputs)

            # get loss for the predicted output
            self.loss = self.loss_fn(outputs, labels)
            
            # get gradients w.r.t to parameters
            self.loss.backward(retain_graph=True)
            self.losses.append(self.loss.detach().numpy())
        
            # update parameters
            self.optimizer.step()

            if verbosity:

                print('epoch {}, loss {}'.format(epoch, self.loss.item()))
        
        
    def predict(self, x_test, numpy_output=True):
        
        if(type(x_test)==torch.Tensor):
                    
            predicted = self.forward(x_test.float())
   
        else:
                    
            predicted = self.forward(torch.tensor(x_test).float())
              
            
        if numpy_output:
                
            predicted = predicted.detach().numpy()
        
        return predicted

    def update_loss(self):
        
        self.loss = self.loss_fn(self.predict(self.X, numpy_output=False), self.y)     


class DE_NLLLoss(nn.Module):
    def __init__(self, reduction='none'):
        super(DE_NLLLoss, self).__init__()
        self.reduction = reduction
    def forward(self, y_pred, y):
        mu_hat, sigma_sq_hat = y_pred[:, 0], y_pred[:, 1]
        y = y[:, 0]
        #sigma_sq_hat_pos = torch.log(1 + torch.exp(sigma_sq_hat)) + 1e-6
        #sigma_sq_hat_pos = torch.stack([sigma_sq_hat_pos, sigma_sq_hat.abs()]).min(0)[0] #Necessary
        sigma_sq_hat_pos = torch.abs(sigma_sq_hat) + 1e-6
        loss = torch.log(sigma_sq_hat_pos) + torch.div(torch.square(y - mu_hat), sigma_sq_hat_pos)
        #if torch.isinf(loss).any():
        #    ipdb.set_trace()
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss

class DNN(nn.Module):
    
    def __init__(self, 
                 n_dim=1, 
                 dropout_prob=0.0,
                 dropout_active=False,  
                 num_layers=2, 
                 num_hidden=100,
                 output_size=1,
                 activation="Tanh",  
                 mode="Regression"
                ):
        
        super(DNN, self).__init__()
        
        self.n_dim          = n_dim
        self.num_layers     = num_layers
        self.num_hidden     = num_hidden
        self.mode           = mode
        self.activation     = activation
        self.device         = torch.device('cpu') # Make this an option
        self.output_size    = output_size
        self.dropout_prob   = dropout_prob
        self.dropout_active = dropout_active  
        self.model          = build_architecture(self)


    def fit(self, X, y, learning_rate=1e-3, loss_type="MSE", batch_size=100, num_iter=500, verbosity=False, clip=None):


        if self.n_dim!=X.shape[1]:

            self.n_dim   = X.shape[1]
            self.model   = build_architecture(self)

        self.X           = torch.tensor(X.reshape((-1, self.n_dim))).float()
        self.y           = torch.tensor(y).float()
        
        loss_dict        = {"MSE": torch.nn.MSELoss, "DE_NLL": DE_NLLLoss}
        
        self.loss_fn     = loss_dict[loss_type](reduction='mean')
        self.loss_trace  = []     
        
        batch_size       = np.min((batch_size, X.shape[0]))
        
        optimizer        = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for _ in range(num_iter):

            batch_idx = np.random.choice(list(range(X.shape[0])), batch_size )
            
            y_pred    = self.model(self.X[batch_idx, :])
            
            self.model.zero_grad()
            
            optimizer.zero_grad()                   # clear gradients for this training step

            self.loss = self.loss_fn(y_pred.reshape((batch_size, -1)), self.y[batch_idx].reshape((batch_size, -1)))
        
            self.loss.backward(retain_graph=True)   # backpropagation, compute gradients
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), clip)
            #ipdb.set_trace()
            #if torch.isnan(self.model[0].weight.grad).any():
            #    ipdb.set_trace()
            optimizer.step()

            self.loss_trace.append(self.loss.detach().numpy())
            
            if verbosity:
    
                print("--- Iteration: %d \t--- Loss: %.3f" % (_, self.loss.item()))
    

    def predict(self, X, numpy_output=True):
        
        X = torch.tensor(X.reshape((-1, self.n_dim))).float()

        if numpy_output:

            prediction = self.model(X).detach().numpy()

        else:

            prediction = self.model(X)    


        return prediction


    def update_loss(self):
        
        self.loss = self.loss_fn(self.predict(self.X, numpy_output=False), self.y)          



class DNN_uncertainty_wrapper():
    
    def __init__(self, model, mode="exact", damp=1e-4, order=1):
        
        self.model            = model
        losses, _ = self.eval_model()
        self.model.loss = torch.mean(torch.stack(losses, 0))
        self.IF               = influence_function(model, train_index=list(range(model.X.shape[0])), 
                                                   mode=mode, damp=damp, order=order)
        self.LOBO_residuals   = [] 

        for k in range(len(self.IF)):
            #print(k)
            perturbed_models  = perturb_model_(self.model, self.IF[k])
            
            ####
            #perturbed_models  = DNN(**params)
            #perturbed_models.fit(np.delete(model.X, k, axis=0).detach().numpy(), 
            #                     np.delete(model.y.detach().numpy(), k, axis=0), **train_params)
            ####

            self.LOBO_residuals.append(np.abs(np.array(self.model.y[k]).reshape(-1, 1) - np.array(perturbed_models.predict(model.X[k, :])).T))
    
            del perturbed_models
        
        self.LOBO_residuals   = np.squeeze(np.array(self.LOBO_residuals))

    def eval_model(self, with_grad=True, sub_index=None):
        losses = []
        pred_ys = []
        with (torch.enable_grad() if with_grad else torch.no_grad()):
            for i, tx in enumerate(self.model.X):
                ty = self.model.y[i]
                if sub_index is not None and i not in sub_index: continue
                pred_y = self.model.model(tx.unsqueeze(0))
                losses.append(self.model.loss_fn(pred_y, ty.unsqueeze(0)))
                pred_ys.append(pred_y)
        return losses, pred_ys


    def predict(self, X_test, coverage=0.95):

        self.variable_preds   = []
        num_samples           = np.array(X_test).shape[0]
        
        for k in range(len(self.IF)):
            #print(k)
            perturbed_models  = perturb_model_(self.model, self.IF[k])

            ####
            #perturbed_models  = DNN(**params)
            #perturbed_models.fit(np.delete(model.X, k, axis=0).detach().numpy(), 
            #                     np.delete(model.y.detach().numpy(), k, axis=0), **train_params)            
            ####

            self.variable_preds.append(perturbed_models.predict(X_test).reshape((-1,)))
    
            del perturbed_models

        self.variable_preds   = np.array(self.variable_preds)     

        y_upper               = np.quantile(self.variable_preds + np.repeat(self.LOBO_residuals.reshape((-1, 1)), num_samples, axis=1), 1 - (1-coverage)/2, axis=0, keepdims=False)
        y_lower               = np.quantile(self.variable_preds - np.repeat(self.LOBO_residuals.reshape((-1, 1)), num_samples, axis=1), (1-coverage)/2, axis=0, keepdims=False)

        y_pred                = self.model.predict(X_test).reshape((-1,))
        #R                     = np.repeat(self.LOBO_residuals.reshape((-1, 1)), num_samples, axis=1)
        #V                     = np.abs(np.repeat(y_pred.reshape((-1, 1)), num_samples, axis=1) - self.variable_preds)
        
        #CI_limit              = np.quantile(V + R, 1 - (1-coverage)/2, axis=0, keepdims=False)

        #y_upper               = y_pred + CI_limit
        #y_lower               = y_pred - CI_limit

        return y_pred, y_lower, y_upper     


def Deep_ensemble(X_train, y_train, X_test, params, n_ensemble=5, train_frac=0.8):
    
    DEmodels = [DNN(**params) for _ in range(n_ensemble)]
    n_data   = X_train.shape[0]
    y_preds  = []
    
    for _ in range(n_ensemble):
    
        indexs   = np.random.choice(list(range(n_data)), int(np.floor(n_data * train_frac)), replace=False)
    
        DEmodels[_].fit(X_train[indexs, :], y_train[indexs])
        y_preds.append(DEmodels[_].predict(X_test).reshape((-1,)))
    
    y_pred   = np.mean(np.array(y_preds), axis=0)
    y_std    = np.std(np.array(y_preds), axis=0)
    
    return y_pred, y_std 


def Deep_ensemble_correct(X_train, y_train, X_test, params, n_ensemble=5, train_frac=0.8):
    params = params.copy()
    params['output_size'] = params.get('output_size', 1) * 2
    DEmodels = [DNN(**params) for _ in range(n_ensemble)]
    n_data = X_train.shape[0]
    y_preds = []

    for _ in range(n_ensemble):
        indexs = np.random.choice(list(range(n_data)), int(np.floor(n_data * train_frac)), replace=False)

        DEmodels[_].fit(X_train[indexs, :], y_train[indexs], loss_type='DE_NLL')
        y_preds.append(DEmodels[_].predict(X_test))  # .reshape((-1,)))
    output_mu_sample = np.array(y_preds)[:, :, 0]  # (M, B)
    out_sig_sq_sample = np.abs( np.array(y_preds)[:, :, 1]) + 1e-6  # (M, B)
    out_mu_sample_final = np.mean(output_mu_sample, 0)
    y_std = np.sqrt(np.mean(out_sig_sq_sample + np.square(output_mu_sample), axis=0) - np.square(out_mu_sample_final))
    assert not (pd.isnull(y_std).any() or pd.isnull(out_mu_sample_final).any())
    return out_mu_sample_final, y_std

class MCDP_DNN(DNN):
    
    def __init__(self, 
                 dropout_prob=0.5,
                 dropout_active=True,                  
                 n_dim=1, 
                 num_layers=2, 
                 num_hidden=200,
                 output_size=1,
                 activation="ReLU", 
                 mode="Regression"):
        
        super(MCDP_DNN, self).__init__()
        
        self.dropout_prob   = dropout_prob 
        self.dropout        = nn.Dropout(p=dropout_prob)
        self.dropout_active = True


    def forward(self, X):
        
        _out= self.dropout(self.model(X))  
        
        return _out

    
    def predict(self, X, alpha=0.1, MC_samples=100):
        
        z_c         = st.norm.ppf(1-alpha/2)
        X           = torch.tensor(X.reshape((-1, self.n_dim))).float()
        samples_    = [self.forward(X).detach().numpy() for u in range(MC_samples)]
        pred_sample = np.concatenate(samples_, axis=1)
        pred_mean   = np.mean(pred_sample, axis=1)  
        pred_std    = z_c * np.std(pred_sample, axis=1)         

        return pred_mean, pred_std           
