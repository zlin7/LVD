import torch
import numpy as np
import tqdm
import pandas as pd
import utils.utils as utils
import _settings
import torch.nn as nn
import ipdb

#List of base regression model names
_LINEAR_REGRESSION = 'linreg'
_SHALLOW_NN = 'ShallowNN'
_KERNEL_MLKR = "KernelRegMLKR"

#==============================================Implement a bunch of simple models
class BaseAlgo(object):
    def __init__(self):
        super(BaseAlgo, self).__init__()
        assert hasattr(self, 'ALGO_NAME'), "Please give this algorithm a name"
    def fit(self, X, Y):
        raise NotImplementedError()
    def predict(self, X):
        raise NotImplementedError()

class LinearRegression(BaseAlgo):
    ALGO_NAME = _LINEAR_REGRESSION
    def __init__(self):
        super(LinearRegression, self).__init__()
        from sklearn.linear_model import LinearRegression as temp_model_name
        self.model = temp_model_name()
    def fit(self, X, Y):
        return self.model.fit(X, Y)
    def predict(self, X):
        return self.model.predict(X)

class _shallow_NN(nn.Module):
    def __init__(self, input_size, hidden_nodes=[100]):
        super(_shallow_NN, self).__init__()

        hidden_nodes = [input_size] + hidden_nodes
        self.layers = nn.ModuleList([nn.Linear(hn, hidden_nodes[i+1]) for i, hn in enumerate(hidden_nodes[:-1])])
        self.readout_layer = nn.Linear(hidden_nodes[-1], 1)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = torch.relu(self.layers[i](x))
        return self.readout_layer(x).squeeze(1)

class ShallowNN(BaseAlgo):
    ALGO_NAME = _SHALLOW_NN
    def __init__(self,
                 n_layers=1,
                 num_hidden=100,
                 seed=7,
                 lr=1e-3, batch_size=100, n_iters=500):
        super(ShallowNN, self).__init__()
        #use relu and MSE
        self.seed = seed
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size
        self.n_iters = n_iters
        self.model = None
        self.input_size = None
        self.hidden_nodes = [num_hidden for _ in range(n_layers)]
        self.lr = lr

    def fit(self, X, Y, quiet=True):
        assert self.model is None and self.input_size is None
        utils.set_all_seeds(self.seed, quiet=quiet)
        X = torch.tensor(X).float().cuda()
        Y = torch.tensor(Y).float().cuda()
        self.input_size = X.shape[1]
        self.model = _shallow_NN(self.input_size, self.hidden_nodes).cuda()
        fidx = np.arange(len(Y))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self._loss_hist = []
        iter_range = range(self.n_iters) if quiet else tqdm.tqdm(range(self.n_iters))
        for _ in iter_range:
            idx = np.random.choice(fidx, self.batch_size) if self.batch_size is not None else fidx
            yhat = self.model.forward(X[idx])
            optimizer.zero_grad()
            loss = self.criterion(yhat, Y[idx])
            loss.backward()
            optimizer.step()
            self._loss_hist.append(loss.cpu().detach().numpy())
        self.model = self.model.to('cpu')

    def predict(self, X):
        X = torch.tensor(X).float()
        yhat = self.model.forward(X)
        return yhat.cpu().detach().numpy()

class ScaleModule(nn.Module):
    def __init__(self, ndim, init_scales=None):
        super(ScaleModule, self).__init__()
        if init_scales is None:
            temp = nn.Linear(ndim, 1)
            self.weight = nn.Parameter(temp.weight.data, requires_grad=True)
        else:
            assert len(init_scales) == ndim
            w = torch.tensor(init_scales)
            self.weight = nn.Parameter(w, requires_grad=True)

    def forward(self, x):
        return x * self.weight

class KernelMLKR(BaseAlgo):
    ALGO_NAME = _KERNEL_MLKR
    def __init__(self,
                 d=20,
                 seed=7,
                 max_n = 1000,
                 lr=1e-3, n_iters=100,
                 batch_size=100,
                 stop_iters=20,
                 device='cuda',
                 **kwargs #Adding kwargs seems fine (not breaking the caching..?
                 ):
        super(KernelMLKR, self).__init__()
        self.seed = seed
        self.criterion = nn.MSELoss()
        self.n_iters = n_iters
        self.d = d
        self.max_n = max_n
        self.lr = lr
        self.batch_size = batch_size
        self.stop_iters = stop_iters

        self.A_form = kwargs.get('A_form', 'mat') #vec(mean scaling) or matrix
        assert self.A_form in {'mat', 'vec'}
        self.ybar_bias = kwargs.get('ybar_bias', False)
        assert self.ybar_bias in {True, False}

        self.to_normalize = kwargs.get('norm', True)
        self.x_mean = self.x_std = None


        #parameters
        self.input_size = self.Ybar = self.AX = self.Y = self.A = None
        self._loss_hist = []

        #mems
        self._device = device

    def preprocess(self, x):
        if self.x_mean is not None and self.x_std is not None:
            val_idx = self.x_std > 1e-3
            x = (x[..., val_idx] - self.x_mean[val_idx]) / self.x_std[val_idx]
        return x

    def to(self, device='cuda'):
        self._device = device
        if self.A is not None:
            self.A = self.A.to(self._device)
        if self.AX is not None:
            self.AX = self.AX.to(self._device)
        if self.Y is not None:
            self.Y = self.Y.to(self._device)
        if self.Ybar is not None:
            self.Ybar = self.Ybar.to(self._device)
        return self


    def fit(self, X, Y, quiet=False):
        assert self.input_size is None
        N = len(Y)
        if self.to_normalize:
            self.x_mean = np.mean(X, 0)
            self.x_std = np.std(X, 0)

        X = self.preprocess(X)
        X = torch.tensor(X, requires_grad=False).float().to(self._device)
        Y = torch.tensor(Y, requires_grad=False).float().to(self._device)
        if self.ybar_bias:
            Ybar_j = (torch.sum(Y) - Y) / (N-1)
        utils.set_all_seeds(self.seed, quiet=quiet)

        if self.max_n is not None and N < self.max_n: self.max_n = N - 1

        self.input_size = X.shape[1]
        if self.A_form == 'mat':
            self.A = torch.nn.Linear(self.input_size, self.d, bias=False).to(self._device)
            self.A.train()
        else:
            scales = 1. / (X.std(0) * (self.input_size ** 0.5))
            self.A = ScaleModule(self.input_size, scales)

        #some prep for training
        iter_range = range(self.n_iters) if quiet else tqdm.tqdm(range(self.n_iters))
        optimizer = torch.optim.Adam(self.A.parameters(), lr=self.lr)
        fidx = np.arange(N)
        _eps = torch.tensor(1e-7, requires_grad=False).to(self._device)
        _zero = torch.zeros(1, requires_grad=False).to(self._device)

        #Start training
        best_results = (np.infty, -1, self.A.weight.detach().cpu().numpy())
        self._loss_hist = []
        for curr_iter in iter_range:
            optimizer.zero_grad()
            loss = 0
            idx = np.random.choice(fidx, self.batch_size, replace=False) if self.batch_size is not None and self.batch_size < len(fidx) else fidx
            for i in idx:
                diff_i = self.A(X[i] - X)
                if self.ybar_bias:
                    #valid_k = torch.exp(-torch.pow(diff_i, 2).sum(-1))
                    valid_k = torch.exp(-torch.pow(diff_i, 2).mean(-1))
                    valid_y = torch.cat([Y[:i], Ybar_j[i:i + 1], Y[i + 1:]], dim=0)
                else:
                    #kij = torch.exp(-torch.pow(diff_i, 2).sum(-1))
                    kij = torch.exp(-torch.pow(diff_i, 2).mean(-1))
                    valid_k = torch.cat([kij[:i], _zero, kij[i + 1:]], dim=0)
                    valid_y = Y

                #skip some kijs for speed
                if self.max_n is not None:
                    subidx_i = torch.topk(valid_k, self.max_n, sorted=False).indices
                    subKij = valid_k[subidx_i]
                    yhat_i = torch.sum(subKij * valid_y[subidx_i])/torch.max(_eps, torch.sum(subKij))
                    #if torch.sum(subKij) < _eps: ipdb.set_trace()
                else:
                    yhat_i = torch.sum(valid_k * valid_y) / torch.max(_eps, torch.sum(valid_k))
                    #if torch.sum(valid_k) < _eps: ipdb.set_trace()
                #ipdb.set_trace()
                curr_loss = self.criterion(yhat_i, Y[i])
                curr_loss.backward(retain_graph=False)
                loss += float(curr_loss)
            if torch.isnan(self.A.weight.grad).sum() > 0:
                print("Encountered invalid gradient. Exitting...")
                break
            #ipdb.set_trace()
            optimizer.step()
            loss /= len(idx)
            self._loss_hist.append(loss)
            if loss < best_results[0]:
                best_results = (loss, curr_iter, self.A.weight.detach().cpu().numpy())
            elif curr_iter > best_results[1] + self.stop_iters:
                print("Not improving for a while.. Exitting...")
                break
        self.A.weight.data = torch.tensor(best_results[2]).float()
        self.A = self.A.to(self._device)
        self.AX = self.A(X)
        self.Y = Y
        #if self.ybar_bias:
        self.Ybar = torch.mean(Y).detach()

    def K(self, x1, x2=None):
        if x2 is None: return 1.
        x1 = self.preprocess(x1)
        x2 = self.preprocess(x2)
        with torch.no_grad():
            diff = torch.tensor(x1 - x2, device=self._device, dtype=torch.float)
            kij = torch.exp(-torch.pow(self.A(diff), 2).mean(-1))
            return float(kij.detach())

    def Ki(self, xi, Xs, speedup_info=None):
        #the speedup info is AX
        xi = self.preprocess(xi)
        AX = speedup_info
        with torch.no_grad():
            if AX is None:
                Xs = self.preprocess(Xs)
                AX = self.A(torch.tensor(Xs, device=self._device, dtype=torch.float, requires_grad=False))
            else:
                if AX.device != self._device:
                    AX = AX.to(self._device)
            xi = torch.tensor(xi, device=self._device, dtype=torch.float)
            diff_i = self.A(xi) - AX
            kij = torch.exp(-torch.pow(diff_i, 2).mean(-1))
            return kij, AX

    def predict_with_info(self, x, batch=None):
        x = self.preprocess(x)
        with torch.no_grad():
            if batch is None:
                AX = self.AX
                Ys = self.Y
            else:
                bidx = np.random.choice(len(self.Y), batch)#replace=True
                AX = self.AX[bidx]
                Ys = self.Y[bidx]
            x = torch.tensor(x, device=self._device, dtype=torch.float)
            if self.ybar_bias:
                AX = torch.cat([AX, self.A(x).unsqueeze(0)], dim=0)
                Ys = torch.cat([Ys, self.Ybar.unsqueeze(0)], dim=0)
            diff_i = self.A(x) - AX
            kij = torch.exp(-torch.pow(diff_i, 2).mean(-1))
            yhat = torch.sum(kij * Ys) / torch.sum(kij)
            if np.isnan(float(yhat)): yhat = float(self.Ybar)
            return float(yhat), kij

    def predict(self, x, batch=None):
        return self.predict_with_info(x, batch)[0]



def get_base_model(algo_name=_LINEAR_REGRESSION):
    if algo_name == _LINEAR_REGRESSION: return LinearRegression
    if algo_name == _SHALLOW_NN: return ShallowNN
    if algo_name == _KERNEL_MLKR: return KernelMLKR

