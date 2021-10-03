import torch
import numpy as np
import tqdm
import pandas as pd
import utils.utils as utils
import _settings
import torch.nn as nn
import ipdb
import models.baselines.DNN as DNN


class MADSplit(nn.Module):
    def __init__(self, model_class=0, model_kwargs={}, train_kwargs={}, eps=1e-5, seed=0, abs_resid=True):
        super(MADSplit, self).__init__()
        if isinstance(model_class, int):
            model_class, model_kwargs, train_kwargs = DNN.get_DNN_and_trainkwargs(setting_id=model_class)
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.train_kwargs = train_kwargs
        self.resid_m = None
        self.mean_m = None
        self.eps = eps
        self.seed = seed
        self.abs_resid = abs_resid

    def fit(self, X, Y, train_X, train_Y, m=None, resid_m=None, quiet=True):
        utils.set_all_seeds(self.seed)
        n_dim = X.shape[1]
        output_size = Y.shape[1] if len(Y.shape) == 2 else 1

        #m can be a given mean predictor
        if m is None:
            m = self.model_class(n_dim=n_dim, output_size=output_size, **self.model_kwargs)
            m.fit(train_X, train_Y, verbosity=not quiet, **self.train_kwargs)
        self.mean_m = m

        #fit the residual predictor
        if resid_m is None:
            train_Yhat = self.mean_m.predict(train_X)
            #train_resids = np.asarray([abs(self.mean_m.predict(xi)[0] - yi) for xi, yi in zip(train_X, train_Y)])
            train_resids = np.asarray([abs(yhati[0] - yi) for yhati, yi in zip(train_Yhat, train_Y)])
            resid_m = self.model_class(n_dim=n_dim, output_size=output_size, **self.model_kwargs)
            resid_m.fit(train_X, train_resids, verbosity=not quiet, **self.train_kwargs)
        self.resid_m = resid_m

        if self.abs_resid:
            self.resids_scaled = np.abs((Y - self.mean_m.predict(X)[:, 0]) / (self.eps + abs(self.resid_m.predict(X)[:, 0])))
        else:
            self.resids_scaled = np.abs((Y - self.mean_m.predict(X)[:, 0])/(self.eps+self.resid_m.predict(X)[:,0]))
        #self.resids_scaled = np.asarray([abs(Y[i]-self.mean_m.predict(xi)[0]) / (self.resid_m.predict(xi)[0] + self.eps) for i, xi in enumerate(X)])
        self.X, self.Y = X, Y
        self.train_X, self.train_Y = train_X, train_Y

    def PI(self, x, alpha=0.05):
        yhat = self.mean_m.predict(x)[0][0]
        q = np.percentile(self.resids_scaled, 100*(1-alpha))
        if self.abs_resid:
            sigma = self.eps + abs(self.resid_m.predict(x)[0][0])
        else:
            sigma = self.eps + self.resid_m.predict(x)[0][0]
        return yhat-q * sigma, yhat+q * sigma, yhat, None


def MADSplit_from_results(valid_y, valid_yhat, valid_rhat, test_yhat, test_rhat, alpha=0.1, eps=1e-5):
    resids_scaled = np.abs((valid_y - valid_yhat) / (eps + np.abs(valid_rhat)))
    q = np.percentile(resids_scaled, 100 * (1 - alpha))
    test_rhat = np.abs(test_rhat) + eps
    return test_yhat - q * test_rhat, test_yhat + q * test_rhat, test_yhat
