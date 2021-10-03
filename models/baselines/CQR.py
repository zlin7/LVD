import torch
import numpy as np
import tqdm
import pandas as pd
import utils.utils as utils
import _settings
import torch.nn as nn
import ipdb
import models.baselines.DNN as DNN

class QuantileLoss:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        pass
    def __call__(self, y_pred, y):
        assert len(y.shape) <= 1, "Can only be batch..."
        alpha = self.alpha
        diff = y-y_pred
        v1 = alpha * diff
        v2 = (1-alpha) * (-diff)
        dim = len(y.shape)
        return torch.stack([v1, v2], dim).max(dim)[0]

class BiQuantilesLoss:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        pass
    def __call__(self, y_pred, y):
        assert y_pred.shape[-1] == 2, "Has to be 2 quantile estimates"
        if len(y.shape) == 2:
            assert y.shape[1] == 1
            y = y[:, 0]
        lo_alpha = self.alpha / 2.
        hi_alpha = 1-lo_alpha
        diff_lo = y - y_pred[:, 0]
        diff_hi = y - y_pred[:, 1]
        dim = len(y.shape)
        loss_lo = torch.stack([lo_alpha * diff_lo, (1 - lo_alpha) * (-diff_lo)], dim).max(dim)[0]
        loss_hi = torch.stack([hi_alpha * diff_hi, (1 - hi_alpha) * (-diff_hi)], dim).max(dim)[0]
        return (loss_lo + loss_hi).mean()

class CQR(nn.Module):
    def __init__(self, alpha=0.1, model_class=1, model_kwargs={}, train_kwargs={}, seed=0):
        super(CQR, self).__init__()
        if isinstance(model_class, int):
            model_class, model_kwargs, train_kwargs = DNN.get_DNN_and_trainkwargs(setting_id=model_class)
        self.loss_fn = BiQuantilesLoss(alpha)
        train_kwargs['loss_type'] = self.loss_fn

        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.train_kwargs = train_kwargs

        self.qs_m = None
        self.alpha = alpha

        self.mean_m = None
        self.seed = seed

    def fit(self, X, Y, train_X, train_Y, m=None, mean_m = None, quiet=True):
        utils.set_all_seeds(self.seed)
        n_dim = X.shape[1]
        output_size = Y.shape[1] if len(Y.shape) == 2 else 1
        assert output_size == 1
        self.mean_m = mean_m


        #m can be a given quantile perdictor
        if m is None:
            m = self.model_class(n_dim=n_dim, output_size=2*output_size, **self.model_kwargs)
            m.fit(train_X, train_Y, verbosity=not quiet, **self.train_kwargs)
        self.qs_m = m

        qs_preds = self.qs_m.predict(X)
        #qs_preds = [self.qs_m.predict(xi)[0] for xi in X]
        self.es = np.asarray([max(qs_preds[i][0] - yi, yi - qs_preds[i][1]) for i, yi in enumerate(Y)])

        self.X, self.Y = X, Y
        self.train_X, self.train_Y = train_X, train_Y

    def PI(self, x, alpha=0.1):
        #assert alpha == self.alpha
        qs = self.qs_m.predict(x)[0]
        yhat = (qs[0] + qs[1]) / 2.
        if self.mean_m is not None: yhat = self.mean_m.predict(x)[0][0]
        w = np.percentile(self.es, 100*(1-alpha))
        return qs[0]-w, qs[1]+w, yhat, None


def CQR_from_results(val_y, val_ylo, val_yhi, test_ylo, test_yhi, alpha=0.1):
    es = np.stack([val_ylo - val_y, val_y - val_yhi], 1)
    es = np.max(es, 1)
    w = np.percentile(es, 100 * (1 - alpha))
    return test_ylo - w, test_yhi+w, (test_ylo+test_yhi)/2.
