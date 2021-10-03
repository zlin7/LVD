import numpy as np
import pandas as pd
import bisect
import tqdm
import utils.utils as utils
import _settings
import ipdb
import torch

_LocalConformal = "LocalConformal"
_LocalConformalMAD = "LocalConformalMAD"

class NaiveKernel():
    def __init__(self, type='Gaussian'):
        self._device = utils.gpuid_to_device(-1)
        self.type = type

    def to(self, device):
        self._device = device

    def K(self, x1, x2=None):
        if self.type == 'Gaussian':
            if x2 is None: return 1.
            with torch.no_grad():
                diff = torch.tensor(x1 - x2, device=self._device, dtype=torch.float)
                kij = torch.exp(-torch.pow(diff, 2).sum(-1))
                return float(kij.detach())
        raise NotImplementedError()

    def Ki(self, xi, Xs, speedup_info=None):
        if self.type == 'Gaussian':
            with torch.no_grad():
                if speedup_info is None:
                    speedup_info = torch.tensor(Xs, device=self._device, dtype=torch.float, requires_grad=False)
                Xs = speedup_info
                if Xs.device != self._device:
                    Xs = Xs.to(self._device)
                xi = torch.tensor(xi, device=self._device, dtype=torch.float)
                diff_i = xi - Xs
                # kij = torch.exp(-torch.pow(self.A(diff_i), 2).sum(-1))
                kij = torch.exp(-torch.pow(diff_i, 2).sum(-1))
                return kij, Xs

def weighted_quantile_faster(arr, weights, q):
    idx = np.argsort(arr)
    arr, weights = arr[idx], weights[idx]
    return weighted_quantile_faster2(arr, weights, q)

def weighted_quantile_faster2(arr, weights, q):
    qs = np.cumsum(weights)
    idx = bisect.bisect_left(qs, q, lo=0, hi=len(qs)-1)
    return arr[idx]

class PIConstructor:
    def __init__(self):
        pass

    def PI(self, x, alpha):
        raise NotImplementedError()

    @classmethod
    def eval(cls, x, y, PI, alpha=0.05, quiet=True, PI_kwargs={}, PI_list_kwargs={}):
        if not quiet:
            iter_ = tqdm.tqdm(enumerate(x), desc='eval_conformal', total=len(x))
        else:
            iter_ = enumerate(x)
        res = []
        for i, xi in iter_:
            PI_params = utils.merge_dict_inline({k:v[i] for k,v in PI_list_kwargs.items()}, PI_kwargs)
            lb, ub, yhat, extra = PI(xi, alpha=alpha, **PI_params)
            res.append({'lo': lb, 'hi': ub, 'y': y[i], 'yhat': yhat, 'extra': extra, 'index': i})
            if xi.shape[0] == 1: res[-1].update({"x": xi[0]})
        return pd.DataFrame(res)

class JKPlus(PIConstructor):
    def __init__(self, model_class):
        """
        :param model_class: can be models.regmodel.LinearRegression for example
        """
        super(JKPlus, self).__init__()
        self.base_model = model_class
        self.models = {}
        self.LOOresids = {}

    def fit(self, X, Y, m=None):
        self.X, self.Y, self.m = X, Y, m
        from sklearn.model_selection import LeaveOneOut
        loo = LeaveOneOut()
        total = loo.get_n_splits(self.X, self.Y)
        for train_idx, test_idx in tqdm.tqdm(loo.split(self.X), total=total, desc='Fit JKP'):
            x_train, y_train = self.X[train_idx], self.Y[train_idx]
            x_test, y_test = self.X[test_idx], self.Y[test_idx]
            model = self.base_model()
            model.fit(x_train, y_train)
            test_idx = test_idx[0]
            self.models[test_idx] = model
            pred_test = model.predict(x_test)
            self.LOOresids[test_idx] = y_test[0] - pred_test[0]
        self.full_model = self.base_model()
        self.full_model.fit(self.X, self.Y)

    def PI(self, x, alpha=0.05, quiet=True):
        x = np.expand_dims(x, 0)
        iter_ = self.models.items()
        if not quiet: iter_ = tqdm.tqdm(iter_, desc='predict')
        lo, hi, yhats = [], [], []
        for idx, model in iter_:
            resid = abs(self.LOOresids[idx])
            yhat = model.predict(x)[0]
            lo.append(yhat - resid)
            hi.append(yhat + resid)
            yhats.append(yhat)
        lo = np.quantile(np.array(lo), alpha)
        hi = np.quantile(np.array(hi), 1-alpha)
        yhats = np.asarray(yhats)
        interquantile = np.quantile(yhats, 1-alpha) - np.quantile(yhats, alpha)
        return lo, hi, self.full_model.predict(x)[0], interquantile

class VanillaJK(JKPlus):
    def __init__(self, model_class):
        super(VanillaJK, self).__init__(model_class)

    def PI(self, x, alpha=0.05, quiet=True):
        x = np.expand_dims(x, 0)
        _w = np.quantile(np.abs(pd.Series(self.LOOresids)), 1 - alpha)
        yhat = self.full_model.predict(x)[0]
        return yhat - _w, yhat + _w, yhat, 0

class VanillaSplit(PIConstructor):
    def __init__(self):
        super(VanillaSplit, self).__init__()

    def fit(self, X, Y, m):
        self.X, self.Y, self.m = X, Y, m
        self.resids = np.asarray([abs(m(xi) - yi) for xi, yi in zip(X, Y)])

    def PI(self, x, alpha=0.05):
        q = np.percentile(self.resids, 100*(1-alpha))
        yhat = self.m(x)
        return yhat-q, yhat+q, yhat, None

class LocalConditional(PIConstructor):
    def __init__(self, K_obj=None, #K_cuda=None,
                 device='cuda'):
        super(LocalConditional, self).__init__()
        if K_obj is None:
            K_obj = NaiveKernel(type='Gaussian')
        self.K_obj = K_obj
        self.K = K_obj.K
        self.Ki = getattr(K_obj, 'Ki', None)
        self._device = device

        self.speedup_info = None

    def to(self, device):
        self._device = device
        self.K_obj.to(device)
        return self

    def fit(self, X, Y, m, Yhats=None):
        self.X, self.Y, self.m = X, Y, m
        if isinstance(self.m, torch.nn.Linear):
            self.m.to(self._device)
        if Yhats is None:
            Yhats = [self.model(xi) for xi in tqdm.tqdm(X, desc='fit')]
        self.resids = np.asarray([abs(yhati- yi) for yhati, yi in zip(Yhats, Y)])
        print(f"MSE={np.mean([np.power(r,2) for r in self.resids])}")
        idx = np.argsort(self.resids)
        self.resids, self.X, self.Y = self.resids[idx], self.X[idx], self.Y[idx]
        self.resids = np.concatenate([self.resids, [np.infty]])
        #precompute speedup information if necessary
        if self.Ki is not None:
            _, self.speedup_info = self.Ki(self.X[0], self.X, self.speedup_info)


    def model(self, x):
        if isinstance(self.m, torch.nn.Linear):
            x = torch.tensor(x, device=self.m.weight.device, dtype=torch.float)
            return float(self.m(x))
        return self.m(x)

    def PI(self, x, alpha=0.05, x0=None, yhat=None):
        assert x0 is None
        if self.Ki is None:
            raise Exception("This can be slow")
            Kis = np.asarray([self.K(xi, x) for xi in self.X] + [self.K(x)])
        else:
            with torch.no_grad():
                Kis, self.speedup_info = self.Ki(x, self.X, self.speedup_info)
                Kis = torch.cat([Kis, torch.tensor([self.K(x)], dtype=torch.float, device=Kis.device)])
                Kis /= torch.sum(Kis)
        weights = Kis.cpu().detach().numpy()
        if len(self.resids) == len(self.Y) + 1:
            q = weighted_quantile_faster2(self.resids, weights, 1-alpha)
        else:
            assert len(self.resids) == len(self.Y)
            ss = np.concatenate([self.resids, [np.infty]])
            q = weighted_quantile_faster(ss, weights, 1 - alpha) #TODO: This is the slower version actually. Will delete this but some cached objects still have the old meomrization
        if yhat is None: yhat = self.model(x)
        lb, ub = yhat - q, yhat + q
        return lb, ub, yhat, None

class LocalConditionalMAD(LocalConditional):
    def __init__(self, K_obj=None, device='cuda', eps=1e-5):
        super(LocalConditionalMAD, self).__init__(K_obj, device)
        self.eps = eps

    def fit(self, X, Y, m, mresid, Yhats=None, rhats=None):
        self.X, self.Y, self.m, self.m_resid = X, Y, m, mresid
        if isinstance(self.m, torch.nn.Linear): self.m.to(self._device)
        if isinstance(self.m_resid, torch.nn.Linear): self.m_resid.to(self._device)

        if Yhats is not None:
            for _i in range(3): assert np.isclose(self.model(X[0]), Yhats[0]), "sanity checks..."
        else:
            Yhats = [self.model(xi) for xi in tqdm.tqdm(X, desc='fit mean')]

        if rhats is None:
            rhats = [self.model_resid(xi) for xi in tqdm.tqdm(X, desc='fit residual')]

        self.resids = np.asarray([abs(yhati- yi) for yhati, yi in zip(Yhats, Y)])
        print(f"MSE={np.mean([np.power(r, 2) for r in self.resids])}")
        self.resids_normalized = self.resids / (self.eps + np.abs(np.asarray(rhats)))
        print(f"scores={np.mean([np.power(r,2) for r in self.resids_normalized])}")

        idx = np.argsort(self.resids_normalized)
        self.resids, self.resids_normalized, self.X, self.Y = self.resids[idx], self.resids_normalized[idx], self.X[idx], self.Y[idx]
        self.resids_normalized = np.concatenate([self.resids_normalized, [np.infty]])
        #precompute speedup information if necessary

        if self.Ki is not None:
            _, self.speedup_info = self.Ki(self.X[0], self.X, self.speedup_info)

    def model_resid(self, x):
        if isinstance(self.m_resid, torch.nn.Linear):
            x = torch.tensor(x, device=self.m_resid.weight.device, dtype=torch.float)
            return float(self.m_resid(x))
        return self.m_resid(x)

    def PI(self, x, alpha=0.05, x0=None, yhat=None, rhat=None):
        assert x0 is None
        with torch.no_grad():
            Kis, self.speedup_info = self.Ki(x, self.X, self.speedup_info)
            Kis = Kis.cpu().detach().tolist() + [self.K(x)]
            Kis = np.asarray(Kis)
        weights = Kis / np.sum(Kis)
        if len(self.resids_normalized) == len(self.Y) + 1:
            q = weighted_quantile_faster2(self.resids_normalized, weights, 1-alpha)
        else:
            assert len(self.resids_normalized) == len(self.Y)
            ss = np.concatenate([self.resids_normalized, [np.infty]])
            q = weighted_quantile_faster(ss, weights, 1 - alpha)
        if yhat is None: yhat = self.model(x)
        if rhat is None: rhat = self.model_resid(x)
        rhat = np.abs(rhat) + self.eps
        lb, ub = yhat - q * rhat, yhat + q * rhat
        return lb, ub, yhat, None

if __name__ == "__main__":
    pass