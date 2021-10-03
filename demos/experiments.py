def get_default_fitkwargs(dataset=None):
    return {'d': 10, 'n_iters': 1000, 'max_n': 3000, 'batch_size': 100, 'lr': 1e-2, 'stop_iters':50, 'norm': True, 'ybar_bias': True}

#=====================================For things implemented in DJKP
import pandas as pd
import numpy as np
import os
import ipdb
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston, load_diabetes, fetch_california_housing
import torch

from models.baselines.djkp.dj_models.static import *
from models.baselines.djkp.dj_models.PBP import *

import _settings #from main repo

TRAIN = 'train'
VALID = 'val'
TEST = 'test'
VALID_1 = 'val1'
VALID_2 = 'val2'

SEED_OFFSETS = {TRAIN: 101, VALID: 202, VALID_1: 203, VALID_2: 204, TEST: 303}

def my_split(idx, split_ratio=[0.8, 0.2], seed=10):
    assert len(split_ratio) == 2, "for this task"
    np.random.seed(seed)
    n = len(idx)
    perm = np.random.permutation(n)
    split_ratio = np.concatenate([[0.], np.cumsum(split_ratio) / sum(split_ratio)])
    splits = np.round(split_ratio * n).astype(np.int)
    idxs = [idx[perm[splits[i]:splits[i + 1]]] for i in range(len(split_ratio) - 1)]
    return idxs[0], idxs[1]

def load_dataset(dataset='UCI_Yacht', seed=7, data_path=_settings.DATA_PATH):
    if dataset == _settings.HOUSING_NAME:
        X, y = load_boston(return_X_y=True)
    elif dataset == _settings.ENERGY_NAME:
        fpath = os.path.join(data_path, dataset, 'ENB2012_data.xlsx')
        raw_df = pd.read_excel(fpath, engine='openpyxl')
        # raw_df = raw_df.iloc[:, :10]
        # raw_df.columns = ["X%d" % d for d in range(self.raw_df.shape[1] - 2)] + ['Y0', 'Y1']
        raw_df = raw_df.iloc[:, :9]
        raw_df.columns = ["X%d" % d for d in range(raw_df.shape[1] - 1)] + ['Y']
        X = raw_df.iloc[:, :-1].values
        y = raw_df.iloc[:, -1].values
    elif dataset == _settings.YACHT_NAME:
        fpath = os.path.join(data_path, 'UCI_Yacht', 'yacht_hydrodynamics.data')
        df = pd.read_fwf(fpath, header=None)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    elif dataset == _settings.KIN8NM_NAME:
        fpath = os.path.join(data_path, 'Kin8nm', 'dataset_2175_kin8nm.csv')
        df = pd.read_csv(fpath)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    elif dataset == _settings.BIKE_NAME:
        fpath = os.path.join(data_path, 'UCI_BikeSharing', 'Bike-Sharing-Dataset.zip')
        from zipfile import ZipFile
        archive = ZipFile(fpath)
        #raw_df = pd.read_csv(archive.open('day.csv')).set_index('dteday', verify_integrity=True).drop('instant',axis=1)
        raw_df = pd.read_csv(archive.open('hour.csv')).set_index('instant', verify_integrity=True)
        drop_cols = ['yr', 'mnth', 'dteday']
        enum_cols = ['season', 'hr', 'weekday', 'weathersit']
        raw_df = raw_df.drop(drop_cols, axis=1)
        for enum_col in enum_cols:
            ser = raw_df[enum_col]
            tdf = pd.get_dummies(ser).rename(columns=lambda x: "%s%d"%(enum_col, x))
            raw_df = pd.concat([raw_df, tdf], axis=1).drop(enum_col, axis=1)
        raw_df.index.name = 'index'
        raw_df = raw_df.reindex(columns=[c for c in raw_df.columns] + ['cnt'])
        #ipdb.set_trace()
        raw_df.columns = ["X%d" % d for d in range(raw_df.shape[1] - 1)] + ['Y']
        X = raw_df.iloc[:, :-1].values
        y = raw_df.iloc[:, -1].values
    elif dataset == _settings.CONCRETE_NAME:
        fpath = os.path.join(data_path, 'UCI_Concrete', 'Concrete_Data.xls')
        raw_df = pd.read_excel(fpath)
        raw_df.columns = ["X%d" % d for d in range(raw_df.shape[1] - 1)] + ['Y']
        raw_df = raw_df.reset_index()
        X = raw_df.iloc[:, :-1].values
        y = raw_df.iloc[:, -1].values
    elif dataset.startswith('DJKPSynthetic'):
        dist = dataset.split("_")[1]
        xs, ys = [], []
        param, sigma, n = 1, 4, 100
        for split in [TRAIN, VALID, TEST]:
            np.random.seed(seed + SEED_OFFSETS[split])
            if dist == 'normal+':
                x1 = np.random.normal(param, param, n)
                x1 = np.max(np.stack([x1-param, param-x1], 1),1) + param #half normal
                x2 = np.random.uniform(-param, param, n)
                eps1 = np.random.normal(0, sigma, n)
                eps2 = np.random.normal(0, sigma, n)
                y1 = np.power(x1, 3) + eps1
                y2 = np.power(x2, 3) + eps2
                x,y,eps = [],[],[]
                us = np.random.uniform(0, 1, n)
                for i in range(n):
                    if us[i] < 0.1:
                        x.append(x1[i])
                        y.append(y1[i])
                        eps.append(eps1[i])
                    else:
                        x.append(x2[i])
                        y.append(y2[i])
                        eps.append(eps2[i])
                x = np.asarray(x)
                y = np.asarray(y)
            else:
                x = np.random.normal(0, param, n)
                eps = np.random.normal(0, sigma, n)
                y = np.power(x, 3) + eps
            xs.append(x)
            ys.append(y)
        X, y = np.concatenate(xs), np.concatenate(ys)
        X = np.expand_dims(X, 1)
        train_idx, test_idx = np.arange(2*n), np.arange(2*n,3*n)
        X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
        return X_train, X_test, y_train, y_test, train_idx, test_idx
    train_idx, test_idx = my_split(np.arange(len(X)), split_ratio=[0.8, 0.2], seed=seed)
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test, train_idx, test_idx


def run_baseline(X_train, y_train, X_test, baseline="DJ", damp=1e-4, mode='exact', coverage=.9,
                 params={}, train_params={}, seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    z_critical = st.norm.ppf(0.5+0.5*coverage)

    if baseline == "DJ":
        model = DNN(**params)
        model.fit(X_train, y_train, **train_params)
        DNN_posthoc = DNN_uncertainty_wrapper(model, damp=damp, mode=mode)
        y_pred, y_l, y_u = DNN_posthoc.predict(X_test, coverage=coverage)

    elif baseline == "MCDP":

        model = MCDP_DNN(**params)
        model.fit(X_train, y_train, **train_params)

        y_MCDP = model.predict(X_test, alpha=1 - coverage)
        y_pred, y_l, y_u = y_MCDP[0], y_MCDP[0] - y_MCDP[1], y_MCDP[0] + y_MCDP[1]

    elif baseline == "PBP":

        pbp_model = Bayes_backprop(input_dim=X_train.shape[1])
        pbp_model.fit(X_train, y_train.reshape((-1, 1)))

        y_pred, y_u, y_l = pbp_model.predict(X_test, alpha=1 - coverage)

    elif baseline == "DE_incorrect":
        raise Exception("This is an incorrect implementation. Please use DE_Correct")
        y_pred, y_std = Deep_ensemble(X_train, y_train, X_test, params, n_ensemble=5, train_frac=0.8)
        y_l, y_u = y_pred - z_critical * y_std, y_pred + z_critical * y_std
    elif baseline == "DE" or baseline == 'DE_Correct':
        y_pred, y_std = Deep_ensemble_correct(X_train, y_train, X_test, params, n_ensemble=5, train_frac=0.8)
        y_l, y_u = y_pred - z_critical * y_std, y_pred + z_critical * y_std
    return y_pred, y_l, y_u


def run_experiments(baselines, datasets, N_exp=10, damp=1e-4, mode='exact', coverage=.9, params={},
                    train_params={},
                    data_path=_settings.DATA_PATH, cache_path=None, quiet=False):
    results = dict.fromkeys(baselines)
    for baseline in baselines:
        results[baseline] = dict.fromkeys(datasets)
        for dataset in datasets:
            results[baseline][dataset] = dict()#dict({"AUPRC": [], "Coverage": [], "MSE": []})

    for dataset in datasets:
        if not quiet: print("Running experiments on dataset: ", dataset)
        for exp_seed in (range(N_exp) if isinstance(N_exp, int) else N_exp):
            if not quiet: print("Exp: %d"%exp_seed)

            for baseline in baselines:
                if cache_path is not None:
                    cache_key = 'seed%d'%exp_seed
                    cache_key += '_act%s'%params.get('activation', 'Tanh')
                    cache_key += '_nh%d'%params.get('num_hidden', 100)
                    cache_key += '_nl%d' % params.get('num_layers', 2)
                    cache_key += '_niter{}'.format(train_params.get('num_iter', 'default'))
                    cache_key += '_lr{}'.format(train_params.get('learning_rate', 'default'))
                    cache_key += '_cov{:.3f}'.format(coverage)
                    if baseline == 'DJ':
                        cache_key += '_damp%f'%damp
                        cache_key += '_mode%s'%mode
                    curr_cache_path = os.path.join(cache_path, dataset, baseline, '%s.pkl'%cache_key)
                if cache_path is not None and os.path.isfile(curr_cache_path):
                    df = pd.read_pickle(curr_cache_path)
                else:
                    X_train, X_test, y_train, y_test, train_idx, test_idx = load_dataset(dataset, seed=exp_seed, data_path=data_path)
                    y_pred, y_l, y_u = run_baseline(X_train, y_train, X_test, baseline=baseline,
                                                    damp=damp, mode=mode,
                                                    coverage=coverage, params=params, train_params=train_params,
                                                    seed=exp_seed)
                    df = pd.DataFrame({"lo": y_l, 'hi': y_u, 'y': y_test, 'yhat': y_pred, 'index': test_idx})
                    if X_test.shape[1] == 1: df['x'] = X_test
                    if cache_path is not None:
                        if not os.path.isdir(os.path.dirname(curr_cache_path)):
                            os.makedirs(os.path.dirname(curr_cache_path))
                        pd.to_pickle(df, curr_cache_path)
                results[baseline][dataset][exp_seed] = df
    return results


def cache(data_path=None, cache_path=None):
    from utils.utils import TaskPartitioner
    if data_path is None:
        data_path = _settings.DATA_PATH
    if cache_path is None:
        cache_path = os.path.join(_settings.WORKSPACE, 'Baselines')
    baselines = ["DJ", "MCDP", "PBP", "DE"]
    datasets = [_settings.BIKE_NAME, _settings.KIN8NM_NAME, _settings.YACHT_NAME,
                _settings.ENERGY_NAME, _settings.HOUSING_NAME, _settings.CONCRETE_NAME]
    params = dict({"activation": 'ReLU', "num_hidden": 100, "num_layers": 1})
    train_params = dict({"num_iter": 1000, "learning_rate": 1e-3})

    task_runner = TaskPartitioner()
    for baseline in baselines:
        for dataset in datasets:
            for seed in range(10):
                for alpha in [0.1, 0.5]:
                    task_runner.add_task(run_experiments, [baseline], [dataset], [seed], damp=1e-2,
                                         mode='exact', coverage=1-alpha,
                                         params=params, train_params=train_params,
                                         data_path=data_path, cache_path=cache_path)

    task_runner.run_multi_process(1)

if __name__ == '__main__':
    pass