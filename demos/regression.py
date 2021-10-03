import torch
import numpy as np
import tqdm
import pandas as pd
import utils.utils as utils
import _settings
import torch.nn as nn
import ipdb
import models.regmodel as regmodel
import data.dataloader as dld
import six
import os

#=======================================================================================================================

def _get_trained_model(dataset=_settings.QM8_NAME, algo_name=regmodel._KERNEL_MLKR, datakwargs={}, gpuid=None, **fitkwargs):
    datakwargs = datakwargs.copy()
    assert 'split' in datakwargs
    datakwargs.setdefault('seed', _settings.RANDOM_SEED)
    dataset_instance = dld.get_default_dataset(dataset, **datakwargs)
    base_model = regmodel.get_base_model(algo_name)
    if algo_name == regmodel._KERNEL_MLKR:
        fitkwargs.setdefault('device', utils.gpuid_to_device(gpuid))
    model = base_model(**fitkwargs)
    model.fit(dataset_instance.get_all_X(), dataset_instance.get_all_Y())
    return model

def get_trained_model(dataset=_settings.QM8_NAME, algo_name=regmodel._KERNEL_MLKR, datakwargs={}, gpuid=None, **fitkwargs):
    return _get_trained_model(dataset, algo_name, datakwargs, gpuid, **fitkwargs)

def get_trained_model_cached(dataset=_settings.QM8_NAME, algo_name=regmodel._KERNEL_MLKR, datakwargs={}, gpuid=None, **fitkwargs):
    cache_dir = os.path.join(_settings.WORKSPACE, 'manual_cache', dataset, 'get_trained_model_cached', algo_name)
    if not os.path.isdir(cache_dir): os.makedirs(cache_dir)
    full_kwargs = {}
    for k, v in datakwargs.items(): full_kwargs['datakwargs|%s' % k] = v
    for k, v in fitkwargs.items(): full_kwargs['fitkwargs|%s' % k] = v
    key = tuple(sorted(six.iteritems(full_kwargs), key=lambda x: x[0]))
    hashed_path = os.path.join(cache_dir, f"{utils._hash(key)}.pkl")
    #I won't use any file lock here because the collision probability is quite low compared to my original caching mechanism.
    if not os.path.isfile(hashed_path):
        pd.to_pickle({}, hashed_path)
    res = pd.read_pickle(hashed_path)
    if key in res: return res[key]

    val = get_trained_model(dataset, algo_name, datakwargs, gpuid, **fitkwargs) #Actually run it

    res = pd.read_pickle(hashed_path)
    res[key] = val
    pd.to_pickle(res, hashed_path)
    return val


def eval_trained_model(dataset=_settings.QM8_NAME, eval_datakwargs={},
                       algo_name=regmodel._KERNEL_MLKR, datakwargs={},
                       gpuid=None, **fitkwargs):
    model = get_trained_model_cached(dataset, algo_name, datakwargs, gpuid, **fitkwargs)
    model = model.to(utils.gpuid_to_device(gpuid))
    dataset_instance = dld.get_default_dataset(dataset, **eval_datakwargs)
    yhats = []
    extras = []
    for xi, yi, _ in tqdm.tqdm(dataset_instance, desc='eval'):
        yhati, extra = model.predict_with_info(xi)
        if len(extra) > 1000: extra = None
        yhats.append(yhati)
        extras.append(extra)
    return np.asarray(yhats), extras


def eval_trained_model_cached(dataset=_settings.QM8_NAME, eval_datakwargs={},
                              algo_name=regmodel._KERNEL_MLKR, datakwargs={},
                              gpuid=None, **fitkwargs):
    cache_dir = os.path.join(_settings.WORKSPACE, 'manual_cache', dataset, 'eval_trained_model_cached', algo_name)
    if not os.path.isdir(cache_dir): os.makedirs(cache_dir)
    full_kwargs = {}
    for k, v in eval_datakwargs.items(): full_kwargs['eval_datakwargs|%s' % k] = v
    for k, v in datakwargs.items(): full_kwargs['datakwargs|%s' % k] = v
    for k, v in fitkwargs.items(): full_kwargs['fitkwargs|%s' % k] = v
    key = tuple(sorted(six.iteritems(full_kwargs), key=lambda x: x[0]))
    hashed_path = os.path.join(cache_dir, f"{utils._hash(key)}.pkl")

    if not os.path.isfile(hashed_path): pd.to_pickle({}, hashed_path)
    res = pd.read_pickle(hashed_path)
    if key in res: return res[key]

    val = eval_trained_model(dataset, eval_datakwargs, algo_name, datakwargs, gpuid, **fitkwargs) #Actually run it

    res = pd.read_pickle(hashed_path)
    res[key] = val
    pd.to_pickle(res, hashed_path)
    return val