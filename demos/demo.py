import utils.utils as utils
import _settings
import models.regmodel as regmodel
import models.conformal as conformal
import demos.regression as reg
import data.dataloader as dld
import numpy as np
import pandas as pd
import six
import os
import ipdb



def eval_Split(dataset=_settings.QM8_NAME, seed=_settings.RANDOM_SEED, alpha=0.05, **kwargs):
    import data.dataloader as dld
    val_data = dld.get_default_dataset(dataset, split=dld.VALID, seed=seed)
    resids = np.asarray(val_data.get_all_Y() - val_data._get_all_Yhat())
    _w = np.quantile(np.abs(resids), 1-alpha)
    test_data = dld.get_default_dataset(dataset, split=dld.TEST, seed=seed)
    test_preds = test_data._get_all_Yhat()
    test_labels = test_data.get_all_Y()
    res = []
    for i, yhat in enumerate(test_preds):
        res.append({"lo": yhat - _w, 'hi': yhat+_w, 'y': test_labels[i], 'yhat': yhat})
    return pd.DataFrame(res)


def sep_datakwargs(datakwargs={}):
    assert 'split' not in datakwargs
    base_datakwargs = {k:v for k,v in datakwargs.items() if not k.endswith("split")}
    train_datakwargs = utils.merge_dict_inline(base_datakwargs, {'split': datakwargs.get('train_split', dld.TRAIN)})
    val_datakwargs = utils.merge_dict_inline(base_datakwargs, {'split': datakwargs.get('val_split', dld.VALID)})
    test_datakwargs = utils.merge_dict_inline(base_datakwargs, {'split': datakwargs.get('test_split', dld.TEST)})
    return train_datakwargs, val_datakwargs, test_datakwargs

def _fit_conformal(dataset=_settings.QM8_NAME, datakwargs={},
                   algo_name=regmodel._KERNEL_MLKR, fitkwargs={},
                   PI_model='LocalConformal', PIkwargs={}, gpuid=None):
    train_datakwargs, val_datakwargs, test_datakwargs = sep_datakwargs(datakwargs)
    m = reg.get_trained_model_cached(dataset, algo_name, datakwargs=train_datakwargs.copy(), gpuid=gpuid, **fitkwargs)
    if algo_name in {regmodel._KERNEL_MLKR}: m = m.to(utils.gpuid_to_device(gpuid))

    dataset_instance = dld.get_default_dataset(dataset, **val_datakwargs)
    X, Y = dataset_instance.get_all_X(), dataset_instance.get_all_Y()
    if PI_model == conformal._LocalConformal:
        kernels = {"K_obj": {'trained': m, 'naive': None}[PIkwargs.get('kernel', 'trained')]}

        which_model = PIkwargs.get('pred', 'KR')
        if which_model == 'KR':
            Yhats, _ = reg.eval_trained_model_cached(dataset, val_datakwargs, algo_name, train_datakwargs, gpuid, **fitkwargs)
            models = {'m': m.predict, 'Yhats': Yhats}
        elif which_model == 'base':
            models = {'m': dataset_instance.model, 'Yhats': dataset_instance._get_all_Yhat()}
        else:
            raise NotImplementedError()
        o = conformal.LocalConditional(**kernels)
        o.fit(X, Y, **models)
        return o

    if PI_model == conformal._LocalConformalMAD:
        kernels = {"K_obj": {'trained': m, 'naive': None}[PIkwargs.get('kernel', 'trained')]}

        which_model = PIkwargs.get('pred', 'KR')
        if which_model == 'KR':
            Yhats, _ = reg.eval_trained_model_cached(dataset, val_datakwargs, algo_name, train_datakwargs, gpuid, **fitkwargs)
            models = {'m': m.predict, 'Yhats': Yhats}
        elif which_model == 'base':
            models = {'m': dataset_instance.model, 'Yhats': dataset_instance._get_all_Yhat()}
        else:
            raise NotImplementedError()
        models.update({'rhats': dataset_instance._get_all_rhat(),
                       'mresid': None})
        o = conformal.LocalConditionalMAD(**kernels)
        o.fit(X, Y, **models)
        return o

def _eval_exp(dataset=_settings.QM8_NAME, datakwargs={},
              algo_name=regmodel._KERNEL_MLKR, fitkwargs={},
              PI_model='LocalConformal', PIkwargs={},
              alpha=0.1, gpuid=None):
    o = fit_conformal_cached(dataset, datakwargs.copy(), algo_name, fitkwargs.copy(), PI_model, PIkwargs.copy(), gpuid=gpuid)
    if PI_model in {conformal._LocalConformal}: o = o.to(utils.gpuid_to_device(gpuid))

    train_datakwargs, val_datakwargs, test_datakwargs = sep_datakwargs(datakwargs)
    dataset_instance = dld.get_default_dataset(dataset, **test_datakwargs)
    X, Y = dataset_instance.get_all_X(), dataset_instance.get_all_Y()
    PI_kwargs, PI_list_kwargs = {}, {}
    if PI_model == conformal._LocalConformal:
        base_yhat = PIkwargs.get('pred', 'KR')
        if base_yhat == 'KR':
            PI_list_kwargs['yhat'] = reg.eval_trained_model_cached(dataset, test_datakwargs.copy(), algo_name, train_datakwargs, gpuid, **fitkwargs)[0]
        elif base_yhat == 'base':
            PI_list_kwargs['yhat'] = dataset_instance._get_all_Yhat()
    if PI_model == conformal._LocalConformalMAD:
        base_yhat = PIkwargs.get('pred', 'KR')
        if base_yhat == 'KR':
            PI_list_kwargs['yhat'] = reg.eval_trained_model_cached(dataset, test_datakwargs.copy(), algo_name, train_datakwargs, gpuid, **fitkwargs)[0]
        elif base_yhat == 'base':
            PI_list_kwargs['yhat'] = dataset_instance._get_all_Yhat()
        PI_list_kwargs['rhat'] = dataset_instance._get_all_rhat()
    res = o.eval(X, Y, o.PI, alpha, quiet=False, PI_kwargs=PI_kwargs, PI_list_kwargs=PI_list_kwargs)
    res['index'] = dataset_instance.index
    return res

def fit_conformal(dataset=_settings.QM8_NAME, datakwargs={},
                  algo_name=regmodel._KERNEL_MLKR, fitkwargs={},
                  PI_model='LocalConformal', PIkwargs={}, gpuid=None):
    return _fit_conformal(dataset, datakwargs, algo_name, fitkwargs, PI_model, PIkwargs, gpuid)

def fit_conformal_cached(dataset=_settings.QM8_NAME, datakwargs={},
                         algo_name=regmodel._KERNEL_MLKR, fitkwargs={},
                         PI_model='LocalConformal', PIkwargs={}, gpuid=None):
    cache_dir = os.path.join(_settings.WORKSPACE, 'manual_cache', dataset, 'fit_conformal_cached', PI_model, algo_name)
    if not os.path.isdir(cache_dir): os.makedirs(cache_dir)
    full_kwargs = {}
    for k, v in datakwargs.items(): full_kwargs['datakwargs|%s' % k] = v
    for k, v in fitkwargs.items(): full_kwargs['fitkwargs|%s' % k] = v
    for k, v in PIkwargs.items(): full_kwargs['PIkwargs|%s' % k] = v
    key = tuple(sorted(six.iteritems(full_kwargs), key=lambda x: x[0]))
    hashed_path = os.path.join(cache_dir, f"{utils._hash(key)}.pkl")

    if not os.path.isfile(hashed_path): pd.to_pickle({}, hashed_path)
    res = pd.read_pickle(hashed_path)
    if key in res: return res[key]

    val = fit_conformal(dataset, datakwargs, algo_name, fitkwargs, PI_model, PIkwargs, gpuid)

    res = pd.read_pickle(hashed_path)
    res[key] = val
    pd.to_pickle(res, hashed_path)
    return val


def eval_exp(dataset=_settings.QM8_NAME, datakwargs={},
             algo_name=regmodel._KERNEL_MLKR, fitkwargs={},
             PI_model='LocalConformal', PIkwargs={'pred': 'base', 'kernel': 'trained'},
             alpha=0.1, gpuid=None):
    return _eval_exp(dataset, datakwargs, algo_name, fitkwargs, PI_model, PIkwargs, alpha, gpuid)

def eval_exp_cached(dataset=_settings.QM8_NAME, datakwargs={},
                    algo_name=regmodel._KERNEL_MLKR, fitkwargs={},
                    PI_model='LocalConformal', PIkwargs={'pred': 'base', 'kernel': 'trained'},
                    alpha=0.1, gpuid=None):
    cache_dir = os.path.join(_settings.WORKSPACE, 'manual_cache', dataset, 'eval_exp_cached', PI_model, algo_name)
    if not os.path.isdir(cache_dir): os.makedirs(cache_dir)
    full_kwargs = {}
    for k, v in datakwargs.items(): full_kwargs['datakwargs|%s' % k] = v
    for k, v in fitkwargs.items(): full_kwargs['fitkwargs|%s' % k] = v
    for k, v in PIkwargs.items(): full_kwargs['PIkwargs|%s' % k] = v
    key = tuple(sorted(six.iteritems(full_kwargs), key=lambda x: x[0]))
    hashed_path = os.path.join(cache_dir, f"{alpha}_{utils._hash(key)}.pkl")

    if not os.path.isfile(hashed_path): pd.to_pickle({}, hashed_path)
    res = pd.read_pickle(hashed_path)
    if key in res: return res[key]

    val = eval_exp(dataset, datakwargs, algo_name, fitkwargs, PI_model, PIkwargs, alpha, gpuid)

    res = pd.read_pickle(hashed_path)
    res[key] = val
    pd.to_pickle(res, hashed_path)
    return val

def summ_dataset(dataset=_settings.YACHT_NAME, datakwargs={}):
    train_datakwargs, val_datakwargs, test_datakwargs = sep_datakwargs(datakwargs)
    dataset_instance = dld.get_default_dataset(dataset, **test_datakwargs)
    x = dataset_instance.get_all_X()
    y = dataset_instance.get_all_Y()
    print("X:")
    print(f"{x.shape[1]} features")
    y_mean = np.mean(y)
    y_var = np.mean(np.power(y-y_mean,2))
    print("\nY:")
    print(f"Mean={y_mean}")
    print(f"Var={y_var}")


def read_QM_data_for_baselines(method, split='test', which_y=0, dataset=_settings.QM8_NAME, seed=0, alpha=0.1):
    import data.preprocess_qm_datasets as qmprep
    if method == 'LocalMAD':
        o = qmprep.QMSplitter(dataset, split_ratio=[60, 20, 20], seed=seed, baseline='MADSplit')
        datadf = pd.read_csv(o.base_obj.get_data_path(split))
        preddf = pd.read_csv(o.base_obj.get_preds_path(split))
        madhatdf = pd.read_csv(o.get_preds_path(split))
        assert datadf['smiles'].eq(preddf['smiles']).all() and madhatdf['smiles'].eq(datadf['smiles']).all()
        y_col = datadf.columns[1 + which_y]
        new_df = pd.DataFrame({"index": datadf['smiles'], 'y': datadf[y_col], 'yhat':preddf[y_col], 'rhat': madhatdf[y_col]})
        return new_df
    if method == 'CQR':
        o = qmprep.QMSplitter(dataset, split_ratio=[60, 20, 20], seed=seed, baseline='CQR', baseline_kwargs={"alpha": alpha})
        datadf = pd.read_csv(o.get_data_path(split))
        preddf = pd.read_csv(o.get_preds_path(split))
        assert datadf['smiles'].eq(preddf['smiles']).all()
        y_col = datadf.columns[1 + which_y]
        new_df = pd.DataFrame({"index": datadf['smiles'], 'y': datadf[y_col], 'yhat': preddf[y_col],
                               'yhat_lo': preddf[y_col+"_lo"], 'yhat_hi': preddf[y_col+"_hi"]})
        return new_df
    raise NotImplementedError()

def conformal_baselines(method='MADSplit', dataset=_settings.KIN8NM_NAME, datakwargs={},
                        model_setting=0, alpha=0.1, quiet=True, **fitkwargs):
    import models.baselines.MADSplit as MADSplit
    import models.baselines.CQR as CQR
    if dataset in {_settings.QM8_NAME, _settings.QM9_NAME}:
        assert 'which_y' in datakwargs and 'seed' in datakwargs
        assert model_setting == 0 and len(fitkwargs) == 0 and quiet and alpha in {0.1, 0.5}
        val_df = read_QM_data_for_baselines(method, 'val', dataset=dataset, alpha=alpha, **datakwargs)
        test_df = read_QM_data_for_baselines(method, 'test', dataset=dataset, alpha=alpha, **datakwargs)
        if method == 'MADSplit':
            los, his, yhats = MADSplit.MADSplit_from_results(val_df['y'], val_df['yhat'], val_df['rhat'],
                                                             test_df['yhat'], test_df['rhat'], alpha=alpha)
        elif method == 'CQR':
            los, his, yhats = CQR.CQR_from_results(val_df['y'], val_df['yhat_lo'], val_df['yhat_hi'],
                                                   test_df['yhat_lo'], test_df['yhat_hi'], alpha=alpha)
        return pd.DataFrame({"lo": los, "hi": his, "y": test_df['y'], "yhat": yhats, 'index': test_df['index']})
    assert model_setting == datakwargs.get('model_setting', model_setting)
    datakwargs['model_setting'] = model_setting
    seed = datakwargs['seed']
    train_datakwargs, val_datakwargs, test_datakwargs = sep_datakwargs(datakwargs)
    train_data = dld.get_default_dataset(dataset, **train_datakwargs)
    val_data = dld.get_default_dataset(dataset, **val_datakwargs)
    test_data = dld.get_default_dataset(dataset, **test_datakwargs)

    train_X, train_Y = train_data.raw_data, train_data.get_all_Y()
    val_X, val_Y = val_data.raw_data, val_data.get_all_Y()
    test_X, test_Y = test_data.raw_data, test_data.get_all_Y()
    if method == 'MADSplit':
        rhat_val, yhat_val = val_data._get_all_rhat(), val_data._get_all_Yhat()
        rhat_test, yhat_test = test_data._get_all_rhat(), test_data._get_all_Yhat()
        los, his, yhats = MADSplit.MADSplit_from_results(val_Y, yhat_val, rhat_val, yhat_test, rhat_test, alpha=alpha)
        return pd.DataFrame({"lo": los, "hi": his, "y": test_Y, "yhat": yhats, 'index': test_data.index})
    elif method == 'CQR':
        o = CQR.CQR(alpha=alpha, model_class=model_setting, seed=seed)
        o.fit(val_X, val_Y, train_X, train_Y, quiet=quiet)
    else:
        raise NotImplementedError()
    df = conformal.PIConstructor.eval(test_X, test_Y, o.PI, alpha=alpha, quiet=quiet)
    df['index'] = test_data.index
    return df

def conformal_baselines_cached(method='MADSplit', dataset=_settings.KIN8NM_NAME, datakwargs={},
                               model_setting=0, alpha=0.1, quiet=True, **fitkwargs):
    cache_dir = os.path.join(_settings.WORKSPACE, 'manual_cache', dataset, 'conformal_baselines_cached', method)
    if not os.path.isdir(cache_dir): os.makedirs(cache_dir)
    full_kwargs = {}
    for k, v in datakwargs.items(): full_kwargs['datakwargs|%s' % k] = v
    for k, v in fitkwargs.items(): full_kwargs['fitkwargs|%s' % k] = v
    key = tuple(sorted(six.iteritems(full_kwargs), key=lambda x: x[0]))
    hashed_path = os.path.join(cache_dir, f"{alpha}_{model_setting}_{utils._hash(key)}.pkl")

    if not os.path.isfile(hashed_path): pd.to_pickle({}, hashed_path)
    res = pd.read_pickle(hashed_path)
    if key in res: return res[key]

    val = conformal_baselines(method, dataset, datakwargs, model_setting, alpha, quiet, **fitkwargs) #Actually run it

    res = pd.read_pickle(hashed_path)
    res[key] = val
    pd.to_pickle(res, hashed_path)
    return val



if __name__ == '__main__':
    #gpuid = 1
    #datakwargs = {'which_y': 0, 'seed': 7}
    #fitkwargs = {'d': 100, 'n_iters': 2, 'max_n': 1000, 'batch_size': 50, 'seed': 7}
    #m = regmodel.get_trained_model(_settings.QM8_NAME, regmodel._KERNEL_MLKR, datakwargs=datakwargs.copy(), **fitkwargs)
    #o = eval_exp(_settings.QM8_NAME, datakwargs.copy(), regmodel._KERNEL_MLKR, fitkwargs, gpuid=gpuid)
    #df = read_QM_data_for_baselines("CQR")
    #df = conformal_baselines('MADSplit', dataset=_settings.WINERED_NAME, datakwargs={"seed": 0})
    #df2 = conformal_baselines('CQR', dataset=_settings.QM8_NAME, datakwargs={"which_y": 0, "seed": 0})
    pass