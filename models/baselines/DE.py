import _settings
import data.preprocess_qm_datasets as qmprep
import pandas as pd
import numpy as np
import ipdb
import scipy.stats as st
import os
import utils.utils as utils

def get_DE_result(dataset=_settings.QM8_NAME, which_y=0, seed=0, alpha=0.1):
    assert dataset in {_settings.QM8_NAME, _settings.QM9_NAME}
    z_critical = st.norm.ppf(1-0.5*alpha)

    dfs = {}
    mus = []
    sigma2s = []
    y = None
    for model_id in range(5):
        to = qmprep.QMSplitter(dataset, split_ratio=[60, 20, 20], seed=seed, baseline='DE', baseline_kwargs={"seed": model_id})
        tdf = pd.read_csv(to.get_preds_path(qmprep.TEST))
        ntasks = int((len(tdf.columns) - 1) / 2)
        for i in range(ntasks):
            assert "%s_sigma2"%tdf.columns[i+1] == tdf.columns[i + ntasks+1]
        dfs[model_id] = tdf
        if model_id - 1 in dfs:
            assert dfs[model_id]['smiles'].equals(dfs[model_id - 1]['smiles'])
        mus.append(tdf.iloc[:, 1 + which_y].values)
        sigma2s.append(tdf.iloc[:, 1 + which_y + ntasks].values)
        if y is None: y = pd.read_csv(to.get_data_path(qmprep.TEST))[dfs[model_id].columns[1+which_y]]
    mus = np.asarray(mus)
    sigma2s = np.abs(np.array(sigma2s)) + 1e-5  # (M, B)
    yhat = np.mean(mus, 0)
    y_std = np.sqrt(np.mean(sigma2s + np.square(mus), axis=0) - np.square(yhat))

    y_l, y_u = yhat - z_critical * y_std, yhat + z_critical * y_std
    df = pd.DataFrame({"lo": y_l, 'hi': y_u, 'y': y, 'yhat': yhat, 'index': dfs[0]['smiles']})
    return df

def get_DE_result_cached(dataset=_settings.QM8_NAME, which_y=0, seed=0, alpha=0.1):
    cache_dir = os.path.join(_settings.WORKSPACE, 'manual_cache', dataset, 'baselines', 'DE')
    if not os.path.isdir(cache_dir): os.makedirs(cache_dir)
    cache_path = os.path.join(cache_dir, f"{which_y}_seed{seed}_alpha{alpha}.pkl")
    if not os.path.isfile(cache_path):
        pd.to_pickle(get_DE_result(dataset, which_y, seed, alpha), cache_path)
    return pd.read_pickle(cache_path)

