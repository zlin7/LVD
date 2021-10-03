import models.baselines.DNN as DNN
import _settings
import numpy as np
import torch
import pandas as pd
import os
import utils.utils as utils
import datetime
import shutil
import tqdm
import ipdb
import data.dataloader as dld

TRAIN, VALID, TEST = 'train', 'val', 'test'

def pretrain_general(X, Y, seed=_settings.RANDOM_SEED, model_setting = 0, quiet=False):
    utils.set_all_seeds(seed)
    n_dim = X.shape[1]
    output_size = Y.shape[1]
    model_class, model_kwargs, train_kwargs = DNN.get_DNN_and_trainkwargs(model_setting)
    model = model_class(n_dim=n_dim, output_size=output_size, **model_kwargs)
    model.fit(X, Y, verbosity=not quiet, **train_kwargs)
    readout_layer = model.model[-1]
    model = model.eval()
    readout_layer = readout_layer.eval()
    return model, readout_layer

def get_raw_data(dataset):
    if dataset == _settings.YACHT_NAME:
        raw_df = pd.read_fwf(os.path.join(_settings.YACHT_PATH, 'yacht_hydrodynamics.data'), header=None)
        raw_df.columns = ["X%d" % d for d in range(raw_df.shape[1] - 1)] + ['Y']
        raw_df = raw_df.reset_index()
    elif dataset == _settings.HOUSING_NAME:
        import sklearn.datasets
        data = sklearn.datasets.load_boston()
        raw_df = pd.DataFrame(data['data'])
        raw_df['Y'] = data['target']
        raw_df.columns = ["X%d" % d for d in range(raw_df.shape[1] - 1)] + ['Y']
        raw_df = raw_df.reset_index()
    elif dataset == _settings.ENERGY_NAME:  # Energy NN does not train...
        raw_df = pd.read_excel(os.path.join(_settings.ENERGY_PATH, 'ENB2012_data.xlsx'), engine='openpyxl')
        # raw_df = raw_df.iloc[:, :10]
        # raw_df.columns = ["X%d" % d for d in range(self.raw_df.shape[1] - 2)] + ['Y0', 'Y1']
        raw_df = raw_df.iloc[:, :9]
        raw_df.columns = ["X%d" % d for d in range(raw_df.shape[1] - 1)] + ['Y']
        raw_df = raw_df.reset_index()
    elif dataset == _settings.KIN8NM_NAME:
        raw_df = pd.read_csv(os.path.join(_settings.KIN8NM_PATH, 'dataset_2175_kin8nm.csv'))
        raw_df.columns = ["X%d" % d for d in range(raw_df.shape[1] - 1)] + ['Y']
        raw_df = raw_df.reset_index()
    elif dataset == _settings.CONCRETE_NAME:
        raw_df = pd.read_excel(os.path.join(_settings.CONCRETE_PATH, 'Concrete_Data.xls'))
        raw_df.columns = ["X%d" % d for d in range(raw_df.shape[1] - 1)] + ['Y']
        raw_df = raw_df.reset_index()
    elif dataset == _settings.BIKE_NAME: #The base DNN does not learn anything
        from zipfile import ZipFile
        archive = ZipFile(os.path.join(_settings.BIKE_PATH, 'Bike-Sharing-Dataset.zip'))
        #raw_df = pd.read_csv(archive.open('day.csv')).set_index('dteday', verify_integrity=True).drop('instant',axis=1)
        raw_df = pd.read_csv(archive.open('hour.csv')).drop('instant', axis=1)#.set_index('instant', verify_integrity=True)
        drop_cols = ['yr', 'mnth', 'dteday']
        enum_cols = ['season', 'hr', 'weekday', 'weathersit']
        raw_df = raw_df.drop(drop_cols, axis=1)
        for enum_col in enum_cols:
            ser = raw_df[enum_col]
            tdf = pd.get_dummies(ser).rename(columns=lambda x: "%s%d"%(enum_col, x))
            raw_df = pd.concat([raw_df, tdf], axis=1).drop(enum_col, axis=1)
        raw_df = raw_df.reindex(columns=[c for c in raw_df.columns] + ['cnt'])
        raw_df.columns = ["X%d" % d for d in range(raw_df.shape[1] - 1)] + ['Y']
        raw_df = raw_df.reset_index()
    elif dataset == _settings.SYNT_DJKP:
        raise NotImplementedError()
    else: #TODO: Add more dataset here
        raise Exception()
    other_cols = [c for c in raw_df.columns if not (c.startswith('X') or c.startswith('Y'))]
    Y_col = ['Y']
    X_cols = [c for c in raw_df.columns if c.startswith('X')]
    raw_df = raw_df.reindex(columns=other_cols + X_cols + Y_col)
    #print(raw_df)
    return raw_df

class GeneralDataSplitter:
    def __init__(self, dataset=_settings.YACHT_NAME,
                 seed=7, split_ratio=[60, 20, 20],
                 model_setting = 0, #DNN
                 init=False, quiet=False):
        key = f'seed{seed}-{"-".join(map(str,split_ratio))}'
        self.save_path = os.path.join(_settings.WORKSPACE, dataset, key)
        self.seed = seed
        self.split_ratio = split_ratio
        self.dataset = dataset
        self.model_setting = model_setting
        self.raw_df = None
        self.quiet = quiet

        if init: self.initialize()

    def initialize(self):
        self.raw_df = get_raw_data(self.dataset)
        # reserved column names: index, X%d, Y (or Y%d)
        self.split_and_save()
        if self.model_setting is not None:
            self.pretrain()
            for split in [TRAIN, VALID, TEST]: self.eval(split)
            self.pretrain_resid()
            for split in [TRAIN, VALID, TEST]: self.eval_resid(split)

    def get_data_dir(self):
        return self.save_path

    def get_data_path(self, split=TRAIN):
        assert split in {TRAIN, VALID, TEST}
        return os.path.join(self.save_path, '%s.csv'%split)

    def get_checkpoint_dir(self, resid=False):
        if resid:
            return os.path.join(self.get_checkpoint_dir(resid=False), 'resid')
        return os.path.join(self.save_path, 'models', 'model%d'%self.model_setting)

    def get_embedding_path(self, split=TRAIN):
        return os.path.join(self.save_path, 'preds', 'model%d'%self.model_setting, '%s.csv'%split)

    def get_readout_weight_path(self):
        return os.path.join(self.save_path, 'preds', 'model%d'%self.model_setting, 'readout.pt')

    def get_preds_path(self, split=TRAIN, resid=False):
        if resid:
            dir_ = os.path.dirname(self.get_preds_path(split, resid=False))
            return os.path.join(dir_, '%s_pred_resid.csv'%split)
        return os.path.join(self.save_path, 'preds', 'model%d'%self.model_setting, '%s_pred.csv'%split)

    @classmethod
    def _split_df(cls, df, seed=7, split_ratio=[60, 20, 20]):
        n = len(df)
        np.random.seed(seed)
        perm = np.random.permutation(n)
        df = df.iloc[perm]
        split_ratio = np.concatenate([[0.], np.cumsum(split_ratio) / sum(split_ratio)])
        splits = np.round(split_ratio * n).astype(np.int)
        return [df.iloc[splits[i]:splits[i+1]] for i in range(len(split_ratio)-1)]

    def split_and_save(self):
        if not os.path.isdir(self.save_path): os.makedirs(self.save_path)
        save_paths = [self.get_data_path(s) for s in [TRAIN, VALID, TEST]]
        if all([os.path.isfile(f) for f in save_paths]): return save_paths
        dfs = self._split_df(self.raw_df, self.seed, self.split_ratio)
        for df, fname in zip(dfs, save_paths): df.to_csv(fname, index=False)
        return save_paths


    def get_data(self, split=TRAIN, colnames=False):
        train_df = pd.read_csv(self.get_data_path(split=split))
        Ys_cols = [c for c in train_df.columns if c.startswith('Y')]
        Xs_cols = [c for c in train_df.columns if c.startswith('X')]
        X, Y = train_df.loc[:, Xs_cols].values, train_df.loc[:, Ys_cols].values
        index = train_df['index'].values
        if colnames: return X, Y, index, Xs_cols, Ys_cols
        return X, Y, index

    def pretrain_resid(self, quiet=None):
        quiet = quiet or self.quiet
        checkpoint_dir = self.get_checkpoint_dir(resid=True)
        flag_pkl = os.path.join(checkpoint_dir, 'meta.pkl')
        if os.path.isfile(flag_pkl): return checkpoint_dir
        if os.path.isdir(checkpoint_dir): #and no meta
            shutil.rmtree(checkpoint_dir)
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        X, Y, _ = self.get_data(TRAIN)
        Yhat = pd.read_csv(self.get_preds_path(TRAIN)).drop('index', axis=1).values
        resid = np.abs(Y - Yhat)
        model, readout = pretrain_general(X, resid, self.seed, self.model_setting, quiet=quiet)

        torch.save(model, os.path.join(checkpoint_dir, 'model.pt'))

        pd.to_pickle({"Done": True}, flag_pkl)
        return checkpoint_dir

    def eval_resid(self, split=VALID):
        checkpoint_dir = self.pretrain_resid()
        model_resid = torch.load(os.path.join(checkpoint_dir, 'model.pt'))
        X, Y, index, X_cols, Y_cols = self.get_data(split, colnames=True)
        #save the prediction etc
        rhat = model_resid.predict(X)
        pred_path = self.get_preds_path(split, resid=True)
        pred_df = pd.DataFrame(rhat, columns=Y_cols)
        pred_df['index'] = index
        pred_df.reindex(columns=['index'] + Y_cols).to_csv(pred_path, index=False)
        return pred_path

    def pretrain(self, force_retrain=False, quiet=None):
        quiet = quiet or self.quiet
        checkpoint_dir = self.get_checkpoint_dir()
        readout_weight_path = self.get_readout_weight_path()
        flag_pkl = os.path.join(checkpoint_dir, 'meta.pkl')
        if os.path.isfile(flag_pkl):
            if force_retrain:
                time_key = "%s" % (datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
                old_mv_to = f"{checkpoint_dir}_Copy%s"%time_key
                os.rename(checkpoint_dir, old_mv_to)
            else:
                return checkpoint_dir, readout_weight_path
        if os.path.isdir(checkpoint_dir): #and no meta
            shutil.rmtree(checkpoint_dir)
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        X, Y, _ = self.get_data(TRAIN)
        model, readout = pretrain_general(X, Y, self.seed, self.model_setting, quiet=quiet)

        #save..
        #save the model
        torch.save(model, os.path.join(checkpoint_dir, 'model.pt'))

        #save the readout layer

        if not os.path.isdir(os.path.dirname(readout_weight_path)): os.makedirs(os.path.dirname(readout_weight_path))
        torch.save(readout, readout_weight_path)

        pd.to_pickle({"Done": True}, flag_pkl)
        return checkpoint_dir, readout_weight_path

    def get_model(self):
        return torch.load(self.get_readout_weight_path()).cpu().eval()

    def eval(self, split=VALID):
        checkpoint_dir, readout_weight_path = self.pretrain()
        model = torch.load(os.path.join(checkpoint_dir, 'model.pt'))
        readout = torch.load(readout_weight_path)
        X, Y, index, X_cols, Y_cols = self.get_data(split, colnames=True)
        #save the prediction etc
        Yhat = model.predict(X)
        embedding = model.embed(X)
        Yhat2 = readout(torch.tensor(embedding).float()).detach().cpu().numpy()
        assert np.allclose(Yhat, Yhat2)

        #save embeddings
        embedding_path = self.get_embedding_path(split)
        fcols = ['f%d'%i for i in range(embedding.shape[1])]
        embedding_df = pd.DataFrame(embedding, columns=fcols)
        embedding_df['index'] = index
        embedding_df.reindex(columns=['index'] + fcols).to_csv(embedding_path, index=False)

        #save predictions
        pred_path = self.get_preds_path(split)
        pred_df = pd.DataFrame(Yhat, columns=Y_cols)
        pred_df['index'] = index
        pred_df.reindex(columns=['index'] + Y_cols).to_csv(pred_path, index=False)

        print(f'{split}: MSE={np.mean(np.power(Y - Yhat, 2))}, Data Var={np.mean(np.power(Y - np.mean(Y), 2))}')
        return pred_path, embedding_path

def cache(*args, **kwargs):
    return GeneralDataSplitter(*args, **kwargs)


if __name__ == '__main__':
    task_runner = utils.TaskPartitioner()
    for dataset in [_settings.KIN8NM_NAME, _settings.YACHT_NAME, _settings.HOUSING_NAME, _settings.CONCRETE_NAME, _settings.ENERGY_NAME, _settings.BIKE_NAME]:
        for seed in tqdm.tqdm(range(10)):
            task_runner.add_task(cache, dataset=dataset, seed=seed, model_setting=0, init=True, quiet=True)
    task_runner.run_multi_process(8)