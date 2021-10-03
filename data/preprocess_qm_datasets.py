import _settings
import numpy as np
import torch
import pandas as pd
import os
import utils.utils as utils
import datetime
import shutil
import ipdb

TRAIN, VALID, TEST = 'train', 'val', 'test'


def pretrain_chemprop(seed, data_path, save_dir, quiet=False,
                      baseline=None, baseline_kwargs={}, gpuid=None):

    utils.set_all_seeds(seed)
    from extern import chemprop
    args = ['--data_path', data_path, '--dataset_type', "regression", '--save_dir', save_dir, '--num_workers', '0']
    if quiet: args.append('--quiet')
    if baseline == 'CQR':
        args.extend(['--alpha', str(baseline_kwargs['alpha'])])
        args[3] = 'quantile_regression'
    if baseline == 'DE':
        args.extend(['--seed', str(baseline_kwargs['seed'])])
        args[3] = 'nll_regression'
    if gpuid is not None:
        args.extend(['--gpu', str(gpuid)])
    #ipdb.set_trace()
    chemprop.train.cross_validate(args=chemprop.args.TrainArgs().parse_args(args),
                                  train_func=chemprop.train.run_training)

def eval_embeddings_chemprop(checkpoint_dir, pred_path, data_path, embed_only=True, readout_weight_path=None,
                             baseline=None, baseline_kwargs={}, gpuid=None):
    from extern import chemprop
    args = ['--test_path', data_path,
            '--preds_path', pred_path,
            '--checkpoint_dir', checkpoint_dir,
            '--readout_weight_path', readout_weight_path,
            '--num_workers', '0']
    if embed_only: args.append('--embed_only')
    if gpuid is not None:
        args.extend(['--gpu', str(gpuid)])
    chemprop.train.make_predictions(args=chemprop.args.PredictArgs().parse_args(args))


class QMSplitter:
    def __init__(self, dataset=_settings.QM8_NAME,
                 seed=7, split_ratio=[60, 20, 20], quiet=False,
                 baseline=None, baseline_kwargs = {},
                 gpuid=None):
        if dataset == _settings.QM8_NAME:
            key = f'seed{seed}-{"-".join(map(str,split_ratio))}'
            self.save_path = os.path.join(_settings.WORKSPACE, _settings.QM8_NAME, key)
            self.raw_df = pd.read_csv(os.path.join(_settings.QM8_PATH, 'qm8.csv'))
        elif dataset == _settings.QM9_NAME:
            key = f'seed{seed}-{"-".join(map(str,split_ratio))}'
            self.save_path = os.path.join(_settings.WORKSPACE, _settings.QM9_NAME, key)
            self.raw_df = pd.read_csv(os.path.join(_settings.QM9_PATH, 'qm9.csv'))
            cnts = self.raw_df.groupby('smiles')['mol_id'].nunique()
            dups = cnts[cnts==2]
            self.raw_df = self.raw_df[~self.raw_df['smiles'].isin(dups.index)].reset_index().drop(['index','mol_id'], axis=1)
            tasks = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'u0', 'u298', 'h298', 'g298', 'cv']
            # https://www.nature.com/articles/sdata201422.pdf
            # https://docs.dgl.ai/en/0.6.x/_modules/dgl/data/qm9.html
            self.raw_df = self.raw_df.reindex(columns=['smiles'] + tasks)
        else:
            raise NotImplementedError()
        self.seed = seed
        self.split_ratio = split_ratio
        self.dataset = dataset
        self.gpuid = gpuid

        #init
        self.baseline = None
        self.baseline_kwargs = {}
        self.split_and_save()
        self.train_chemprop(quiet=quiet)
        [self.eval_embed_chemprop(s) for s in [TRAIN, VALID, TEST]]
        [self.eval_pred_chemprop(s) for s in [TRAIN, VALID, TEST]]

        #for training different models...
        self.baseline = baseline
        self.baseline_kwargs = baseline_kwargs
        self.base_obj = None
        if self.baseline is None:
            pass
        elif self.baseline == 'MADSplit':
            self.base_obj = QMSplitter(dataset, seed, split_ratio, quiet=quiet)
            self.save_path = os.path.join(self.save_path, self.baseline)
            if not os.path.isdir(self.save_path): os.makedirs(self.save_path)
            diff_dfs = {}
            for s in [TRAIN, VALID, TEST]:
                pred_df_ = pd.read_csv(self.base_obj.get_preds_path(s))
                y_df_ = pd.read_csv(self.base_obj.get_data_path(s))
                assert y_df_['smiles'].eq(pred_df_['smiles']).all()
                diff_dfs[s] = y_df_.reindex()
                diff_dfs[s].iloc[:, 1:] -= pred_df_.iloc[:, 1:]
                diff_dfs[s].iloc[:, 1:] = diff_dfs[s].iloc[:, 1:].abs()
                if not os.path.isfile(self.get_data_path(s)):
                    diff_dfs[s].to_csv(self.get_data_path(s), index=False)
            self.train_chemprop(quiet=quiet)
            [self.eval_embed_chemprop(s) for s in [TRAIN, VALID, TEST]]
            [self.eval_pred_chemprop(s) for s in [TRAIN, VALID, TEST]]
        elif self.baseline == 'CQR':
            assert self.baseline_kwargs['alpha'] in {0.1, 0.5}
            self.base_obj = QMSplitter(dataset, seed, split_ratio, quiet=quiet)
            self.save_path = os.path.join(self.save_path, self.baseline, f"alpha={self.baseline_kwargs['alpha']:.2f}")
            if not os.path.isdir(self.save_path): os.makedirs(self.save_path)
            for s in [TRAIN, VALID, TEST]:
                if not os.path.isfile(self.get_data_path(s)):
                    shutil.copy(self.base_obj.get_data_path(s), self.get_data_path(s))
            self.train_chemprop(quiet=quiet)
            [self.eval_embed_chemprop(s) for s in [TRAIN, VALID, TEST]]
            [self.eval_pred_chemprop(s) for s in [TRAIN, VALID, TEST]]
        elif self.baseline == 'DE':
            assert self.baseline_kwargs['seed'] in {0,1,2,3,4}
            self.base_obj = QMSplitter(dataset, seed, split_ratio, quiet=quiet)
            self.save_path = os.path.join(self.save_path, self.baseline, f"model={self.baseline_kwargs['seed']}")
            if not os.path.isdir(self.save_path): os.makedirs(self.save_path)
            for s in [TRAIN, VALID, TEST]:
                if not os.path.isfile(self.get_data_path(s)):
                    if s == VALID:
                        odf = pd.read_csv(self.base_obj.get_data_path(s))
                        odf.reindex([]).to_csv(self.get_data_path(s), index=False)
                    if s == TRAIN:
                        odf1 = pd.read_csv(self.base_obj.get_data_path(TRAIN))
                        odf2 = pd.read_csv(self.base_obj.get_data_path(VALID))
                        assert odf1.columns.equals(odf2.columns)
                        tdf = pd.concat([odf1, odf2], axis=0)
                        tdf.to_csv(self.get_data_path(s), index=False)
                    if s == TEST:
                        shutil.copy(self.base_obj.get_data_path(s), self.get_data_path(s))
            self.split_ratio = [self.split_ratio[0] + self.split_ratio[1], 0, self.split_ratio[2]]
            self.train_chemprop(quiet=quiet)
            [self.eval_pred_chemprop(s) for s in [TRAIN, TEST]]
        else:
            raise NotImplementedError()

    def get_data_dir(self):
        if self.base_obj is not None: return self.base_obj.get_data_dir()
        return self.save_path

    def get_checkpoint_dir(self):
        return os.path.join(self.save_path, 'models', 'default')

    def get_data_path(self, split=TRAIN):
        assert split in {TRAIN, VALID, TEST}
        return os.path.join(self.save_path, '%s.csv'%split)

    def get_embedding_path(self, split=TRAIN):
        return os.path.join(self.save_path, 'preds', 'default', '%s.csv'%split)

    def get_readout_weight_path(self):
        return os.path.join(self.save_path, 'preds', 'default', 'readout.pt')

    def get_preds_path(self, split=TRAIN):
        return os.path.join(self.save_path, 'preds', 'default', '%s_pred.csv'%split)

    @classmethod
    def _split_df(cls, df, seed=7, split_ratio=[80, 10, 10]):
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
        print("Saving files")
        dfs = self._split_df(self.raw_df, self.seed, self.split_ratio)
        for df, fname in zip(dfs, save_paths): df.to_csv(fname, index=False)
        return save_paths

    def train_chemprop(self, force_retrain=False, quiet=False):
        checkpoint_dir = self.get_checkpoint_dir()
        flag_pkl = os.path.join(checkpoint_dir, 'meta.pkl')
        if os.path.isfile(flag_pkl):
            if force_retrain:
                time_key = "%s" % (datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
                #os.rename(checkpoint_dir, os.path.join())
                old_mv_to = os.path.join(os.path.dirname(checkpoint_dir), 'Copy%s'%time_key)
                os.rename(checkpoint_dir, old_mv_to)
                print(old_mv_to)
                ipdb.set_trace()
            else:
                return checkpoint_dir
        if os.path.isdir(checkpoint_dir): #and no meta
            shutil.rmtree(checkpoint_dir)
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        pretrain_chemprop(self.seed, self.get_data_path(TRAIN), checkpoint_dir, quiet=quiet,
                          baseline=self.baseline, baseline_kwargs=self.baseline_kwargs, gpuid=self.gpuid)

        pd.to_pickle({"Done": True}, flag_pkl)
        return checkpoint_dir

    def eval_embed_chemprop(self, split=VALID):
        checkpoint_dir = self.train_chemprop()
        pred_path = self.get_embedding_path(split)
        if split == TRAIN:
            readout_weight_path = self.get_readout_weight_path()
        else:
            readout_weight_path = None
        if (readout_weight_path is not None and not os.path.isfile(readout_weight_path)) or not os.path.isfile(pred_path):
            eval_embeddings_chemprop(checkpoint_dir, pred_path, self.get_data_path(split), embed_only=True,
                                     readout_weight_path=readout_weight_path,
                                     baseline=self.baseline, baseline_kwargs=self.baseline_kwargs, gpuid=self.gpuid)
        return pred_path

    def eval_pred_chemprop(self, split=VALID):
        checkpoint_dir = self.train_chemprop()
        pred_path = self.get_preds_path(split)
        if (not os.path.isfile(pred_path)):# or (hasattr(self, 'baseline') and self.baseline == 'CQR'):
            eval_embeddings_chemprop(checkpoint_dir, pred_path, self.get_data_path(split), embed_only=False)
        return pred_path

    def get_model(self):
        return torch.load(self.get_readout_weight_path())

def test_readout_model(dataset=_settings.QM8_NAME):
    o = QMSplitter(dataset=dataset)
    readout = torch.load(o.get_readout_weight_path())
    x = pd.read_csv(o.get_embedding_path(split=VALID)).iloc[:,1:].values
    y = pd.read_csv(o.get_preds_path(split=VALID)).iloc[:,1:].values
    yhat = readout(torch.tensor(x).float())
    diff = yhat - torch.tensor(y).float()
    assert diff.abs().max() < 1e-4

def cache_predictions(*args, **kwargs):
    o = QMSplitter(*args, **kwargs)

if __name__ == '__main__':
    #Train the base chemprop model
    task_runner = utils.TaskPartitioner()
    for dataset in [_settings.QM8_NAME, _settings.QM9_NAME]:
        for seed in range(10):
            task_runner.add_task(cache_predictions, dataset, split_ratio=[60, 20, 20], seed=seed)
    task_runner.run_multi_process(8)

    # Train other predictors (quantile, residual, ensemble with NLL loss) for baselines
    task_runner = utils.TaskPartitioner()
    for dataset in [_settings.QM8_NAME, _settings.QM9_NAME]:
        for seed in range(10):
            task_runner.add_task(cache_predictions, dataset, split_ratio=[60, 20, 20], seed=seed, baseline='MADSplit')
            task_runner.add_task(cache_predictions, dataset, split_ratio=[60, 20, 20], seed=seed, baseline='CQR', baseline_kwargs={"alpha": 0.1})
            task_runner.add_task(cache_predictions, dataset, split_ratio=[60, 20, 20], seed=seed, baseline='CQR', baseline_kwargs={"alpha": 0.5})
            for model_id in range(5):
                task_runner.add_task(cache_predictions, dataset, split_ratio=[60, 20, 20], seed=seed,
                                     baseline='DE', baseline_kwargs={"seed": model_id})
    task_runner.run_multi_process(8)