import numpy as np
import os, glob
import bisect
import ipdb
import _settings as _settings
import torch
from torch.utils.data import Dataset
import pandas as pd
import utils.utils as utils
from importlib import reload
import torchvision
reload(_settings)

TRAIN = 'train'
VALID = 'val'
TEST = 'test'
VALID_1 = 'val1'
VALID_2 = 'val2'

SEED_OFFSETS = {TRAIN: 101, VALID: 202, VALID_1: 203, VALID_2: 204, TEST: 303}

SYNT_NAME = 'Synthetic'

class DatasetWrapper(Dataset):
    def __init__(self, mode=TRAIN):
        super(DatasetWrapper, self).__init__()
        self.mode = mode
        assert hasattr(self, 'DATASET'), "Please give this dataset a name"
        assert hasattr(self, 'LABEL_MAP'), "Please give a name to each class {NAME: class_id}"
    def is_train(self):
        return self.mode == TRAIN
    def is_test(self):
        return self.mode == TEST
    def is_valid(self):
        return self.mode == VALID
    def idx2pid(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        return idx
    #def get_class_frequencies(self):
    #    raise NotImplementedError()


#======================Synthetic Data
class DJKPSyntData(DatasetWrapper):
    DATASET = _settings.SYNT_DJKP
    LABEL_MAP = None
    def __init__(self, split=TRAIN, n=None, seed=7,
                 sigma=1., dist='normal', param=1.):
        super(DJKPSyntData, self).__init__(split)
        if n is None: n = 300 if split == TRAIN else 100
        np.random.seed(seed + SEED_OFFSETS[split])
        if dist == 'unif':
            self.x = np.random.uniform(-param, param, n)
            self.eps = np.random.normal(0, self.sigma, n)
            self.y = np.power(self.x, 3) + self.eps
        elif dist == 'normal':
            self.x = np.random.normal(0, param, n)
            self.eps = np.random.normal(0, self.sigma, n)
            self.y = np.power(self.x, 3) + self.eps
        elif dist == 'normal+':
            x1 = np.random.normal(param, param, n)
            x1 = np.max(np.stack([x1-param, param-x1], 1),1) + param #half normal
            x2 = np.random.uniform(-param, param, n)
            eps1 = np.random.normal(0, sigma, n)
            eps2 = np.random.normal(0, sigma, n)
            y1 = np.power(x1, 3) + eps1
            y2 = np.power(x2, 3) + eps2
            self.x, self.y, self.eps = [], [], []
            us = np.random.uniform(0, 1, n)
            for i in range(n):
                if us[i] < 0.1:
                    self.x.append(x1[i])
                    self.y.append(y1[i])
                    self.eps.append(eps1[i])
                else:
                    self.x.append(x2[i])
                    self.y.append(y2[i])
                    self.eps.append(eps2[i])
            self.x = np.asarray(self.x)
            self.y = np.asarray(self.y)
        self.dist = dist
        self.sigma = sigma

        self.n = n
        self.index = np.arange(n)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x_i = np.expand_dims(self.x[idx], 0)
        y_i = self.y[idx]
        return x_i, y_i, idx

    def get_all_X(self):
        return np.expand_dims(self.x, 1)

    def get_all_Y(self):
        return self.y


#=============================QM Datasets
class QMDataset(DatasetWrapper):
    LABEL_MAP = None
    def __init__(self, dataset=_settings.QM8_NAME, split=VALID, which_y=0, seed=7, split_ratio=[80, 10, 10], read_resid=True):
        """

        :param split: TRAIN/VAL/TEST
        :param which_y: which regression target. If None, we will use all values
        :param seed: for QMSplitter
        :param split_ratio: for QMSplitter
        """
        self.DATASET = dataset
        super(QMDataset, self).__init__(split)

        import data.preprocess_qm_datasets
        if isinstance(split_ratio, tuple): split_ratio = list(split_ratio)
        self.qmsplitter_resid = data.preprocess_qm_datasets.QMSplitter(self.DATASET, seed=seed, split_ratio=split_ratio, baseline='MADSplit')
        self.qmsplitter = self.qmsplitter_resid.base_obj
        read_split = split if split not in {VALID_1, VALID_2} else VALID
        self.data = pd.read_csv(self.qmsplitter.get_embedding_path(split=read_split))
        self.truths = pd.read_csv(self.qmsplitter.get_data_path(split=read_split))
        self._preds = pd.read_csv(self.qmsplitter.get_preds_path(split=read_split))
        self._resid_preds = pd.read_csv(self.qmsplitter_resid.get_preds_path(split=read_split))
        if split == VALID_1 or split == VALID_2:
            n1 = (len(self.truths)+1)//2
            if split == VALID_1:
                self.data, self.truths, self._preds, self._resid_preds = self.data.iloc[:n1], self.truths.iloc[:n1], self._preds.iloc[:n1], self._resid_preds[:n1]
            else:
                self.data, self.truths, self._preds, self._resid_preds = self.data.iloc[n1:], self.truths.iloc[n1:], self._preds.iloc[n1:], self._resid_preds[n1:]
        self.n = len(self.truths)

        assert len(self.data) == self.n
        assert self.data['smiles'].equals(self.truths['smiles']) and self.data['smiles'].equals(self._preds['smiles']) and self.data['smiles'].equals(self._resid_preds['smiles'])

        self.index = self.data['smiles'].values
        self.data = self.data.iloc[:, 1:].values
        model = self.qmsplitter.get_model()
        #model_resid = self.qmsplitter_resid.get_model()
        self.model_resid = None
        if which_y is not None:
            self.y = self.truths.iloc[:, 1 + which_y].values

            self._yhat = self._preds.iloc[:, 1 + which_y].values
            self.model = torch.nn.Linear(model.in_features, 1)
            self.model.weight.data = model.weight.data[which_y, :]
            self.model.bias.data = model.bias.data[which_y]

            self._rhat = self._resid_preds.iloc[:, 1 + which_y].values
            #self.model_resid = torch.nn.Linear(model_resid.in_features, 1)
            #self.model_resid.weight.data = model_resid.weight.data[which_y, :]
            #self.model_resid.bias.data = model_resid.bias.data[which_y]
        else:
            self.y = self.truths.iloc[:, 1:].values
            self._yhat = self._preds.iloc[:, 1:].values
            self._rhat = self._resid_preds.iloc[:, 1:].values
            self.model = model
            #self.model_resid = model_resid

    @classmethod
    def model_predict(cls, x, readout):
        with torch.no_grad():
            x = torch.tensor(x).float()
            return float(readout(x).detach().numpy())

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        x_i = self.data[idx]
        y_i = self.y[idx]
        return x_i, y_i, idx

    def get_all_X(self):
        return self.data

    def get_all_Y(self):
        return self.y

    def _get_all_Yhat(self):
        return self._yhat

    def _get_all_rhat(self):
        return self._rhat

class QM8Dataset(QMDataset):
    Y_NAMES = {i:v for i, v in enumerate("E1-CC2	E2-CC2	f1-CC2	f2-CC2	E1-PBE0	E2-PBE0	f1-PBE0	f2-PBE0	E1-PBE0.1	E2-PBE0.1	f1-PBE0.1	f2-PBE0.1	E1-CAM	E2-CAM	f1-CAM	f2-CAM".split())}
    def __init__(self, split=VALID, which_y=0, seed=7, split_ratio=[60, 20, 20], read_resid=True):
        super(QM8Dataset, self).__init__(_settings.QM8_NAME, split=split, which_y=which_y, seed=seed, split_ratio=split_ratio, read_resid=read_resid)

class QM9Dataset(QMDataset):
    Y_NAMES = {i:v for i,v in enumerate("mu	alpha	homo	lumo	gap	r2	zpve	u0	u298	h298	g298	cv".split())}
    def __init__(self, split=VALID, which_y=0, seed=7, split_ratio=[80, 10, 10], read_resid=True):
        super(QM9Dataset, self).__init__(_settings.QM9_NAME, split=split, which_y=which_y, seed=seed, split_ratio=split_ratio, read_resid=read_resid)

#======================================================================================================
class ProcessedDataset(DatasetWrapper):
    LABEL_MAP = None
    def __init__(self, dataset, split=VALID, seed=7, split_ratio=[60, 20, 20],
                 which_y = None,
                 **kwargs):
        self.DATASET = dataset
        super(ProcessedDataset, self).__init__(mode=split)
        import data.preprocess_small_datasets as ppmt
        if isinstance(split_ratio, tuple): split_ratio = list(split_ratio)
        self.splitter = ppmt.GeneralDataSplitter(dataset, seed=seed, split_ratio=split_ratio, **kwargs)
        read_split = split if split not in {VALID_1, VALID_2} else VALID
        try:
            self.data = pd.read_csv(self.splitter.get_embedding_path(read_split))
        except:
            self.splitter.initialize()
            self.data = pd.read_csv(self.splitter.get_embedding_path(read_split))
        self.truths = pd.read_csv(self.splitter.get_data_path(read_split))
        self._preds = pd.read_csv(self.splitter.get_preds_path(read_split))
        self._resid_preds = pd.read_csv(self.splitter.get_preds_path(read_split, resid=True))

        self.raw_data = self.splitter.get_data(read_split)[0]
        if split == VALID_1 or split == VALID_2:
            n1 = (len(self.truths)+1)//2
            if split == VALID_1:
                self.data, self.truths, self._preds, self._resid_preds = self.data.iloc[:n1], self.truths.iloc[:n1], self._preds.iloc[:n1], self._resid_preds[:n1]
            else:
                self.data, self.truths, self._preds, self._resid_preds = self.data.iloc[n1:], self.truths.iloc[n1:], self._preds.iloc[n1:], self._resid_preds[n1:]
        self.n = len(self.truths)
        assert len(self.data) == self.n
        assert self.data['index'].equals(self.truths['index']) and self.data['index'].equals(self._preds['index'])and self.data['index'].equals(self._resid_preds['index'])

        self.index = self.data['index'].values
        self.data = self.data.drop('index', axis=1).values
        model = self.splitter.get_model()

        if which_y is not None:
            self.y = self.truths.loc[:, 'Y%d'%which_y].values
            self._yhat = self._preds.loc[:, 'Y%d'%which_y].values
            assert isinstance(model, torch.nn.Linear)
            self.model = torch.nn.Linear(model.in_features, 1)
            self.model.weight.data = model.weight.data[which_y, :]
            self.model.bias.data = model.bias.data[which_y]

            self._rhat = self._resid_preds.loc[:, 'Y%d'%which_y].values
        else:
            self.y = self.truths.loc[:, 'Y'].values
            self._yhat = self._preds.loc[:, 'Y'].values
            self.model = model
            self._rhat = self._resid_preds.loc[:, 'Y'].values

        self.model_resid = None

        self.model_resid_from_raw = None

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        x_i = self.data[idx]
        y_i = self.y[idx]
        return x_i, y_i, idx

    def get_all_X(self):
        return self.data

    def get_all_Y(self):
        return self.y

    def _get_all_Yhat(self):
        return self._yhat

    def _get_all_rhat(self):
        return self._rhat

class UCIYachtEmbed(ProcessedDataset):
    def __init__(self, split=VALID, seed=7, split_ratio=[60, 20, 20], **kwargs):
        super(UCIYachtEmbed, self).__init__(_settings.YACHT_NAME, split=split, seed=seed, split_ratio=split_ratio, **kwargs)

class UCIEnergyEmbed(ProcessedDataset):
    def __init__(self, split=VALID, seed=7, split_ratio=[60, 20, 20], **kwargs):
        super(UCIEnergyEmbed, self).__init__(_settings.ENERGY_NAME, split=split, seed=seed, split_ratio=split_ratio, **kwargs)

class Kin8nmEmbed(ProcessedDataset):
    def __init__(self, split=VALID, seed=7, split_ratio=[60, 20, 20], **kwargs):
        super(Kin8nmEmbed, self).__init__(_settings.KIN8NM_NAME, split=split, seed=seed, split_ratio=split_ratio, **kwargs)

class BikeEmbed(ProcessedDataset):
    def __init__(self, split=VALID, seed=7, split_ratio=[60, 20, 20], **kwargs):
        super(BikeEmbed, self).__init__(_settings.BIKE_NAME, split=split, seed=seed, split_ratio=split_ratio, **kwargs)

class ConcreteEmbed(ProcessedDataset):
    def __init__(self, split=VALID, seed=7, split_ratio=[60, 20, 20], **kwargs):
        super(ConcreteEmbed, self).__init__(_settings.CONCRETE_NAME, split=split, seed=seed, split_ratio=split_ratio, **kwargs)

class HousingEmbed(ProcessedDataset):
    def __init__(self, split=VALID, seed=7, split_ratio=[60, 20, 20], **kwargs):
        super(HousingEmbed, self).__init__(_settings.HOUSING_NAME, split=split, seed=seed, split_ratio=split_ratio, **kwargs)

def get_default_dataset(dataset=_settings.QM8_NAME, split=VALID, seed=_settings.RANDOM_SEED, **kwargs):
    if dataset == _settings.QM8_NAME:
        kwargs.setdefault('which_y', 0)
        return QM8Dataset(split=split, seed=seed, **kwargs)
    if dataset == _settings.QM9_NAME:
        kwargs.setdefault('which_y', 0)
        return QM9Dataset(split=split, seed=seed, **kwargs)
    if dataset == _settings.SYNT_DJKP:
        return DJKPSyntData(split=split, seed=seed, **kwargs)
    if dataset == _settings.YACHT_NAME:
        return UCIYachtEmbed(split, seed=seed, **kwargs)
    if dataset == _settings.ENERGY_NAME:
        return UCIEnergyEmbed(split, seed=seed, **kwargs)
    if dataset == _settings.KIN8NM_NAME:
        return Kin8nmEmbed(split, seed=seed, **kwargs)
    if dataset == _settings.BIKE_NAME:
        return BikeEmbed(split, seed=seed, **kwargs)
    if dataset == _settings.CONCRETE_NAME:
        return ConcreteEmbed(split, seed=seed, **kwargs)
    if dataset == _settings.HOUSING_NAME:
        return HousingEmbed(split, seed=seed, **kwargs)
    raise NotImplementedError()

if __name__ == '__main__':
    o = QM8Dataset()
