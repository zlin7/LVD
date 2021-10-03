import sys, os, numpy as np, pandas as pd
from importlib import reload
import _settings
import itertools
import data.dataloader as dld; reload(dld)
import models.regmodel as regmodel; reload(regmodel)
import models.conformal as conformal; reload(conformal)
import demos.demo as demo; reload(demo)
import demos.regression as reg; reload(reg)
import demos.experiments as exp; reload(exp)
import models.baselines.DE as DE; reload(DE)
import utils.utils as utils
import time, datetime
import ipdb

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task_id', type=int, help="task id")
parser.add_argument('-np', '--npartitions', type=int, default=1, help="number of partitions")
parser.add_argument('-pkw', '--pkwarg', type=str, default=None, help="what argument to pass to each process (e.g. gpuid)")
args = parser.parse_args()

task_runner_seq = [utils.TaskPartitioner() for _ in range(20)] #sequential runners
non_pkw_task_runner_seq = [utils.TaskPartitioner() for _ in range(20)] #sequential runners, without pkwarg

if args.task_id == 100 or args.task_id == 101: #Smaller datasets (100 is default, 101 includes all variants)
    smoothness_variants = [True] #Add the smoothness requirement
    MN_variants = ['LocalConformal'] #Without normalization by MAD prediction
    mu_variants = ['base'] #i.e. NN model
    if args.task_id == 101:
        smoothness_variants.append(False)
        MN_variants.append('LocalConformalMAD') #For Variants: this is the "MN" version.
        mu_variants.append('KR')

    nseeds = 10
    alphas=  [0.1, 0.5]
    conf_baselines = ['MADSplit', 'CQR']

    for dataset in ['UCI_Yacht']:#  _settings.SMALL_DATASETS:
        for seed in range(nseeds):
            datakwargs = {'seed': seed, 'model_setting': 0, 'test_split': dld.TEST, 'train_split': dld.TRAIN, 'val_split': dld.VALID}
            train_datakwargs, val_datakwargs, test_datakwargs = demo.sep_datakwargs(datakwargs)
            default_fitkwargs = exp.get_default_fitkwargs()
            for ybar in smoothness_variants:
                fitkwargs = utils.merge_dict_inline(default_fitkwargs, {"seed": seed, "ybar_bias": ybar})
                # Step 1: Train the kernel
                task_runner_seq[0].add_task(reg.get_trained_model_cached, dataset, regmodel._KERNEL_MLKR, train_datakwargs,
                                            **fitkwargs)
                # Step 2: Save the predictions (so if we have variants in the next step, we do not need to make the predictions again.)
                for _eval_datakwargs in [val_datakwargs, test_datakwargs]:
                    task_runner_seq[1].add_task(reg.eval_trained_model_cached, dataset, _eval_datakwargs,
                                                regmodel._KERNEL_MLKR, train_datakwargs, **fitkwargs)
                # Step 3:
                for pred in mu_variants:
                    PIkwargs = {"kernel": "trained", 'pred': pred}
                    for alpha in alphas:
                        for PI_model in MN_variants:
                            task_runner_seq[2].add_task(demo.eval_exp_cached, dataset, datakwargs,
                                                        regmodel._KERNEL_MLKR, fitkwargs,
                                                        PI_model=PI_model, PIkwargs=PIkwargs,
                                                        alpha=alpha)
            #Step 4: Save results for conformal baselines
            for m in conf_baselines:
                for alpha in alphas:
                    non_pkw_task_runner_seq[0].add_task(demo.conformal_baselines_cached, m, dataset, datakwargs, alpha=alpha)

            # Step 5: Save results for non-valid baselines
            params = dict({"activation": 'ReLU', "num_hidden": 100, "num_layers": 1})
            train_params = dict({"num_iter": 1000, "learning_rate": 1e-3})
            for m in ["DJ", "MCDP", "PBP", "DE"]:
                for alpha in alphas:
                    non_pkw_task_runner_seq[0].add_task(exp.run_experiments, [m], [dataset], [seed], damp=1e-2,
                                                        mode='exact', coverage=1 - alpha,
                                                        params=params, train_params=train_params,
                                                        data_path=_settings.DATA_PATH,
                                                        cache_path=os.path.join(_settings.WORKSPACE, 'Baselines'),
                                                        quiet=True)


if args.task_id == 200 or args.task_id == 201: #Smaller datasets (100 is default, 101 includes all variants)
    smoothness_variants = [True] #Add the smoothness requirement
    MN_variants = ['LocalConformal'] #Without normalization by MAD prediction
    mu_variants = ['base'] #i.e. NN model
    if args.task_id == 201:
        smoothness_variants.append(False)
        MN_variants.append('LocalConformalMAD') #For Variants: this is the "MN" version.
        mu_variants.append('KR')

    nseeds = 10
    alphas=  [0.1, 0.5]
    conf_baselines = ['MADSplit', 'CQR']

    for dataset, ntasks in _settings.QM_DATASETS.items():
        if dataset == _settings.QM9_NAME: continue
        for seed in range(nseeds):
            for which_y in range(ntasks):
                datakwargs = {'which_y': which_y, 'seed': seed, 'test_split': dld.TEST, 'train_split': dld.TRAIN, 'val_split': dld.VALID}
                datakwargs['split_ratio'] = (60, 20, 20)
                train_datakwargs, val_datakwargs, test_datakwargs = demo.sep_datakwargs(datakwargs)
                default_fitkwargs = exp.get_default_fitkwargs()
                for ybar in smoothness_variants:
                    fitkwargs = utils.merge_dict_inline(default_fitkwargs, {"seed": seed, "ybar_bias": ybar})
                    # Step 1: Train the kernel

                    task_runner_seq[0].add_task(reg.get_trained_model_cached, dataset, regmodel._KERNEL_MLKR,
                                                train_datakwargs,
                                                **fitkwargs)
                    # Step 2: Save the predictions (so if we have variants in the next step, we do not need to make the predictions again.)
                    for _eval_datakwargs in [val_datakwargs, test_datakwargs]:
                        task_runner_seq[1].add_task(reg.eval_trained_model_cached, dataset, _eval_datakwargs,
                                                    regmodel._KERNEL_MLKR, train_datakwargs, **fitkwargs)
                    # Step 3:
                    for pred in mu_variants:
                        PIkwargs = {"kernel": "trained", 'pred': pred}
                        for alpha in alphas:
                            for PI_model in MN_variants:
                                task_runner_seq[2].add_task(demo.eval_exp_cached, dataset, datakwargs,
                                                            regmodel._KERNEL_MLKR, fitkwargs,
                                                            PI_model=PI_model, PIkwargs=PIkwargs,
                                                            alpha=alpha)
                for m in conf_baselines:
                    for alpha in alphas:
                        non_pkw_task_runner_seq[0].add_task(demo.conformal_baselines_cached, m, dataset,
                                                            {'which_y': which_y, 'seed': seed}, alpha=alpha)
                for alpha in alphas:
                    non_pkw_task_runner_seq[0].add_task(DE.get_DE_result_cached, dataset, which_y, seed, alpha)

#Actually running the tasks
times = [datetime.datetime.now()]
for _tr in task_runner_seq:
    continue
    if _tr.task_list is None or len(_tr.task_list) == 0: continue
    _tr.run_multi_process(args.npartitions, process_kwarg=args.pkwarg)
    times.append(datetime.datetime.now())
for _tr in non_pkw_task_runner_seq:
    if _tr.task_list is None or len(_tr.task_list) == 0: continue
    _tr.run_multi_process(args.npartitions)
    times.append(datetime.datetime.now())
print("\n\nFinsihed:\n")
print(times)