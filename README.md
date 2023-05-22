# LVD: Locally Valid and Discriminative Prediction Intervals for Deep Learning Models
This is the code for "Locally Valid and Discriminative Prediction Intervals for Deep Learning Models" (NeurIPS 2021). 


# Quick Start/Demo
See `notebooks/Demo_Synthetic.ipynb` for the experiment on synthetic dataset.
This notebook also serves as a simple-to-follow example of using LVD.

See `notebooks/Demo_SmallData.ipynb` for the experiment on smaller datasets (Bike, Yacht, Concrete, Kin8nm, BostonHousing, Energy)
The result for UCI_Yacht is included in "Temp" folder (which is the default cache folder according to `_settings.py`)

# Dependency

The full environment could be found in `conda_Linux.yml` and `conda_Win.yml` (depending on which OS).

## Credits

For some baselines (part of DE, MCDP, PBP) we used the implementation in [Discriminative Jackknife](https://github.com/ahmedmalaa/discriminative-jackknife).
We include the codes under `models/baselines/djkp`

 



# All Experiments (Replicate the paper)

See `notebooks/Demo_AllExperiments.ipynb` for all the experiment we showed in the paper.
Before running it, do the following:

## 0. Data Download
Before anything, first check whether `DATA_PATH` in `_settings.py` is correctly set. 
You would also need to download the data from the following websites and put them into the corresponding folders as specified by `_settings.py`.
1. `ENERGY_PATH`: https://archive.ics.uci.edu/ml/datasets/energy+efficiency
2. `YACHT_PATH`: http://archive.ics.uci.edu/ml/datasets/yacht+hydrodynamics
3. `KIN8NM_PATH`: https://www.openml.org/d/189
4. `CONCRETE_PATH`: http://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength
5. `BIKE_PATH`: https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
6. Boston Housing dataset is loaded from `sklearn.datasets.load_boston()`.
7. `QM8_PATH`: http://moleculenet.ai/datasets-1 ("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm8.csv")
8. `QM9_PATH`: http://moleculenet.ai/datasets-1 ("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv")

(All links are working as of 5/27/2021.)

## 1. Prepare QM8 and QM9 datasets
The first step is to run the chemprop models and cache down the predictions (and embeddings).
The original implementation is [here](https://github.com/chemprop/chemprop). 
However, please use [this branch](https://github.com/zlin7/chemprop/tree/zl), which contains some modifications for baselines to work.
Then, specify the path in `DEPENDENCY_PATHS` in`extern.py`

To run the chemprop models, run `python -m data.preprocessing_qm_datasets`.
This will first cache down the results (embeddings, predictions, ...) necessary for LVD, and then those for other baselines including MADSplit, CQR and DE.
Note that it has to be run sequentially as the some modifications depend on the base model.

## 2. Prepare Other Datasets
For all other datasets, run `python -m data.preprocess_small_datasets` to cache down DNN predictions and embeddings for LVD. 
This should finish very fast.

## 3. Cache down all experiment results
Due the (large) number of experiments we run, purely reading the results takes quite a while.
Also, there are repeated computation if we run multiple variants of LVD.
As a result, we save down all results on all datasets and baselines before reading them.
To fully replicate our experiment, you would need to run `python -m demos.run_all_experiments -t [task_id] -np [num_processes]`.
`task_id` just defines what to cache, and `num_processes` is a number of your choice (how many queues of tasks to run simultaneously).

If you have multiple GPUs to leverage (during the training and inference using LVD), you can add `-pkw gpuid`. 
As an example, if you run the tasks in 8 different queues/processes on a machine with 4 GPUs, and you turn on this option, then the queue's id mod 4 would be the GPU that queue is using.

Here are a list of the task_ids:
- 100: Run all baselines on the non-QM datasets, and the default LVD included in the main text.
- 101: Like 100, but with all variants of LVD included in Appendix. 
     Note both 101 and 100 run all the baselines, and the results won't be run twice if you have already run task_id=100.
- 200: Run the applicable baselines for QM datasets, and the default LVD included in the main text.
- 201: Like 200, but with all variants of LVD included in Appendix. 
     
## Bibtex
```
@inproceedings{NEURIPS2021_46c7cb50,
 author = {Lin, Zhen and Trivedi, Shubhendu and Sun, Jimeng},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {M. Ranzato and A. Beygelzimer and Y. Dauphin and P.S. Liang and J. Wortman Vaughan},
 pages = {8378--8391},
 publisher = {Curran Associates, Inc.},
 title = {Locally Valid and Discriminative Prediction Intervals for Deep Learning Models},
 url = {https://proceedings.neurips.cc/paper_files/paper/2021/file/46c7cb50b373877fb2f8d5c4517bb969-Paper.pdf},
 volume = {34},
 year = {2021}
}
```
