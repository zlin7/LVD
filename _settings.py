import os
import getpass
import sys
DATA_PATH = "Z:/Data" #Local
if not os.path.isdir(DATA_PATH): DATA_PATH = "/srv/local/data" #Workstation 1
if not os.path.isdir(DATA_PATH): DATA_PATH = "/srv/home/%s/data"%(getpass.getuser()) #Workstations and servers

__CUR_FILE_PATH = os.path.dirname(os.path.abspath(__file__))


#==============================Data Related
SYNT_DJKP = "DJKPSynthetic"
QM8_NAME = "QM8"
QM9_NAME = "QM9"
YACHT_NAME = "UCI_Yacht"
KIN8NM_NAME = 'Kin8nm'
ENERGY_NAME = "UCI_Energy"
BIKE_NAME = "UCI_BikeSharing"
HOUSING_NAME = 'BostonHousing'
CONCRETE_NAME = "UCI_Concrete"
SMALL_DATASETS = [BIKE_NAME, KIN8NM_NAME, YACHT_NAME, ENERGY_NAME, HOUSING_NAME, CONCRETE_NAME]
QM_DATASETS = {QM8_NAME: 16, QM9_NAME: 12}

ENERGY_PATH = os.path.join(DATA_PATH, ENERGY_NAME)
QM8_PATH = os.path.join(DATA_PATH, QM8_NAME)
QM9_PATH = os.path.join(DATA_PATH, QM9_NAME)
YACHT_PATH = os.path.join(DATA_PATH, YACHT_NAME)
KIN8NM_PATH = os.path.join(DATA_PATH, KIN8NM_NAME)
CONCRETE_PATH = os.path.join(DATA_PATH, CONCRETE_NAME)
BIKE_PATH = os.path.join(DATA_PATH, BIKE_NAME)

WORKSPACE = os.path.join(__CUR_FILE_PATH, "Temp")# r"Z:\gitRes\AEJKP\Temp"
_PERSIST_PATH = os.path.join(WORKSPACE, 'cache')

LOG_OUTPUT_DIR = os.path.join(WORKSPACE, 'logs')
RANDOM_SEED = 7

NCOLS = 80

import torch
import numpy as np
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


#====================Paper Related
METHOD_NAME = "LVD"
METHOD_PLACEHOLDER = 'Method'