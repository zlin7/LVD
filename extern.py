import _settings
import os
import importlib.machinery
import sys
DEPENDENCY_PATHS = {'chemprop': os.path.abspath(os.path.join(_settings.__CUR_FILE_PATH, "../chemprop"))}

#chemprop for QM datasets
sys.path.append(DEPENDENCY_PATHS['chemprop'])
import chemprop
sys.path.pop(-1)

