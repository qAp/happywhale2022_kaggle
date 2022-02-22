
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from happyid.data.config import *
from happyid.data.preprocess import correct_species_names, make_cv_folds



train = pd.read_csv(f'{DIR_BASE}/train.csv')
train['dir_img'] = f'{DIR_BASE}/train_images'
correct_species_names(train)

train.to_csv('/kaggle/working/train.csv', index=False)

make_cv_folds(train, n_splits=5, random_state=42)
