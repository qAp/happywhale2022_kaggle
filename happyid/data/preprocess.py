
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from happyid.data.config import *

train = pd.read_csv(f'{DIR_BASE}/train.csv')
train.species.replace(
    {"globis": "short_finned_pilot_whale",
     "pilot_whale": "short_finned_pilot_whale",
     "kiler_whale": "killer_whale",
     "bottlenose_dolpin": "bottlenose_dolphin"}, 
     inplace=True)


skf = StratifiedKFold(n_splits=3)
splits = skf.split(X=train, y=train['individual_id'])
for fold, (train_idxs, valid_idxs) in enumerate(splits):
    train.loc[valid_idxs, 'fold'] = fold

train.to_csv('/kaggle/working/train.csv')



