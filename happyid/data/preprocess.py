

from happyid.data.config import *

train = pd.read_csv(f'{DIR_BASE}/train.csv')
train.species.replace(
    {"globis": "short_finned_pilot_whale",
     "pilot_whale": "short_finned_pilot_whale",
     "kiler_whale": "killer_whale",
     "bottlenose_dolpin": "bottlenose_dolphin"}, 
     inplace=True)
