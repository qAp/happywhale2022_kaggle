
import pandas as pd
import torch
import cv2
from sklearn.model_selection import StratifiedKFold
from happyid.data.config import *



def correct_species_names(train):
    train.species.replace(
        {"globis": "short_finned_pilot_whale",
        "pilot_whale": "short_finned_pilot_whale",
        "kiler_whale": "killer_whale",
        "bottlenose_dolpin": "bottlenose_dolphin"}, 
        inplace=True)

def create_cv_folds(train):
    skf = StratifiedKFold(n_splits=3)
    splits = skf.split(X=train, y=train['individual_id'])
    for fold, (train_idxs, valid_idxs) in enumerate(splits):
        train.loc[valid_idxs, 'fold'] = fold

@torch.no_grad()
def image_embedding(model, jpg='image.jpg', image_size=128):
    model.eval()

    img = cv2.imread(jpg)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(src=img, dsize=(image_size, image_size))
    img = img.astype(np.float32)
    img = img / 255
    img = (img - MEAN_IMG[None, None, :]) / STD_IMG[None, None, :]

    xb = torch.from_numpy(img)
    xb = xb.permute(2, 0, 1)
    xb = xb[None, ...]
    xb = xb.type(torch.float32)

    logits = model(xb)

    return logits[0]



if __name__ == '__name__':
    train = pd.read_csv(f'{DIR_BASE}/train.csv')
    train.to_csv('/kaggle/working/train.csv')



