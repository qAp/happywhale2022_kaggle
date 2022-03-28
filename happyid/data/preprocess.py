from IPython.display import display
import pandas as pd
import torch
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from happyid.data.config import *



def correct_species_names(train):
    train.species.replace(
        {"globis": "short_finned_pilot_whale",
        "pilot_whale": "short_finned_pilot_whale",
        "kiler_whale": "killer_whale",
        "bottlenose_dolpin": "bottlenose_dolphin"}, 
        inplace=True)


def make_cv_folds(df, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state)
    splits = skf.split(X=df, y=df['individual_id'])

    for fold, (train_index, valid_index) in enumerate(splits):
        print(
            f'Fold {fold} Train size {len(train_index)} Valid size {len(valid_index)}')

        df_train = df.loc[train_index].reset_index()
        df_valid = df.loc[valid_index].reset_index()

        df_train = df_train.rename(columns={'index': 'idx_embed'})
        df_valid = df_valid.rename(columns={'index': 'idx_embed'})

        df_train.to_csv(f'/kaggle/working/train_fold{fold}.csv', index=False)
        df_valid.to_csv(f'/kaggle/working/valid_fold{fold}.csv', index=False)

    display(
        pd.concat([df_train['individual_id'].value_counts(),
                   df_valid['individual_id'].value_counts()],
                  axis=1, keys=['train', 'valid'])
    )


def make_train_subset_meta():
    df = pd.read_csv(f'{DIR_BASE}/train.csv')

    vc = df.individual_id.value_counts()

    min_vc = 50

    id_isin_subset = vc > min_vc

    print('Original number of ids:', len(vc))
    print('Subset number of ids:', id_isin_subset.sum())

    image_isin_subset = df.individual_id.isin(vc.index[id_isin_subset])

    print('Original number of images:', len(df))
    print('Subset number of images:', image_isin_subset.sum())

    subset_df = df[image_isin_subset].reset_index(drop=True)

    label_encoder = LabelEncoder()
    label_encoder.fit(subset_df.individual_id.values)
    joblib.dump(label_encoder, '/kaggle/working/label_encoder')

    make_cv_folds(subset_df, n_splits=5, random_state=None)


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










