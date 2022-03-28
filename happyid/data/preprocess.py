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


def make_tvet_meta(min_id_vc=40, valid_pct=.1):
    '''
    TVET := train, valid, extra, text.
    
    This splits the samples in df into 4 splits: train, valid, extra, and test.
    Also fits label encoder on the train split.
    All splits and label encoder are saved to /kaggle/working.
    
    Args:
        df (pd.DataFrame): meta data
        min_id_cv (int): Minimum individual_id value count above which the individual_id
            will be included during training (the union of train and valid splits).
        valid_pct (float): fraction of the union of train and valid splits that the valid
            split takes up.
    '''
    df = pd.read_csv(f'{DIR_BASE}/train.csv')

    kf = StratifiedKFold(n_splits=5)

    folds_summary = []
    for fold, (bulk_idx, test_idx) in enumerate(kf.split(X=df, y=df.individual_id)):

        bulk_df = df.iloc[bulk_idx]
        test_df = df.iloc[test_idx]

        vc = bulk_df.individual_id.value_counts()
        id_in_subset = vc > min_id_vc
        image_in_subset = bulk_df.individual_id.isin(vc.index[id_in_subset])

        subset_df = bulk_df[image_in_subset]
        extra_df = bulk_df[~image_in_subset]

        valid_df = subset_df.sample(frac=valid_pct, replace=False)
        train_df = subset_df[~subset_df.index.isin(valid_df.index)]

        label_encoder = LabelEncoder()
        label_encoder.fit(train_df.individual_id)

        train_df.to_csv(f'/kaggle/working/train_fold{fold}.csv', index=False)
        valid_df.to_csv(f'/kaggle/working/valid_fold{fold}.csv', index=False)
        extra_df.to_csv(f'/kaggle/working/extra_fold{fold}.csv', index=False)
        test_df.to_csv(f'/kaggle/working/test_fold{fold}.csv', index=False)
        joblib.dump(label_encoder, f'/kaggle/working/label_encoder_fold{fold}')

        summary = pd.concat(
            [train_df.describe(),
             valid_df.describe(),
             extra_df.describe(),
             test_df.describe()],
            axis=1, keys=['train', 'valid', 'extra', 'test']
        )
        folds_summary.append(summary)

    folds_summary = pd.concat(folds_summary, axis=0, keys=range(5))
    display(folds_summary)


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










