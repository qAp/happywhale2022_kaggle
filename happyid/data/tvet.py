
import joblib
import albumentations as albu
import pandas as pd
import torch
from happyid.data import IndividualID
from happyid.data.individual_id import IndividualIDDataset


class TvetRetrievalCVIndividual(IndividualID):
    def setup(self):
        train_df = pd.read_csv(
            f'{self.meta_data_path}/train_fold{self.fold}.csv'
            )
        valid_df = pd.read_csv(
            f'{self.meta_data_path}/valid_fold{self.fold}.csv'
            )
        extra_df = pd.read_csv(
            f'{self.meta_data_path}/extra_fold{self.fold}.csv'
        )
        test_df = pd.read_csv(
            f'{self.meta_data_path}/test_fold{self.fold}.csv'
        ).iloc[:200]

        ref_df = pd.concat([train_df, valid_df, extra_df], axis=0)
        ref_id_set = ref_df.individual_id.unique()

        is_known_id = test_df.individual_id.isin(ref_id_set)
        test_df.loc[~is_known_id, 'individual_id'] = 'new_individual'
        print(test_df.describe())

        # Training set is only needed to get num_class to create model
        id_encoder = joblib.load(self.id_encoder_path)
        if self.image_dir is not None:
            train_df['dir_img'] = self.image_dir
        else:
            assert 'dir_img' in train_df
        self.train_ds = IndividualIDDataset(
            train_df,
            transform=albu.Compose(self.train_tfms),
            id_encoder=id_encoder
        )

        if self.image_dir is not None:
            test_df['dir_img'] = self.image_dir
        else:
            assert 'dir_img' in test_df
        self.valid_ds = IndividualIDDataset(
            test_df, 
            transform=albu.Compose(self.valid_tfms),
            id_encoder=id_encoder)

        self.test_ds = None

    def config(self):
        return {'num_class': len(self.valid_ds.id_encoder.classes_)}


class TvetEmbeddedIndividual(IndividualID):
    def setup(self):
        train_df = pd.read_csv(
            f'{self.meta_data_path}/train_fold{self.fold}.csv'
        )
        valid_df = pd.read_csv(
            f'{self.meta_data_path}/valid_fold{self.fold}.csv'
        )
        extra_df = pd.read_csv(
            f'{self.meta_data_path}/extra_fold{self.fold}.csv'
        )
        ref_df = pd.concat([train_df, valid_df, extra_df], axis=0)
        if self.image_dir is not None:
            ref_df['dir_img'] = self.image_dir
        else:
            assert 'dir_img' in ref_df

        self.train_ds = None
        self.valid_ds = None

        id_encoder = joblib.load(self.id_encoder_path)
        self.test_ds = IndividualIDDataset(
            ref_df, 
            transform=albu.Compose(self.test_tfms), 
            id_encoder=id_encoder)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu)

    def config(self):
        return {'num_class': len(self.test_ds.id_encoder.classes_)}
