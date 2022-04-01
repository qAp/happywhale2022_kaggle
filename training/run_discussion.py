
'''
https://www.kaggle.com/code/clemchris/pytorch-backfin-convnext-arcface
'''

from happyid.models.discussion import ArcMarginProduct
from happyid.lit_models.discussion import LitModule


def train(
    train_csv_encoded_folded: str = str(TRAIN_CSV_ENCODED_FOLDED_PATH),
    test_csv: str = str(TEST_CSV_PATH),
    val_fold: float = 0.0,
    image_size: int = 256,
    batch_size: int = 64,
    num_workers: int = 2,
    model_name: str = "tf_efficientnet_b0",
    pretrained: bool = True,
    drop_rate: float = 0.0,
    embedding_size: int = 512,
    num_classes: int = 15587,
    arc_s: float = 30.0,
    arc_m: float = 0.5,
    arc_easy_margin: bool = False,
    arc_ls_eps: float = 0.0,
    optimizer: str = "adam",
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-6,
    checkpoints_dir: str = str(CHECKPOINTS_DIR),
    accumulate_grad_batches: int = 1,
    auto_lr_find: bool = False,
    auto_scale_batch_size: bool = False,
    fast_dev_run: bool = False,
    gpus: int = 1,
    max_epochs: int = 10,
    precision: int = 16,
    stochastic_weight_avg: bool = True,
):
    pl.seed_everything(42)

    datamodule = LitDataModule(
        train_csv_encoded_folded=train_csv_encoded_folded,
        test_csv=test_csv,
        val_fold=val_fold,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    datamodule.setup()
    len_train_dl = len(datamodule.train_dataloader())

    module = LitModule(
        model_name=model_name,
        pretrained=pretrained,
        drop_rate=drop_rate,
        embedding_size=embedding_size,
        num_classes=num_classes,
        arc_s=arc_s,
        arc_m=arc_m,
        arc_easy_margin=arc_easy_margin,
        arc_ls_eps=arc_ls_eps,
        optimizer=optimizer,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        len_train_dl=len_train_dl,
        epochs=max_epochs
    )

    model_checkpoint = ModelCheckpoint(
        checkpoints_dir,
        filename=f"{model_name}_{image_size}",
        monitor="val_loss",
    )

    trainer = pl.Trainer(
        accumulate_grad_batches=accumulate_grad_batches,
        auto_lr_find=auto_lr_find,
        auto_scale_batch_size=auto_scale_batch_size,
        benchmark=True,
        callbacks=[model_checkpoint],
        deterministic=True,
        fast_dev_run=fast_dev_run,
        gpus=gpus,
        max_epochs=2 if DEBUG else max_epochs,
        precision=precision,
        stochastic_weight_avg=stochastic_weight_avg,
        limit_train_batches=0.1 if DEBUG else 1.0,
        limit_val_batches=0.1 if DEBUG else 1.0,
    )

    trainer.tune(module, datamodule=datamodule)

    trainer.fit(module, datamodule=datamodule)


if __name__ == '__main__':

    train_df = pd.read_csv(TRAIN_CSV_PATH)

    train_df["image_path"] = train_df["image"].apply(get_image_path, dir=TRAIN_DIR)

    encoder = LabelEncoder()
    train_df["individual_id"] = encoder.fit_transform(train_df["individual_id"])
    np.save(ENCODER_CLASSES_PATH, encoder.classes_)

    skf = StratifiedKFold(n_splits=N_SPLITS)
    for fold, (_, val_) in enumerate(skf.split(X=train_df, y=train_df.individual_id)):
        train_df.loc[val_, "kfold"] = fold

    train_df.to_csv(TRAIN_CSV_ENCODED_FOLDED_PATH, index=False)

    train_df.head()

    # Use sample submission csv as template
    test_df = pd.read_csv(SAMPLE_SUBMISSION_CSV_PATH)
    test_df["image_path"] = test_df["image"].apply(get_image_path, dir=TEST_DIR)

    test_df.drop(columns=["predictions"], inplace=True)

    # Dummy id
    test_df["individual_id"] = 0

    test_df.to_csv(TEST_CSV_PATH, index=False)

    test_df.head()

    # Train
    model_name = "convnext_small"
    image_size = 384
    batch_size = 32
    train(
        model_name=model_name,
        image_size=image_size,
        batch_size=batch_size
    )
