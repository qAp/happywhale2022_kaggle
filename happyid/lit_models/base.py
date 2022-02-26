
import albumentations as albu
import pytorch_lightning as pl
import torch
from happyid.utils import import_class



LR = 1e-4
OPTIMIZER = 'Adam'
LOSS = 'cross_entropy'
METRIC = 'map_per_set'
ONE_CYCLE_MAX_STEPS = 20


class BaseLitModel(pl.LightningModule):

    def __init__(self, model, args=None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        loss = self.args.get('loss', LOSS)
        self.loss_fn = getattr(torch.nn.functional, loss)

        metric = self.args.get('metric', METRIC)
        self.metric_fn = import_class(f'happyid.lit_models.metrics.{metric}')

        self.lr = self.args.get('lr', LR)
        optimizer = self.args.get('optimizer', OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, optimizer)
        self.one_cycle_max_lr = self.args.get('one_cycle_max_lr', None)
        self.one_cycle_max_steps = self.args.get(
            'one_cycle_max_steps', ONE_CYCLE_MAX_STEPS)

    @staticmethod
    def add_argparse_args(parser):
        add = parser.add_argument
        add('--loss', type=str, default=LOSS)
        add('--metric', type=str, default=METRIC)
        add('--lr', type=float, default=LR)
        add('--optimizer', type=str, default=OPTIMIZER)
        add('--one_cycle_max_lr', type=float, default=None)
        add('--one_cycle_max_steps', type=float, default=ONE_CYCLE_MAX_STEPS)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = self.optimizer_class(params=self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return {'optimizer': optimizer}
        else:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.one_cycle_max_lr,
                max_steps=self.one_cycle_max_steps)
            return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}

    def training_step(self, batch, batch_idx):
        xb, yb = batch
        xb = xb.permute(0, 3, 1, 2)

        logits = self(xb)

        loss = self.loss_fn(logits, yb.squeeze())

        pred_batch = logits.data.topk(k=5, dim=1, largest=True).values
        pred_batch = [[p for p in pred] for pred in pred_batch.numpy()]
        metric = self.metric_fn(labels=list(
            yb.squeeze().numpy()), predictions=pred_batch)

        self.log('train_loss', loss)
        self.log('train_metric', metric)

        return loss

    def validation_step(self, batch, batch_idx):
        xb, yb = batch
        xb = xb.permute(0, 3, 1, 2)

        logits = self(xb)

        loss = self.loss_fn(logits, yb.squeeze())

        pred_batch = logits.data.topk(k=5, dim=1, largest=True).values
        pred_batch = [[p for p in pred] for pred in pred_batch.numpy()]
        metric = self.metric_fn(labels=list(
            yb.squeeze().numpy()), predictions=pred_batch)

        self.log('valid_loss', loss)
        self.log('valid_metric', metric)