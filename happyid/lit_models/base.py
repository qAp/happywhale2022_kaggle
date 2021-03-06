import ast
import albumentations as albu
import pytorch_lightning as pl
import torch
from happyid.utils import import_class



LR = 1e-4
OPTIMIZER = 'Adam'
LOSS = 'ArcFaceLoss'
METRIC = 'mean_average_precision_5'
ONE_CYCLE_MAX_STEPS = 20
RETURN_EMB = 'True'


class BaseLitModel(pl.LightningModule):

    def __init__(self, model, args=None):
        super().__init__()
        self.model = model
        self.args = vars(args) if args is not None else {}

        loss = self.args.get('loss', LOSS)
        if loss not in ('ArcLoss1', ):
            loss_class = import_class(f'happyid.lit_models.losses.{loss}')
            self.loss_fn = loss_class()

        metric = self.args.get('metric', METRIC)
        self.metric_fn = import_class(f'happyid.lit_models.metrics.{metric}')

        self.lr = self.args.get('lr', LR)
        self.optimizer = self.args.get('optimizer', OPTIMIZER)
        self.optimizer_class = getattr(torch.optim, self.optimizer)
        self.one_cycle_max_lr = self.args.get('one_cycle_max_lr', None)
        self.one_cycle_max_steps = self.model.data_config.get(
            'total_steps', ONE_CYCLE_MAX_STEPS)

        self.return_emb = self.args.get(
            'return_emb', ast.literal_eval(RETURN_EMB))

    @staticmethod
    def add_argparse_args(parser):
        add = parser.add_argument
        add('--loss', type=str, default=LOSS)
        add('--metric', type=str, default=METRIC)
        add('--lr', type=float, default=LR)
        add('--optimizer', type=str, default=OPTIMIZER)
        add('--one_cycle_max_lr', type=float, default=None)
        add('--one_cycle_max_steps', type=int, default=ONE_CYCLE_MAX_STEPS)
        add('--return_emb', type=ast.literal_eval, default=RETURN_EMB)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        print('!@#$!#$!#$!@!@!@@$@#F!$ Configure Optimizer')
        print('###########################################')
        params = (p for p in self.parameters() if p.requires_grad == True)
        optimizer = self.optimizer_class(params=params, lr=self.lr)
        if self.one_cycle_max_lr is None:
            return {'optimizer': optimizer}
        else:
            print('self.one_cycle_max_lr', self.one_cycle_max_lr)
            print('self.one_cycle_max_steps', self.one_cycle_max_steps)
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=self.one_cycle_max_lr,
                total_steps=self.one_cycle_max_steps)
                
            return {'optimizer': optimizer, 
                    'lr_scheduler': 
                        {'scheduler': lr_scheduler, 'interval': 'step'}
                    }

    def training_step(self, batch, batch_idx):
        xb, yb = batch
        xb = xb.permute(0, 3, 1, 2)

        logits = self(xb)

        loss = self.loss_fn(logits, yb.squeeze())

        self.log('train_loss', loss,
                 on_step=True, on_epoch=False, prog_bar=True)
                 
        return loss

    def validation_step(self, batch, batch_idx):
        xb, yb = batch
        xb = xb.permute(0, 3, 1, 2)

        logits = self(xb)

        loss = self.loss_fn(logits, yb.squeeze())

        self.log('valid_loss', loss,
                 on_step=False, on_epoch=True, prog_bar=True)

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        xb, *_ = batch
        xb = xb.permute(0, 3, 1, 2)
        return self.model(xb, return_emb=self.return_emb)



