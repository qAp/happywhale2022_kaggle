
import ast
from happyid.lit_models import BaseLitModel
from happyid.lit_models.losses import ArcLoss1
from happyid.models.dolg import ArcMarginProduct_subcenter


ARC_S = 30.
ARC_M = .5
ARC_EASY_MARGIN = 'False'
ARC_LS_EPS = 0.


class ArcLoss1LitModel(BaseLitModel):
    def __init__(self, model, args=None):
        super().__init__(model, args)

        self.arc_s = self.args.get('arc_s', ARC_S)
        self.arc_m = self.args.get('arc_m', ARC_M)
        self.arc_easy_margin = self.args.get('arc_easy_margin', ARC_EASY_MARGIN)
        self.arc_ls_eps = self.args.get('arc_ls_eps', ARC_LS_EPS)

        self.loss_fn = ArcLoss1(
            in_features=self.model.embedding_size,
            out_features=self.model.n_classes,
            s=self.arc_s,
            m=self.arc_m,
            easy_margin=self.arc_easy_margin,
            ls_eps=self.arc_ls_eps)

    @staticmethod
    def add_argparse_args(parser):
        BaseLitModel.add_argparse_args(parser)
        _add = parser.add_argument
        _add('--arc_s', type=float, default=ARC_S)
        _add('--arc_m', type=float, default=ARC_M)
        _add('--arc_easy_margin', type=ast.literal_eval, 
             default=ARC_EASY_MARGIN)
        _add('--arc_ls_eps', type=float, default=ARC_LS_EPS)

    
