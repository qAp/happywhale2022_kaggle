
import argparse
import importlib
import pytorch_lightning as pl


def import_class(s):
    module_name, class_name = s.rsplit(sep='.', maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _setup_parser():
    parser = argparse.ArgumentParser(add_help=False)
    pl.Trainer.add_argparse_args(parser)

    _add = parser.add_argument
    _add('--data_class', type=str, default='IndividualID')
    _add('--model_class', type=str, default='DOLG')
    _add('--lit_model_class', type=str, default='BaseLitModel')
    _add('--dir_out', type=str, default='training/logs')
    _add('--load_from_checkpoint', type=str, default=None)
    _add('--wandb', action='store_true', default=False)

    args, _ = parser.parse_known_args([])

    data_class = import_class(f'happyid.data.{args.data_class}')
    data_group = parser.add_argument_group('Data Args')
    data_class.add_argparse_args(data_group)

    model_class = import_class(f'happyid.models.{args.model_class}')
    model_group = parser.add_argument_group('Model Args')
    model_class.add_argparse_args(model_group)

    lit_model_class = import_class(
        f'happyid.lit_models.{args.lit_model_class}')
    lit_model_group = parser.add_argument_group('LitModel Args')
    lit_model_class.add_argparse_args(lit_model_group)

    parser.add_argument('--help', '-h', action='help')
    return parser
