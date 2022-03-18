
import os
import argparse
import pandas as pd
from happyid.data.config import * 
from happyid.utils import import_class
from happyid.lit_models import BaseLitModel



def _setup_parser():
    parser = argparse.ArgumentParser()
    _add = parser.add_argument
    _add('--predictor_class', type=str, default='SoftmaxPredictor')
    _add('--checkpoint_path', type=str, nargs='+', default='best.pth')
    _add('--model_class', type=str, nargs='+', default='DOLG')
    _add('--pretrained', type=bool, default=False)
    _add('--image_size', type=int, default=128)
    _add('--batch_size', type=int, default=512)
    _add('--infer_subset', type=int, default=None, 
         help='Number of samples to infer on. (for debugging)')

    return parser


def main():

    parser = _setup_parser()
    args = parser.parse_args()

    assert len(args.checkpoint_path) == len(args.model_class)
    for p in args.checkpoint_path:
        assert os.path.exists(p)

    models = []
    for p, model_class in zip(args.checkpoint_path, args.model_class):

        model_class = import_class(f'happyid.models.{model_class}')
        model = model_class(args=args)
        lit_model = BaseLitModel.load_from_checkpoint(
            checkpoint_path=p, model=model, args=args)

        models.append(lit_model)

    predictor_class = import_class(f'happyid.{args.predictor_class}')
    predictor = predictor_class(models=models, image_size=args.image_size)

    df = pd.read_csv(f'{DIR_BASE}/sample_submission.csv')
    test_image_paths = (f'{DIR_BASE}/test_images/' + df['image']).to_list()

    if args.infer_subset is None:
        select_index = df.index.values
    else:
        select_index = np.random.permutation(len(df))[:args.infer_subset]

    pred_list = predictor.predict(pths=test_image_paths[select_index],
                                  batch_size=args.batch_size)
    pred_list = [' '.join(pred) for pred in pred_list]

    df.loc[select_index, 'predictions'] = pred_list
    df.to_csv('/kaggle/working/submission.csv', index=False)


if __name__ == '__main__':
    main()
