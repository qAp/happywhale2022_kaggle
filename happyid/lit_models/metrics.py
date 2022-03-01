

import numpy as np
import torch


def map_per_image(label, predictions):
    """Computes the precision score of one image.

    Parameters
    ----------
    label : string
            The true label of the image
    predictions : list
            A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    try:
        return 1 / (predictions[:5].index(label) + 1)
    except ValueError:
        return 0.0


def map_per_set(labels, predictions):
    """Computes the average over multiple images.

    Parameters
    ----------
    labels : list
             A list of the true labels. (Only one true label per images allowed!)
    predictions : list of list
             A list of predicted elements (order does matter, 5 predictions allowed per image)

    Returns
    -------
    score : double
    """
    return np.mean([map_per_image(l, p) for l, p in zip(labels, predictions)])


def mean_average_precision_5(labels=None, predictions=None):
    '''
    Mean average Precision @ 5

    Args:
        labels (torch.tensor) [batch_size, 1]: Labels (targets)
        predictions (torch.tensor) [batch_size, 5]: Top 5 predictions
    '''
    scores = torch.zeros_like(labels, dtype=torch.float32)
    i_sample, i_k = torch.where(labels == predictions)

    if len(i_sample) != 0:
        scores[i_sample] = 1 / (i_k.reshape(scores[i_sample].shape) + 1)

    return scores.mean().item()


def test():
    assert map_per_image('x', []) == 0.0
    assert map_per_image('x', ['y']) == 0.0
    assert map_per_image('x', ['x']) == 1.0
    assert map_per_image('x', ['x', 'y', 'z']) == 1.0
    assert map_per_image('x', ['y', 'x']) == 0.5
    assert map_per_image('x', ['y', 'x', 'x']) == 0.5
    assert map_per_image('x', ['y', 'z']) == 0.0
    assert map_per_image('x', ['y', 'z', 'x']) == 1/3
    assert map_per_image('x', ['y', 'z', 'a', 'b', 'c']) == 0.0
    assert map_per_image('x', ['x', 'z', 'a', 'b', 'c']) == 1.0
    assert map_per_image('x', ['y', 'z', 'a', 'b', 'x']) == 1/5
    assert map_per_image('x', ['y', 'z', 'a', 'b', 'c', 'x']) == 0.0

    assert map_per_set(['x'], [['x', 'y']]) == 1.0
    assert map_per_set(['x', 'z'], [['x', 'y'], ['x', 'y']]) == 1/2
    assert map_per_set(['x', 'z'], [['x', 'y'], ['x', 'y', 'z']]) == 2/3
    assert map_per_set(['x', 'z', 'k'], [['x', 'y'], ['x', 'y', 'z'], 
                       ['a', 'b', 'c', 'd', 'e']]) == 4/9


if __name__ == '__main__':
    test()
    
