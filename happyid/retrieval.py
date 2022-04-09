
from tqdm.auto import tqdm
import numpy as np, pandas as pd
import torch
from happyid.lit_models.metrics import map_per_set


def load_embedding(emb_dir, ifold):
    emb_df = pd.read_csv(f'{emb_dir}/fold{ifold}_emb.csv')
    emb = np.load(f'{emb_dir}/fold{ifold}_emb.npz')['embed']
    emb = torch.from_numpy(emb)
    return emb_df, emb


def load_ref_test_dfs(meta_data_path='./', ifold=0,
                      ref_splits=['train', 'valid', 'extra'],
                      test_splits=['test'], 
                      new_individual=True):
    dfs = [
        pd.read_csv(f'{meta_data_path}/{s}_fold{ifold}.csv')
        for s in ref_splits
        ]
    ref_df = pd.concat(dfs, axis=0, ignore_index=True)

    dfs = [
        pd.read_csv(f'{meta_data_path}/{s}_fold{ifold}.csv')
        for s in test_splits
    ]
    test_df = pd.concat(dfs, axis=0, ignore_index=True)

    if new_individual:
        is_oldid = test_df.individual_id.isin(ref_df.individual_id.unique())
        test_df.loc[~is_oldid, 'individual_id'] = 'new_individual'

    return ref_df, test_df


def get_emb_subset(emb_df, emb, subset_df):
    subset_idx = (
        subset_df
        .merge(emb_df.reset_index(), on='image', how='inner')['index']
        .to_list()
    )
    subset_emb = emb[subset_idx]
    return subset_df, subset_emb


def cosine_similarity(test_emb, ref_emb):
    assert test_emb.shape[1] == ref_emb.shape[1]

    test_norm = test_emb.norm(p='fro', dim=1, keepdim=True)
    ref_norm = ref_emb.norm(p='fro', dim=1, keepdim=True)

    test_emb /= torch.max(test_norm, 1e-10 * torch.ones_like(test_norm))
    ref_emb /= torch.max(ref_norm, 1e-10 * torch.ones_like(ref_norm))

    cosim = test_emb.mm(ref_emb.transpose(1, 0))

    return cosim


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    x_norm = x.norm(p='fro', dim=1, keepdim=True).clamp(min=1e-12)
    y_norm = y.norm(p='fro', dim=1, keepdim=True).clamp(min=1e-12)

    x /= x_norm
    y /= y_norm

    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def retrieve_topk(test_emb, ref_emb, k=50, batch_size=10_000,
                  retrieval_crit='cossim'):

    if retrieval_crit == 'cossim':
        close_func = cosine_similarity
        largest = True
    else:
        close_func = euclidean_dist
        largest = False

    values_list, indices_list = [], []
    for emb in test_emb.split(batch_size):
        close_matrix = close_func(emb, ref_emb)
        topked = torch.topk(close_matrix, k=k, dim=1, largest=largest)
        values_list += [topked.values]
        indices_list += [topked.indices]

    values = torch.cat(values_list, dim=0)
    indices = torch.cat(indices_list, dim=0)
    topked = {'values': values, 'indices': indices}
    return topked


def get_closest_ids_df(test_df, ref_df, topked, retrieval_crit='cossim'):
    '''
    Args:
        test_df (pd.DataFrame): Test image meta data.
        ref_df (pd.DataFrame): Database image meta data.
        topked (dict): Top k closest database images to each
            test image. {'values': torch.Tensor, 'indices': torch.Tensor}
        retrieval_crit (str): Retrieve by cosine similarity ('cossim')
            or euclidean distance ('distance')
    
    Returns:
        close_df (pd.DataFrame): Closest criterion values and corrsponding 
            individual_ids to test images.
    '''
    values, indices = topked['values'], topked['indices']
    assert len(indices) == len(test_df)
    num_closest = indices.shape[1]

    df_list = []
    for i, image in enumerate(test_df.image.values):
        df = pd.DataFrame(
            {'closeness': values[i].numpy(),
             'individual_id': ref_df.loc[indices[i], 'individual_id'].values,
             'image': num_closest * [image]})

        df_list.append(df)

    close_df = pd.concat(df_list, axis=0)
    grpd = close_df.groupby(['image', 'individual_id'])
    close_df = grpd.max() if retrieval_crit == 'cossim' else grpd.min()
    close_df = (
        close_df
        .reset_index()
        .sort_values('closeness', axis=0,
                     ascending=False if retrieval_crit=='cossim' else True)
    )
    return close_df


def predict_top5(close_df, newid_close_thres=.2, retrieval_crit='cossim'):
    '''
    Args:
        close_df (pd.DataFrame): Shortest distances and corrsponding 
            individual_ids to test images.
        newid_close_thres (float): Retrieval criterion threshold 
            for 'new_individual'.
    Returns:
        preds (dict): Final predicted individual_ids for 
            each test image.
    '''
    preds = {}
    for _, r in tqdm(close_df.iterrows(), total=len(close_df)):
        if r.image not in preds:
            if retrieval_crit == 'cossim':
                newid_is_closest = newid_close_thres > r.closeness
            else:
                newid_is_closest = newid_close_thres < r.closeness

            if newid_is_closest:
                preds[r.image] = ['new_individual', r.individual_id]
            else:
                preds[r.image] = [r.individual_id, 'new_individual']

        else:
            if len(preds[r.image]) == 5:
                continue
            else:
                preds[r.image].append(r.individual_id)

    return preds


def get_map5_score(test_df, preds, newid_weight=.1):
    test_df['prediction'] = test_df.image.apply(lambda x: preds[x])

    is_newid = test_df.individual_id == 'new_individual'

    newid_score = map_per_set(
        labels=test_df.loc[is_newid, 'individual_id'].to_list(),
        predictions=test_df.loc[is_newid, 'prediction'].to_list()
    )

    oldid_score = map_per_set(
        labels=test_df.loc[~is_newid, 'individual_id'].to_list(),
        predictions=test_df.loc[~is_newid, 'prediction'].to_list()
    )

    score = newid_weight * newid_score + (1 - newid_weight) * oldid_score
    return score


def retrieval_predict(test_df=None, emb=None, ref_emb_df=None, ref_emb=None,
                      newid_dist_thres=.2):

    emb = emb / emb.norm(p='fro', dim=1, keepdim=True)
    ref_emb = ref_emb / ref_emb.norm(p='fro', dim=1, keepdim=True)

    dist_matrix = euclidean_dist(emb, ref_emb)
    shortest_dist, ref_idx = dist_matrix.topk(k=50, largest=False, dim=1)
    dist_df = get_closest_ids_df(test_df, ref_df, shortest_dist, ref_idx)
    preds = final_predict(dist_df, newid_dist_thres=newid_dist_thres)

    return preds




