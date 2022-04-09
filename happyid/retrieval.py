
from tqdm.auto import tqdm
import numpy as np, pandas as pd
import torch
from happyid.lit_models.losses import euclidean_dist
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


def get_closest_ids_df(test_df, ref_df, shortest_dist, ref_idx):
    '''
    Args:
        test_df (pd.DataFrame): Test image meta data.
        ref_df (pd.DataFrame): Database image meta data.
        shortest_dist (torch.Tensor): Distances of k closest 
            database images to each test image.
        ref_idx (torch.Tensor): Indices of database images with shortest k
            distances to each test image.
    
    Returns:
        dist_df (pd.DataFrame): Shortest distances and corrsponding 
            individual_ids to test images.
    '''
    assert len(shortest_dist) == len(ref_idx) == len(test_df)
    assert shortest_dist.shape[1] == ref_idx.shape[1]
    num_closest = shortest_dist.shape[1]

    df_list = []
    for i, image in enumerate(test_df.image.values):
        df = pd.DataFrame(
            {'distance': shortest_dist[i].numpy(),
             'individual_id': ref_df.loc[ref_idx[i], 'individual_id'].values,
             'image': num_closest * [image]})

        df_list.append(df)

    dist_df = pd.concat(df_list, axis=0)
    dist_df = dist_df.groupby(['image', 'individual_id']).min().reset_index()
    dist_df.sort_values('distance', axis=0, inplace=True)
    return dist_df


def distance_predict_top5(dist_df, newid_dist_thres=.2):
    '''
    Args:
        dist_df (pd.DataFrame): Shortest distances and corrsponding 
            individual_ids to test images.
        newid_dist_thres (float): Distance threshold for 'new_individual'.
    Returns:
        preds (dict): Final predicted individual_ids for 
            each test image.
    '''
    preds = {}
    for _, r in tqdm(dist_df.iterrows(), total=len(dist_df)):
        if r.image not in preds:
            if r.distance > newid_dist_thres:
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




