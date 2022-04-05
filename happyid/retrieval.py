

from tqdm.auto import tqdm
import numpy as np, pandas as pd
import torch
from happyid.lit_models.losses import euclidean_dist


def get_closest_ids_df(test_df, ref_emb_df, shortest_dist, ref_idx):
    '''
    Args:
        test_df (pd.DataFrame): Test image meta data.
        ref_emb_df (pd.DataFrame): Database image meta data.
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
             'individual_id': ref_emb_df.loc[ref_idx[i], 'individual_id'].values,
             'image': num_closest * [image]})

        df_list.append(df)

    dist_df = pd.concat(df_list, axis=0)
    dist_df = dist_df.groupby(['image', 'individual_id']).min().reset_index()
    dist_df.sort_values('distance', axis=0, inplace=True)
    return dist_df


def predict_top5(dist_df, newid_dist_thres=.2):
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


def retrieval_predict(test_df=None, emb=None, ref_emb_df=None, ref_emb=None,
                      newid_dist_thres=.2):

    emb = emb / emb.norm(p='fro', dim=1, keepdim=True)
    ref_emb = ref_emb / ref_emb.norm(p='fro', dim=1, keepdim=True)

    dist_matrix = euclidean_dist(emb, ref_emb)
    shortest_dist, ref_idx = dist_matrix.topk(k=50, largest=False, dim=1)
    dist_df = get_closest_ids_df(test_df, ref_emb_df, shortest_dist, ref_idx)
    preds = final_predict(dist_df, newid_dist_thres=newid_dist_thres)

    return preds




