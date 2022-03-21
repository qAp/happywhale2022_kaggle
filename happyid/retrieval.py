

from tqdm.auto import tqdm
from happyid.lit_models.losses import euclidean_dist



def retrieval_predict(pred_emb=None, ref_emb=None, ref_emb_df=None,
                      batch_size=64):

    num_batch = (len(pred_emb) - 1) // batch_size + 1

    pbar = tqdm(range(num_batch), total=num_batch)
    predictions = []
    for i in pbar:
        p = pred_emb[i * batch_size: (i + 1) * batch_size]
        dist_mat = euclidean_dist(p, ref_emb).squeeze()
        vals, idxs = dist_mat.topk(k=5, largest=False, dim=1)
        predictions.extend(
            [list(ref_emb_df.iloc[idx]['individual_id'].values)
             for idx in idxs]
        )

    return predictions
