
from tqdm.auto import tqdm
import torch
import cv2

from happyid.lit_models.losses import euclidean_dist
from happyid.data.config import *


class RetrievalPredictor:
    def __init__(self, model=None, embed_df=None, embed=None,
                 image_size=128):

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model
        self.model.to(self.device)
        self.model.eval()

        self.embed_df = embed_df
        self.embed = embed.to(self.device)
        self.image_size = image_size

    def load_batch(self, pths):
        img_list = []
        for pth in pths:
            img = cv2.imread(pth)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size, self.image_size))
            img_list.append(img)

        xb = np.stack(img_list, axis=0)
        xb = xb.astype(np.float32)
        xb = xb / 255
        xb = (xb - MEAN_IMG) / STD_IMG
        xb = torch.from_numpy(xb)
        xb = xb.permute(0, 3, 1, 2)
        xb = xb.type(torch.float32)
        return xb

    def predict(self, pths, batch_size=4):

        if isinstance(pths, str):
            pths = [pths]

        n_batch = (len(pths) - 1) // batch_size + 1
        pr_list = []

        with tqdm(total=n_batch) as pbar:
            for ib in range(n_batch):
                xb = self.load_batch(
                    pths[ib * batch_size: (ib + 1) * batch_size])

                with torch.no_grad():
                    xb = xb.to(self.device)
                    embed = self.model(xb)

                dist_mat = euclidean_dist(embed, self.embed).squeeze()
                valb, indb = dist_mat.topk(k=5, largest=False, dim=1)

                indb = indb.to('cpu')
                prb = [list(self.embed_df.iloc[ind]['individual_id'].values)
                       for ind in indb]

                pr_list.extend(prb)

                pbar.set_description(f'Batch {ib + 1}/{n_batch} processed.')
                pbar.update(1)

        return pr_list
