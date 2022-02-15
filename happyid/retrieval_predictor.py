
import torch
import cv2

from happyid.lit_models.losses import euclidean_dist
from happyid.data.config import *

class RetrievalPredictor:
    def __init__(self, model=None, embed_df=None, embed=None,
                 image_size=128):

        self.model = model
        self.embed_df = embed_df
        self.embed = embed
        self.image_size = image_size

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self.model.eval()

    def predict(self, fn):
        img = cv2.imread(fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32)
        img = img / 255
        img = (img - MEAN_IMG) / STD_IMG
        xb = torch.from_numpy(img)
        xb = xb.permute(2, 0, 1)
        xb = xb.type(torch.float32)
        xb = xb[None, ...]

        with torch.no_grad():
            xb = xb.to(self.device)
            embed = self.model(xb)

        dist_mat = euclidean_dist(embed, self.embed).squeeze()
        val, ind = dist_mat.topk(k=5, largest=False)
        pr = self.embed_df.iloc[ind]['individual_id']

        return list(pr.values)
