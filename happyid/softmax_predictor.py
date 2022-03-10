
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import cv2
from happyid.data.config import *





class SoftmaxPredictor:
    def __init__(self, models=None, image_size=128):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        if not isinstance(models, list):
            models = [models]
        self.models = models
        for m in self.models:
            m.to(self.device)
            m.eval()

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

    @torch.no_grad()
    def predict(self, pths, batch_size=4):
        if isinstance(pths, str):
            pths = [pths]

        n_batch = (len(pths) - 1) // batch_size + 1
        pred_list = []
        with tqdm(total=n_batch) as pbar:
            for ib in range(n_batch):
                xb = self.load_batch(
                    pths[ib * batch_size: (ib + 1) * batch_size])
                xb = xb.to(self.device)

                yb_list = []
                for m in self.models:
                    yb = m(xb)
                    yb_list.append(yb)
                yb = torch.stack(yb_list, dim=0).mean(dim=0)

                probs = F.softmax(yb, dim=1)
                valb, indb = probs.topk(k=5, largest=True, dim=1)

                indb = indb.to('cpu').numpy()
                predb = [list(ID_ENCODER.inverse_transform(ind))
                         for ind in indb]
                pred_list.extend(predb)

                pbar.set_description(f'Batch {ib + 1}/{n_batch} processed.')
                pbar.update(1)

        return pred_list
