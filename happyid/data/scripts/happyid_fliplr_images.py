
import os
from tqdm.auto import tqdm
from IPython.display import display
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from happyid.data.config import *


df = pd.read_csv(f'{DIR_BASE}/train.csv')

# Take a fraction of the images because /kaggle/working holds only 20 GB.
vc = df.individual_id.value_counts()
weights = 1 / vc
weights = df.individual_id.map(weights.to_dict()).values
df = df.sample(n=20_000, weights=weights, random_state=42)  

existing_img_ids = list(df.image.str[:-4].values)
existing_individual_ids = list(df.individual_id.values)

dir_out = '/kaggle/working/train_images'

os.makedirs(dir_out, exist_ok=True)

img_id_list = []
individual_id_list = []


for _, row in tqdm(df.iterrows(), total=len(df)):
    img = cv2.imread(f'{DIR_BASE}/train_images/{row.image}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_fliplr = img[:, ::-1, :]

    while True:
        img_id = f'{random.randrange(16**12):014x}'
        if img_id not in existing_img_ids:
            existing_img_ids.append(img_id)
            img_id_list.append(img_id)
            cv2.imwrite(f'{dir_out}/{img_id}.jpg', img_fliplr[..., ::-1])
            break

    while True:
        individual_id = f'{random.randrange(16**12):012x}'
        if individual_id not in existing_individual_ids:
            existing_individual_ids.append(individual_id)
            individual_id_list.append(individual_id)
            break

df.rename(columns={'individual_id': 'src_individual_id',
                   'image': 'src_image'},
          inplace=True)
df['image'] = [f'{id}.jpg' for id in img_id_list]
df['individual_id'] = individual_id_list

assert set(df.src_image.values).intersection(set(df.image.values)) == set()
assert set(df.src_individual_id.values).intersection(
    set(df.individual_id.values)) == set()

df.to_csv('/kaggle/working/train.csv', index=False)

# # Display the last processed image
# img_loaded = cv2.imread(f'{dir_out}/{img_id}.jpg')
# img_loaded = cv2.cvtColor(img_loaded, cv2.COLOR_BGR2RGB)
# fig, axs = plt.subplots(figsize=(12, 6), nrows=1, ncols=2)
# axs = axs.flatten()
# axs[0].imshow(img)
# axs[1].imshow(img_loaded)
# plt.tight_layout();
# plt.show()
