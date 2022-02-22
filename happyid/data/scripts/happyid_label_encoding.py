import os, sys
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from happyid.data.config import *


df = pd.read_csv(f'{DIR_BASE}/train.csv')
le = LabelEncoder().fit(df['individual_id'])
joblib.dump(le, '/kaggle/working/label_encoder')