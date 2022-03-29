import joblib
import numpy as np



DIR_BASE = '/kaggle/input/happy-whale-and-dolphin'

MEAN_IMG = np.array([0.485, 0.456, 0.406], dtype='float32')
STD_IMG = np.array([0.229, 0.224, 0.225], dtype='float32')

NUM_FOLD = 5


