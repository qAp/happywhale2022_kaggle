import joblib
import numpy as np



DIR_BASE = '/kaggle/input/happy-whale-and-dolphin'

NUM_INDIVIDUALS = 15587

MEAN_IMG = np.array([0.485, 0.456, 0.406], dtype='float32')
STD_IMG = np.array([0.229, 0.224, 0.225], dtype='float32')

ID_ENCODER = joblib.load('/kaggle/input/happyid-label-encoding/label_encoder')


