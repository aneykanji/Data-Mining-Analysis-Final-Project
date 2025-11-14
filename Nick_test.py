import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


import pandas as pd

metadata_path = f"{base_dir}/metadata/UrbanSound8K.csv"
metadata = pd.read_csv(metadata_path)
metadata.head()

metadata['file_path'] = metadata.apply(
    lambda row: f"{base_dir}/audio/fold{row['fold']}/{row['slice_file_name']}",
    axis=1
)


import librosa
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        audio = librosa.effects.time_stretch(audio,rate=0.5)
        audio_data = audio[:3*sr]
        audio_data = librosa.util.fix_length(audio_data,size=3*sr)
        #pre process here
        #changed the name of audio to audio_data
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

features = []
labels = []

for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
    f = extract_features(row['file_path'])
    if f is not None:
        features.append(f)
        labels.append(row['classID'])

X = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_vaildate, X_test, y_validate, y_test= (X_test,y_test,test_size=0.5, random_state=42)
