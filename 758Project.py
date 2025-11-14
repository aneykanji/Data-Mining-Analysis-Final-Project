import soundata
import librosa
import librosa.display
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

# I think using Google Drive is the most reliable way 
# to download our dataset in Google Colab bcz 
# i t avoids repeated downloads, prevents data loss when 
# sessions disconnect, and also the Professor can easily run the file
# -----------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

import os
dataset_path = "/content/drive/MyDrive/UrbanSound8K"
os.makedirs(dataset_path, exist_ok=True)


# The line below downloads the UrbanSound8K dataset directly 
# from Zenodo into your Google Drive. This URL points to the 
# official dataset source, and you can replace it with a 
# different download link if needed.
# ------------------------------------------------------------
!wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz -O /content/drive/MyDrive/UrbanSound8K/UrbanSound8K.tar.gz

# This extracts the downloaded tar.gz file into your Drive. 
# You can change the extraction path depending on where you 
# want to keep your dataset.
# ------------------------------------------------------------
!tar -xzf /content/drive/MyDrive/UrbanSound8K/UrbanSound8K.tar.gz -C /content/drive/MyDrive/

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


# This line creates a new column 'file_path' in the metadata 
# dataFrame. For each row, it builds the complete path to the 
# corresponding audio file using:
#    base_dir: the root location of UrbanSound8K
#   fold: the subfolder for that audio file
#   slice_file_name: the filename of the audio clip
# This ensures every audio file can be easily loaded later.
# ------------------------------------------------------------
metadata['file_path'] = metadata.apply(
    lambda row: f"{base_dir}/audio/fold{row['fold']}/{row['slice_file_name']}",
    axis=1
)



# The function extract features takes the path to an audio file and performs:
#   1. Load the audio using librosa.
#   2. Apply time-stretching (rate=0.5 slows audio down).
#   3. Extract exactly 3 seconds of audio (padding/trimming).
#   4. Compute MFCCs (Mel-Frequency Cepstral Coefficients),
#      which are commonly used audio features.
#   5. Return the mean MFCC vector for the clip.
#
# If any file fails to load or process, it prints an error and
# returns None so the loop can safely continue.
# ------------------------------------------------------------

import librosa
import numpy as np
from tqdm import tqdm

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=None)
        audio = librosa.effects.time_stretch(audio,rate=0.5)
        audio_data = audio[:3*sr]
        audio_data = librosa.util.fix_length(audio_data,size=3*sr)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

features = []
labels = []


# We iterate through every row in the metadata DataFrame.
# For each audio file:
#   Call extract_features()
#   If features are successfully extracted, store them
#   Save the corresponding class label

for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
    f = extract_features(row['file_path'])
    if f is not None:
        features.append(f)
        labels.append(row['classID'])

X = np.array(features) #inputs
y = np.array(labels)# targets



dataset = soundata.initialize('urbansound8k', data_home='/home/nckmrtn25/urban_dataset')
#Add your dataset lines here and comment out the above line when you are testing
ids = dataset.clip_ids
clips = dataset.load_clips()


#Gets the data labels
data_labels = []
for i in range(len(clips)):

    example_clip = clips[ids[i]]
    example_tags = example_clip.tags
    data_labels.append(example_tags.labels)

#Gets the audio data path
audio_paths = []
for key, clip in dataset.load_clips().items():
    audio_paths.append(clip.audio_path)
    #print(clip.audio_path)
    
    
#Classes and corresponding numbers (can be changed if needed)
label_dict = {
    "air_conditioner": 0,
    "car_horn": 1,
    "children_playing": 2,
    "dog_bark": 3,
    "drilling": 4,
    "engine_idling": 5,
    "gun_shot": 6,
    "jackhammer": 7,
    "siren": 8,
    "street_music": 9  
}

#Convert labels into numbers from dict
class_labels = [item[0] for item in data_labels]
labels = [label_dict[label] for label in class_labels]
num_labels= (len(labels))

#Train,val,test labels
train_labels = labels[:int(num_labels * 0.6)]
valid_labels =labels[int(num_labels * 0.6):int(num_labels * 0.8)]
test_labels = labels[int(num_labels * 0.8):]
print(len(train_labels))
print(len(valid_labels))
print(len(test_labels))

#Train, val,test audio data paths
train_data = audio_paths[:int(num_labels * 0.6)]
valid_data = audio_paths[int(num_labels * 0.6):int(num_labels * 0.8)]
test_data = audio_paths[int(num_labels * 0.8):]
print(len(train_data))
print(len(valid_data))
print(len(test_data))

#write a loops for all training audio samples store spectrograms data for CNN?
wave_data,sr = librosa.load(train_paths[0], sr=30000)
train_data = librosa.effects.time_stretch(wave_data,rate=0.5)
train_data = train_data[:4*sr]
train_data = librosa.util.fix_length(train_data,size=4*sr)

#Loop for all validate data
wave_data,sr = librosa.load(valid_paths[0], sr=30000)
valid_data = librosa.effects.time_stretch(wave_data,rate=0.5)
valid_data = valid_data[:3*sr]
valid_data = librosa.util.fix_length(valid_data,size=3*sr)


#Loop for all testing data
wave_data,sr = librosa.load(test_paths[0], sr=30000)
test_data = librosa.effects.time_stretch(wave_data,rate=0.5)
test_data = test_data[:3*sr]
test_data = librosa.util.fix_length(test_data,size=3*sr)

#Plotting the waveform/data
#print(wave_data.size)
#Plotting raw data for checking
#plt.figure(figsize=(10, 4))
#librosa.display.waveshow(wave_data,max_points=10000)
#plt.show()


#Spectrogram
Spectro = librosa.feature.melspectrogram(y=train_data,sr=sr)

#convert to DB scale
S_dB = librosa.power_to_db(Spectro, ref=np.max)



'''
#plotting spectrogram
fig, ax = plt.subplots()
img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sr,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram')
plt.show()
'''

