import soundata
import librosa
import librosa.display
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np



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

