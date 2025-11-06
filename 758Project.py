import soundata
#import librosa
#import matplotlib.pyplot as plt
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

