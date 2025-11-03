import librosa
import numpy as np
import os

def extract_features(file_path, n_mfcc=40, sr= 16000):
    #Load audio file
    y, sr = librosa.load(file_path, sr=sr)
    #Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)  
    #Transpose to shape(time_steps, features)
    return mfcc.T

def prepare_dataset(audio_dir):
    x, y = [], []
    
    for file in os.listdir(audio_dir):
        if file.endswith(".wav"):
            mfcc = extract_features(os.path.join(audio_dir, file))
            x.append(mfcc)
            
            # Extract label from filename (everything before first underscore or dot)
            label = file.split('_')[0].split('.')[0]
            y.append(label)
    
    return x, y