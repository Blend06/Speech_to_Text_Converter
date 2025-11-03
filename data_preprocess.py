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
    
    # Also look for transcript file with full sentences
    transcript_file = os.path.join(audio_dir, "transcripts.txt")
    transcripts = {}
    
    # Try to load full transcripts if available
    if os.path.exists(transcript_file):
        with open(transcript_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    filename = parts[0]
                    transcript = parts[1]
                    transcripts[filename] = transcript
    
    for file in os.listdir(audio_dir):
        if file.endswith(".wav"):
            mfcc = extract_features(os.path.join(audio_dir, file))
            x.append(mfcc)
            
            # Use full transcript if available, otherwise use filename
            if file in transcripts:
                label = transcripts[file]
                print(f"Using transcript for {file}: {label[:50]}...")
            else:
                # Fallback: extract first word from filename
                label = file.split('_')[0].split('.')[0]
                print(f"Using filename label for {file}: {label}")
            
            y.append(label)
    
    print(f"Loaded {len(x)} audio files with labels")
    return x, y