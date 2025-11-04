import numpy as np
from tensorflow.keras.models import load_model
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocess import extract_features
from text_utils import sequence_to_text

model = load_model("my_rnn_model.h5")

def predict(file_path):
    # Extract features (same as training)
    mfcc = extract_features(file_path)
    
    # Pad to same length as training (200 time steps)
    max_audio_len = 200
    if len(mfcc) > max_audio_len:
        mfcc = mfcc[:max_audio_len]
    elif len(mfcc) < max_audio_len:
        padding = np.zeros((max_audio_len - len(mfcc), mfcc.shape[1]))
        mfcc = np.vstack([mfcc, padding])
    
    # Add batch dimension
    mfcc = np.expand_dims(mfcc, axis=0)
    
    # Predict character sequence
    preds = model.predict(mfcc)
    predicted_chars = np.argmax(preds, axis=-1)[0]
    
    # Convert to text and clean up
    text = sequence_to_text(predicted_chars)
    # Remove repeated characters and clean up
    cleaned_text = ""
    prev_char = ""
    for char in text:
        if char != prev_char and char != ' ' or (char == ' ' and prev_char != ' '):
            cleaned_text += char
        prev_char = char
    
    print(f"Model output: '{cleaned_text.strip()}'")
    return cleaned_text.strip()
    
# Test with one of our training files (use absolute path)
import os
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
audio_file = os.path.join(data_dir, "as_008.wav")
print(f"Looking for audio file at: {audio_file}")
predict(audio_file)