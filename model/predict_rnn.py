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
    mfcc = extract_features(file_path)
    mfcc = np.expand_dims(mfcc, axis=0)
    preds = model.predict(mfcc)
    preds = np.argmax(preds, axis=-1)
    text = sequence_to_text(preds[0])
    print("Model output:", text)
    
# Test with one of our training files (use absolute path)
import os
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
audio_file = os.path.join(data_dir, "a_007.wav")
print(f"Looking for audio file at: {audio_file}")
predict(audio_file)