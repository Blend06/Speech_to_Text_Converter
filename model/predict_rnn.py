import numpy as np
from tensorflow.keras.models import load_model
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocess import extract_features
from text_utils import sequence_to_text, VOCAB_SIZE
from model_rnn import ctc_loss_func  # Import the custom loss function

# Load model with custom loss function
model = load_model("my_rnn_model.h5", custom_objects={'ctc_loss_func': ctc_loss_func})

def predict(file_path):
    # Extract features (same as training)
    mfcc = extract_features(file_path)
    
    # Pad to same length as training (500 time steps now)
    max_audio_len = 500
    if len(mfcc) > max_audio_len:
        mfcc = mfcc[:max_audio_len]
    elif len(mfcc) < max_audio_len:
        padding = np.zeros((max_audio_len - len(mfcc), mfcc.shape[1]))
        mfcc = np.vstack([mfcc, padding])
    
    # Add batch dimension
    mfcc = np.expand_dims(mfcc, axis=0)
    
    # Predict with CTC model
    preds = model.predict(mfcc)
    
    # CTC decoding - get most likely character at each time step
    predicted_chars = np.argmax(preds, axis=-1)[0]
    
    # Debug: Show what model actually predicts
    unique_predictions = np.unique(predicted_chars, return_counts=True)
    print(f"Raw predictions: {unique_predictions}")
    print(f"Blank token (33) count: {np.sum(predicted_chars == 33)}/{len(predicted_chars)}")
    
    # CTC post-processing: remove blanks and collapse repeats
    decoded_chars = []
    prev_char = -1
    
    for char_idx in predicted_chars:
        # Skip blank tokens (index 33)
        if char_idx == 33:  # blank token
            prev_char = char_idx
            continue
        # Skip repeated characters (CTC collapse)
        if char_idx != prev_char:
            decoded_chars.append(char_idx)
        prev_char = char_idx
    
    print(f"After CTC decoding: {decoded_chars}")
    
    # Convert to text
    text = sequence_to_text(decoded_chars)
    
    print(f"Model output: '{text}'")
    return text
    
# Test with one of our training files (use absolute path)
import os
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
audio_file = os.path.join(data_dir, "as_008.wav")
print(f"Looking for audio file at: {audio_file}")
predict(audio_file)