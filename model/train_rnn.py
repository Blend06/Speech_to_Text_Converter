import numpy as np
from tensorflow.keras.utils import to_categorical
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocess import prepare_dataset
from text_utils import text_to_sequence, VOCAB
from model_rnn import build_model

# Use local audio files
print("Loading local audio files...")
x, y_texts = prepare_dataset("./data/")

# Limit sequence length and pad to same size
max_audio_len = 200
x_padded = []

for sample in x:
    # Truncate if too long
    if len(sample) > max_audio_len:
        sample = sample[:max_audio_len]
    
    # Pad if too short
    if len(sample) < max_audio_len:
        padding = np.zeros((max_audio_len - len(sample), sample.shape[1]))
        sample = np.vstack([sample, padding])
    
    x_padded.append(sample)

x = x_padded

#Encode text labels
y_sequences = [text_to_sequence(t) for t in y_texts]

# Pad text sequences to match audio length (200 time steps)
y_padded = np.zeros((len(y_sequences), max_audio_len))
for i, seq in enumerate(y_sequences):
    # Repeat the sequence to fill the audio length
    if len(seq) > 0:
        # Repeat the sequence to match audio length
        repeated_seq = (seq * (max_audio_len // len(seq) + 1))[:max_audio_len]
        y_padded[i, :] = repeated_seq
    else:
        # If empty sequence, use padding token (0)
        y_padded[i, :] = 0

y_onehot = to_categorical(y_padded, num_classes=len(VOCAB))

#BUILD MODEL
input_dim = x[0].shape[1]
output_dim = len(VOCAB)
model = build_model(input_dim, output_dim)

model.fit(np.array(x), y_onehot, epochs=10, batch_size=4, validation_split=0.2)

model.save("my_rnn_model.h5")
