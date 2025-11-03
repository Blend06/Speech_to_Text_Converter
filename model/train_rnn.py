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

# Process full sentences (not just single words)
print("Processing full sentences...")
print(f"Sample texts: {y_texts[:3]}")  # Show first 3 examples

# Convert text to character sequences
y_sequences = [text_to_sequence(t) for t in y_texts]
max_text_len = max(len(seq) for seq in y_sequences) if y_sequences else 1

print(f"Max text length: {max_text_len} characters")

# Pad text sequences to match audio length (200 time steps)
y_padded = np.zeros((len(y_sequences), max_audio_len))
for i, seq in enumerate(y_sequences):
    if len(seq) > 0:
        # Truncate if text is longer than audio
        if len(seq) > max_audio_len:
            seq = seq[:max_audio_len]
        # Place text at beginning, pad rest with zeros
        y_padded[i, :len(seq)] = seq

y_onehot = to_categorical(y_padded, num_classes=len(VOCAB))

#BUILD MODEL
input_dim = x[0].shape[1]
output_dim = len(VOCAB)  # Number of characters in vocabulary
model = build_model(input_dim, output_dim)

model.fit(np.array(x), y_onehot, epochs=10, batch_size=4, validation_split=0.2)

model.save("my_rnn_model.h5")
