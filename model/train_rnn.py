import numpy as np
from tensorflow.keras.utils import to_categorical
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_preprocess import prepare_dataset
from text_utils import text_to_sequence, VOCAB_SIZE
from model_rnn import build_model

# Use local audio files
print("Loading local audio files...")
x_all, y_texts_all = prepare_dataset("./data/")

# Filter out sentences that are too long for CTC (keep complete sentences)
min_chars = 10   # Minimum useful length
max_chars = 300  # Maximum manageable length
x, y_texts = [], []

for i, text in enumerate(y_texts_all):
    text_len = len(text)
    if min_chars <= text_len <= max_chars:
        x.append(x_all[i])
        y_texts.append(text)

print(f"Filtered from {len(x_all)} to {len(x)} samples")
print(f"Kept sentences between {min_chars}-{max_chars} characters")
print(f"Sample lengths: {[len(t) for t in y_texts[:5]]}")

# Limit sequence length and pad to same size (increased for long sentences)
max_audio_len = 500  # Increased from 200 to handle longer sentences
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

# Process full sentences for CTC
print("Processing full sentences for CTC training...")
print(f"Sample texts: {y_texts[:3]}")

# Convert text to character sequences (for CTC labels)
y_sequences = [text_to_sequence(t) for t in y_texts]

# No truncation needed - we filtered by length already

max_text_len = max(len(seq) for seq in y_sequences) if y_sequences else 1

print(f"Max text length: {max_text_len} characters")

# For CTC, we need labels as sequences (not one-hot)
# Pad label sequences to same length
y_padded = np.zeros((len(y_sequences), max_text_len), dtype=np.int32)
for i, seq in enumerate(y_sequences):
    if len(seq) > 0:
        # Truncate if too long
        if len(seq) > max_text_len:
            seq = seq[:max_text_len]
        y_padded[i, :len(seq)] = seq

print(f"Label shape: {y_padded.shape}")
print(f"Audio shape: {np.array(x).shape}")

#BUILD CTC MODEL
input_dim = x[0].shape[1]
output_dim = VOCAB_SIZE  # Include CTC blank token
model = build_model(input_dim, output_dim)

print(f"Model input dim: {input_dim}")
print(f"Model output dim: {output_dim}")

# Train with CTC loss and early stopping
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-6)
]

model.fit(np.array(x), y_padded, epochs=20, batch_size=4, validation_split=0.2, callbacks=callbacks)

model.save("my_rnn_model.h5")
