import string

#Create vocabulary for full sentences (LibriSpeech has punctuation)
VOCAB = list(string.ascii_lowercase + " .,!?'-") + ['<PAD>']  # Add padding token
char_to_index = {c: i for i, c in enumerate(VOCAB)}
index_to_char = {i: c for c, i in char_to_index.items()}

print(f"Vocabulary size: {len(VOCAB)} characters")
print(f"Characters: {''.join(VOCAB[:10])}... (showing first 10)")

def text_to_sequence(text):
    # Clean text and convert to sequence
    text = text.lower()
    # Remove characters not in vocabulary
    text = ''.join([c for c in text if c in char_to_index])
    return [char_to_index[c] for c in text]

def sequence_to_text(seq):
    # Convert sequence to text and clean up
    text = "".join([index_to_char[i] for i in seq if i in index_to_char])
    # Remove padding tokens
    text = text.replace('<PAD>', '')
    # Clean up multiple spaces
    import re
    text = re.sub(r'\s+', ' ', text)
    return text.strip()