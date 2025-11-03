import string

#Create vocabulary for full sentences (LibriSpeech has punctuation)
VOCAB = list(string.ascii_lowercase + " .,!?'-")
char_to_index = {c: i for i, c in enumerate(VOCAB)}
index_to_char = {i: c for c, i in char_to_index.items()}

def text_to_sequence(text):
    # Clean text and convert to sequence
    text = text.lower()
    # Remove characters not in vocabulary
    text = ''.join([c for c in text if c in char_to_index])
    return [char_to_index[c] for c in text]

def sequence_to_text(seq):
    return "".join([index_to_char[i] for i in seq if i in index_to_char])