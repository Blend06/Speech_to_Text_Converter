import string

#Create vocabulary for CTC (includes blank token)
VOCAB = list(string.ascii_lowercase + " .,!?'-")
BLANK_TOKEN = len(VOCAB)  # CTC blank token index
VOCAB_SIZE = len(VOCAB) + 1  # +1 for blank token

char_to_index = {c: i for i, c in enumerate(VOCAB)}
index_to_char = {i: c for c, i in char_to_index.items()}

print(f"Vocabulary size: {VOCAB_SIZE} characters (including CTC blank)")
print(f"Characters: {''.join(VOCAB[:10])}... (showing first 10)")
print(f"Blank token index: {BLANK_TOKEN}")

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