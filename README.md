# Speech-to-Text Converter

## Project Motive

This project implements a **neural network-based speech-to-text converter** built from scratch using TensorFlow/Keras. The primary motivation is to create an end-to-end speech recognition system that can:

- **Convert spoken audio into written text**
- **Learn from professional speech datasets** (LibriSpeech)
- **Process real-world audio files** with reasonable accuracy
- **Demonstrate the complete pipeline** from raw audio to text output
- **Provide a foundation** for further speech recognition research and development

The system uses **Recurrent Neural Networks (RNNs)** with **LSTM layers** to process sequential audio data and **MFCC (Mel-Frequency Cepstral Coefficients)** for feature extraction, following industry-standard approaches to speech recognition.

---

## Project Structure

```
Speech_to_Text_Converter/
├── data/                          # Processed training audio files
├── model/                         # Neural network components
│   ├── model_rnn.py              # RNN architecture definition
│   ├── train_rnn.py              # Model training script
│   ├── predict_rnn.py            # Audio-to-text prediction
│   └── my_rnn_model.h5           # Trained model weights
├── data_preprocess.py            # Audio feature extraction
├── text_utils.py                 # Text processing utilities
├── process_librispeech.py        # Dataset preparation script
└── README.md                     # This documentation
```

---

## File Descriptions

### Core Files

#### `data_preprocess.py`
**Purpose**: Audio preprocessing and feature extraction

**Key Functions**:
- `extract_features(file_path, n_mfcc=40, sr=16000)`: 
  - Loads audio files using librosa
  - Extracts 40 MFCC coefficients at 16kHz sample rate
  - Returns transposed feature matrix (time_steps, features)
- `prepare_dataset(audio_dir)`:
  - Processes all WAV files in a directory
  - Extracts labels from filenames (e.g., "hello_001.wav" → "hello")
  - Returns feature arrays and corresponding text labels

**Technical Details**:
- Uses **MFCC features** which mimic human auditory perception
- **16kHz sampling rate** standard for speech recognition
- **40 coefficients** capture essential spectral characteristics
- Handles automatic label extraction from filename patterns

#### `text_utils.py`
**Purpose**: Text processing and character-level encoding

**Key Components**:
- `VOCAB`: Character vocabulary including lowercase letters, space, and punctuation
- `char_to_index`: Dictionary mapping characters to numerical indices
- `index_to_char`: Reverse mapping for decoding predictions

**Key Functions**:
- `text_to_sequence(text)`: Converts text strings to numerical sequences
- `sequence_to_text(seq)`: Converts numerical predictions back to readable text

**Technical Details**:
- **Character-level processing** for fine-grained text generation
- **33-character vocabulary** (a-z, space, punctuation)
- Handles text cleaning and normalization
- Essential for neural network input/output processing

### Model Components

#### `model/model_rnn.py`
**Purpose**: Neural network architecture definition

**Architecture**:
```python
Sequential([
    Bidirectional(LSTM(128, return_sequences=True)),  # Processes sequences in both directions
    TimeDistributed(Dense(128, activation='relu')),   # Applies dense layer to each time step
    TimeDistributed(Dense(output_dim, activation='softmax'))  # Character probability output
])
```

**Key Features**:
- **Bidirectional LSTM**: Processes audio sequences forward and backward for better context
- **128 hidden units**: Balances model capacity with computational efficiency
- **TimeDistributed layers**: Applies same transformation to each time step
- **Softmax output**: Produces probability distribution over character vocabulary

**Technical Specifications**:
- **Input**: (batch_size, time_steps, 40) MFCC features
- **Output**: (batch_size, time_steps, 33) character probabilities
- **Optimizer**: Adam with default learning rate
- **Loss**: Categorical crossentropy for multi-class classification

#### `model/train_rnn.py`
**Purpose**: Model training pipeline

**Training Process**:
1. **Data Loading**: Loads processed audio files from `data/` directory
2. **Sequence Padding**: Ensures all audio sequences are exactly 200 time steps
3. **Text Encoding**: Converts text labels to one-hot encoded sequences
4. **Model Training**: Trains for 10 epochs with batch size 4
5. **Model Saving**: Saves trained weights to `my_rnn_model.h5`

**Key Parameters**:
- **max_audio_len**: 200 time steps (~4-5 seconds of audio)
- **epochs**: 10 training iterations
- **batch_size**: 4 samples processed simultaneously
- **validation_split**: 20% of data reserved for validation

**Technical Details**:
- Handles variable-length audio by padding/truncating to fixed size
- Repeats text sequences to match audio length for sequence-to-sequence learning
- Uses categorical crossentropy loss for character-level prediction

#### `model/predict_rnn.py`
**Purpose**: Audio-to-text inference

**Prediction Process**:
1. **Model Loading**: Loads trained model from `my_rnn_model.h5`
2. **Feature Extraction**: Processes input audio file through MFCC extraction
3. **Batch Preparation**: Adds batch dimension for model input
4. **Inference**: Runs neural network prediction
5. **Decoding**: Converts numerical output to readable text

**Key Functions**:
- `predict(file_path)`: Complete audio-to-text conversion pipeline
- Handles path resolution and error checking
- Outputs predicted text to console

### Dataset Processing

#### `process_librispeech.py`
**Purpose**: LibriSpeech dataset preparation and conversion

**Key Functions**:
- `process_librispeech_files(librispeech_path, output_dir="data", max_files=50)`:
  - Recursively walks through LibriSpeech directory structure
  - Reads transcript files (.trans.txt) for ground truth labels
  - Converts FLAC audio files to 16kHz WAV format
  - Creates filename-based labels for training
  - Saves processed files to `data/` directory

**Processing Pipeline**:
1. **Directory Traversal**: Uses `os.walk()` to find all audio files
2. **Transcript Reading**: Parses .trans.txt files for accurate transcriptions
3. **Audio Conversion**: Loads FLAC files and resamples to 16kHz WAV
4. **Label Generation**: Extracts first word from transcript as filename label
5. **File Organization**: Saves as `{word}_{count:03d}.wav` format

**Technical Features**:
- **Automatic format conversion**: FLAC → WAV
- **Sample rate standardization**: All audio converted to 16kHz
- **Intelligent labeling**: Uses actual transcript content for labels
- **Error handling**: Skips corrupted or problematic files
- **Progress tracking**: Shows processing status and file counts

---

## Usage Instructions

### 1. Dataset Preparation
```bash
# Download LibriSpeech dataset from https://www.openslr.org/12/
# Extract test-clean.tar.gz or dev-clean.tar.gz

# Process dataset
python process_librispeech.py
# Enter path when prompted: C:\path\to\LibriSpeech\test-clean
```

### 2. Model Training
```bash
# Train the neural network
python model/train_rnn.py
# This will create my_rnn_model.h5 with trained weights
```

### 3. Audio-to-Text Conversion
```bash
# Convert audio file to text
python model/predict_rnn.py
# Edit the script to change the input audio file path
```

---

## Technical Requirements

### Dependencies
```bash
pip install tensorflow
pip install librosa
pip install soundfile
pip install numpy
```

### System Requirements
- **Python 3.8+**
- **TensorFlow 2.x**
- **8GB+ RAM** (for model training)
- **Windows/Linux/macOS** compatible

### Audio Format Requirements
- **Format**: WAV files preferred
- **Sample Rate**: 16kHz (automatically converted)
- **Duration**: 1-10 seconds optimal
- **Quality**: Clear speech, minimal background noise

---

## Model Performance

### Current Capabilities
- **Training Data**: 30-50 LibriSpeech samples
- **Vocabulary**: 33 characters (a-z, space, punctuation)
- **Audio Length**: Up to 200 time steps (~4-5 seconds)
- **Accuracy**: Basic word recognition (proof of concept)

### Limitations
- **Limited Training Data**: Small dataset affects accuracy
- **Simple Architecture**: Basic RNN may miss complex patterns
- **Character Repetition**: Current training method repeats characters
- **No Language Model**: Lacks linguistic context for better predictions

### Potential Improvements
- **More Training Data**: Use full LibriSpeech datasets (100+ hours)
- **Advanced Architecture**: Implement Transformer or CTC-based models
- **Better Alignment**: Use forced alignment for character-level timing
- **Language Model**: Add n-gram or neural language model
- **Data Augmentation**: Apply noise, speed, and pitch variations

---

## Project Architecture

### Data Flow
```
Audio File (.wav) 
    ↓
MFCC Feature Extraction (40 coefficients)
    ↓
Sequence Padding (200 time steps)
    ↓
Bidirectional LSTM Processing
    ↓
Dense Layer Transformation
    ↓
Softmax Character Probabilities
    ↓
Text Decoding
    ↓
Final Text Output
```

### Training Pipeline
```
LibriSpeech Dataset
    ↓
Audio Processing (FLAC → WAV)
    ↓
Feature Extraction (MFCC)
    ↓
Text Encoding (Character → Numbers)
    ↓
Sequence Alignment (Padding/Truncation)
    ↓
Neural Network Training
    ↓
Model Weights Saving
```

---

## Contributing

This project serves as an educational foundation for speech recognition. Potential contributions include:

- **Dataset Expansion**: Adding more diverse speech data
- **Architecture Improvements**: Implementing modern speech recognition models
- **Performance Optimization**: GPU acceleration and batch processing
- **Evaluation Metrics**: Adding WER (Word Error Rate) calculation
- **Real-time Processing**: Streaming audio recognition capabilities

---

## License

This project is for educational purposes. LibriSpeech dataset usage follows their respective licensing terms.

---

## Acknowledgments

- **LibriSpeech**: Open-source speech corpus for training data
- **TensorFlow/Keras**: Deep learning framework
- **Librosa**: Audio processing library
- **OpenSLR**: Speech and language resources repository