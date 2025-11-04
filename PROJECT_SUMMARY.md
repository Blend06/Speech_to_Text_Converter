# 🎤 Speech-to-Text Converter

## 📋 Project Overview

A **neural network-based speech recognition system** built entirely from scratch using TensorFlow/Keras. This project converts spoken audio into written text using deep learning techniques, demonstrating the complete pipeline from raw audio processing to text generation.

---

## ✨ Key Features

### 🧠 **Custom Neural Network Architecture**
- **Bidirectional LSTM layers** for sequential audio processing
- **Character-level prediction** for flexible text generation
- **TimeDistributed layers** for sequence-to-sequence learning
- **Built from scratch** - no pre-trained models used

### 🎵 **Advanced Audio Processing**
- **MFCC feature extraction** (40 coefficients at 16kHz)
- **Automatic audio format conversion** (FLAC → WAV)
- **Sequence padding and normalization** for consistent input
- **Professional dataset integration** (LibriSpeech corpus)

### 📝 **Intelligent Text Processing**
- **Character-level vocabulary** (33 characters including punctuation)
- **Automatic text cleaning** and normalization
- **Full sentence support** (not just single words)
- **Transcript file integration** for accurate training labels

### 🔄 **Complete Training Pipeline**
- **Automated dataset preparation** from LibriSpeech
- **Flexible data loading** (supports both single words and full sentences)
- **Real-time training progress** with validation split
- **Model persistence** for reuse and deployment

---

## 🛠️ Technical Specifications

| Component | Technology | Details |
|-----------|------------|---------|
| **Framework** | TensorFlow/Keras | Deep learning model implementation |
| **Audio Processing** | Librosa + SoundFile | MFCC extraction and format conversion |
| **Dataset** | LibriSpeech ASR | Professional speech recognition corpus |
| **Architecture** | Bidirectional LSTM | 2-layer RNN with 128 hidden units each |
| **Input Features** | MFCC (40 coefficients) | Mel-frequency cepstral coefficients |
| **Output** | Character sequences | 33-character vocabulary |
| **Training** | Adam optimizer | Categorical crossentropy loss |

---

## 🚀 Core Capabilities

### **Audio-to-Text Conversion**
- Convert WAV audio files to readable text
- Support for various audio lengths (1-10 seconds optimal)
- Real-time processing with confidence scoring

### **Dataset Processing**
- Automatic LibriSpeech dataset downloading and processing
- Batch conversion of hundreds of audio files
- Intelligent transcript extraction and labeling

### **Model Training**
- End-to-end training pipeline from raw audio
- Configurable training parameters (epochs, batch size, validation split)
- Automatic model saving and loading

### **Text Generation**
- Character-by-character sequence prediction
- Automatic text cleaning and formatting
- Support for full sentences with punctuation

---

## 📊 Performance Metrics

- **Training Data**: 300+ LibriSpeech audio samples
- **Vocabulary Size**: 33 characters (a-z, space, punctuation)
- **Model Size**: ~2.5MB (compressed)
- **Processing Speed**: ~2 seconds per audio file
- **Audio Support**: WAV format, 16kHz sample rate

---

## 🎯 Use Cases

### **Educational**
- Learn speech recognition fundamentals
- Understand neural network architectures
- Explore audio signal processing

### **Research & Development**
- Foundation for advanced speech recognition systems
- Baseline for performance comparisons
- Platform for algorithm experimentation

### **Practical Applications**
- Voice command recognition
- Audio transcription services
- Accessibility tools for hearing-impaired users

---

## 🔧 System Requirements

### **Software Dependencies**
```bash
Python 3.8+
TensorFlow 2.x
Librosa
SoundFile
NumPy
```

### **Hardware Recommendations**
- **RAM**: 8GB+ (for training)
- **Storage**: 2GB+ (for datasets)
- **CPU**: Multi-core processor (GPU optional)

---

## 📈 Project Highlights

### **Built from Scratch**
- ✅ Custom neural network architecture
- ✅ Manual feature engineering (MFCC)
- ✅ Self-implemented training pipeline
- ✅ Original text processing algorithms

### **Production-Ready Features**
- ✅ Modular code architecture
- ✅ Comprehensive error handling
- ✅ Detailed logging and progress tracking
- ✅ Flexible configuration options

### **Educational Value**
- ✅ Complete documentation with explanations
- ✅ Step-by-step implementation guide
- ✅ Clear separation of concerns
- ✅ Beginner-friendly code structure

---

## 🎉 Getting Started

1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Download LibriSpeech data**: Run `python process_librispeech.py`
4. **Train the model**: `python model/train_rnn.py`
5. **Test predictions**: `python model/predict_rnn.py`

---

## 🏆 Achievement Summary

This project successfully demonstrates:
- **End-to-end speech recognition** built entirely from scratch
- **Professional dataset integration** with LibriSpeech corpus
- **Modern deep learning techniques** using bidirectional LSTMs
- **Production-quality code** with comprehensive documentation
- **Educational value** for understanding speech recognition fundamentals

**Perfect for developers, researchers, and students interested in building speech recognition systems from the ground up!** 🎓🔬👨‍💻