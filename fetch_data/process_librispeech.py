import os
import shutil
import librosa
import soundfile as sf

def process_librispeech_files(librispeech_path, output_dir="data", max_files=1000):
    """
    Process LibriSpeech files and convert them for training
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    
    print(f"Processing LibriSpeech files from: {librispeech_path}")
    
    # Walk through all directories
    for root, dirs, files in os.walk(librispeech_path):
        if processed_count >= max_files:
            break
            
        # Look for transcript files
        trans_files = [f for f in files if f.endswith('.trans.txt')]
        
        for trans_file in trans_files:
            if processed_count >= max_files:
                break
                
            trans_path = os.path.join(root, trans_file)
            
            # Read transcriptions
            with open(trans_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                if processed_count >= max_files:
                    break
                    
                parts = line.strip().split(' ', 1)
                if len(parts) < 2:
                    continue
                    
                file_id = parts[0]
                transcript = parts[1].lower()
                
                # Find corresponding audio file
                audio_file = None
                for ext in ['.flac', '.wav']:
                    potential_file = os.path.join(root, file_id + ext)
                    if os.path.exists(potential_file):
                        audio_file = potential_file
                        break
                
                if audio_file:
                    try:
                        # Load and convert audio
                        audio, sr = librosa.load(audio_file, sr=16000)
                        
                        # Create filename based on first word of transcript
                        first_word = transcript.split()[0] if transcript.split() else "unknown"
                        # Clean the word (remove punctuation)
                        first_word = ''.join(c for c in first_word if c.isalnum())
                        
                        output_filename = f"{first_word}_{processed_count:03d}.wav"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        # Save as WAV
                        sf.write(output_path, audio, 16000)
                        
                        # Save full transcript to transcripts.txt
                        transcript_file = os.path.join(output_dir, "transcripts.txt")
                        with open(transcript_file, 'a', encoding='utf-8') as f:
                            f.write(f"{output_filename}\t{transcript}\n")
                        
                        processed_count += 1
                        print(f"Processed {processed_count}: {output_filename} -> '{transcript[:50]}...'")
                        
                    except Exception as e:
                        print(f"Error processing {audio_file}: {e}")
                        continue
    
    print(f"\nProcessing complete! Created {processed_count} audio files in '{output_dir}' folder.")
    
    # List created files
    print("\nCreated files:")
    for file in sorted(os.listdir(output_dir)):
        if file.endswith('.wav'):
            print(f"  - {file}")

if __name__ == "__main__":
    # Update this path to where you extracted test-clean
    librispeech_path = input("Enter the path to your test-clean folder: ").strip().strip('"')
    
    if not os.path.exists(librispeech_path):
        print(f"Path not found: {librispeech_path}")
        print("Please check the path and try again.")
    else:
        process_librispeech_files(librispeech_path, max_files=1000)
        print("\nReady to train! Run: python model/train_rnn.py")