#  Advanced Music Source Separation

## Overview
This project uses Facebook's Demucs library to separate music into multiple components: vocals, drums, bass, and other instruments. It supports 4, 6, and 8-component separation with advanced techniques.

## Features

###  Multi-Component Separation
- **4 Components**: vocals, drums, bass, other
- **6 Components**: vocals, drums, bass, piano, guitar, other  
- **8 Components**: Multiple model ensemble separation

###  Advanced Features
- **Multiple Demucs Models**: htdemucs, mdx_extra, htdemucs_ft
- **Two-Stems Technique**: Further separate "other" into piano and guitar
- **Multi-Model Ensemble**: Combine different models for better results
- **Professional CLI**: Command-line interface with comprehensive options
- **Batch Processing**: Process multiple files (coming soon)
- **Multiple Formats**: MP3, WAV, FLAC, M4A, AAC support

## Installation

1. **Clone/Download** this repository
2. **Install Demucs** in virtual environment:
   ```bash
   python -m venv venv_demucs
   venv_demucs\Scripts\activate
   pip install demucs
   ```

## Usage

### Basic Usage
```bash
# 4-component separation (standard)
python music_separator.py input_file.mp3 -c 4

# 6-component separation (advanced)
python music_separator.py input_file.mp3 -c 6

# 8-component separation (ultra)
python music_separator.py input_file.mp3 -c 8
```

### Advanced Usage
```bash
# Use different Demucs model
python music_separator.py input.mp3 -m mdx_extra -c 6

# Custom output directory
python music_separator.py input.mp3 -o my_output -c 4

# Use GPU acceleration (if available)
python music_separator.py input.mp3 -d cuda -c 8
```

### Command Line Options
- `input` - Input audio file
- `-c, --components` - Number of components (4, 6, or 8)
- `-m, --model` - Demucs model: htdemucs, mdx_extra, htdemucs_ft
- `-d, --device` - Processing device: cpu, cuda, mps
- `-o, --output` - Output directory (default: output_demucs)

## Output Structure

### 4-Component Output
```
output_demucs/
 htdemucs/
     song_name/
         vocals.wav      # Lead vocals and harmonies
         drums.wav       # Drums and percussion
         bass.wav        # Bass guitar and low frequencies
         other.wav       # Other instruments
```

### 6-Component Output
```
output_demucs/
 6_components/
     song_name/
         vocals.wav      # Lead vocals and harmonies
         drums.wav       # Drums and percussion
         bass.wav        # Bass guitar and low frequencies
         piano.wav       # Piano and keyboard instruments
         guitar.wav      # Electric and acoustic guitars
         other.wav       # Brass, strings, synths, etc.
```

### 8-Component Output
```
output_demucs/
 8_components/
     song_name/
         vocals.wav_1    # Vocals (htdemucs model)
         drums.wav_1     # Drums (htdemucs model)
         bass.wav_1      # Bass (htdemucs model)
         other.wav_1     # Other (htdemucs model)
         vocals.wav_2    # Vocals (mdx_extra model)
         drums.wav_2     # Drums (mdx_extra model)
         bass.wav_2      # Bass (mdx_extra model)
         other.wav_2     # Other (mdx_extra model)
```

## Examples

### 4-Component Separation
```bash
python music_separator.py 7years.mp3 -c 4
```
**Output:**
```
 Starting ULTRA 4-component separation for: 7years.mp3
 Running standard 4-component separation...
 Ultra separation complete!

 ULTRA 4-COMPONENT SEPARATION RESULTS:
  VOCALS - Lead vocals and harmonies
    vocals.wav (39.9 MB)
  DRUMS - Kick, snare, hi-hats, cymbals
    drums.wav (39.9 MB)
  BASS - Bass guitar and low-frequency instruments
    bass.wav (39.9 MB)
  OTHER - Piano, guitar, synths, strings, etc.
    other.wav (39.9 MB)
```

### 6-Component Separation
```bash
python music_separator.py 7years.mp3 -c 6
```
**Output:**
```
 Starting ULTRA 6-component separation for: 7years.mp3
 Running 6-component separation using two-stems technique...
 Further separating 'other' component...
 Created piano.wav
 Created guitar.wav
 Ultra separation complete!

 ULTRA 6-COMPONENT SEPARATION RESULTS:
  VOCALS - Lead vocals and harmonies
  DRUMS - Kick, snare, hi-hats, cymbals
  BASS - Bass guitar and low-frequency instruments
  PIANO - Piano and keyboard instruments
  GUITAR - Electric and acoustic guitars
  OTHER - Brass, strings, synths, etc.
```

## Applications

###  Music Production
- **Karaoke Creation** - Use vocals.wav for karaoke tracks
- **Remixing** - Isolate specific instruments for remixing
- **Sample Libraries** - Extract drum loops, bass lines, etc.
- **Cover Songs** - Use instrumental backing tracks

###  Audio Analysis
- **Music Education** - Study individual instrument parts
- **Transcription** - Easier to transcribe specific instruments
- **Mixing Practice** - Learn mixing with isolated stems

###  Creative Projects
- **Mashup Creation** - Combine different stems
- **Audio Effects** - Process isolated components
- **Live Performance** - Real-time stem separation

## Technical Details

### Models Available
- **htdemucs** - Default model, good balance of quality and speed
- **mdx_extra** - Higher quality, trained with extra data
- **htdemucs_ft** - Fine-tuned version, highest quality

### Performance
- **CPU Processing** - Slower but works on any system
- **GPU Processing** - Much faster with CUDA-compatible GPU
- **Processing Time** - Varies by file length and hardware

## Troubleshooting

### Common Issues
1. **"No module named demucs"** - Activate virtual environment
2. **CUDA errors** - Use `-d cpu` for CPU processing
3. **File not found** - Check file path and extension

### Solutions
```bash
# Activate virtual environment
venv_demucs\Scripts\activate

# Use CPU if GPU issues
python music_separator.py input.mp3 -d cpu -c 4

# Check file exists
dir *.mp3
```

## Project Structure
```
music-separator/
 music_separator.py      # Main separation script
 README.md              # This file
 venv_demucs/           # Virtual environment
 7years.mp3            # Sample audio file
 my_song.mp3           # Sample audio file
 whatsappAudio.mp3     # Sample audio file
```

## Future Enhancements
- **Web Interface** - Browser-based GUI
- **Real-time Processing** - Live audio separation
- **Cloud Integration** - Upload to cloud storage
- **Mobile App** - Smartphone application
- **API Development** - REST API for integration

## License
This project uses Demucs library. Please check Demucs license for usage terms.

---
**Created by:** Advanced Music Source Separation Project  
**Version:** 2.0 (Multi-Component Enhanced)  
**Last Updated:** 2025

# Music_Deconstructor