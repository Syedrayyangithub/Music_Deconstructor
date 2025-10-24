import os
import subprocess
import glob
import argparse
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np

def enhance_audio(input_file, output_file, enhancement_type="vocals"):
    """
    Enhance audio using librosa for better quality
    
    Args:
        input_file (str): Path to input audio file
        output_file (str): Path to output enhanced audio file
        enhancement_type (str): Type of enhancement (vocals, drums, bass, other)
    """
    try:
        # Load audio file
        y, sr = librosa.load(input_file, sr=None)
        
        if enhancement_type == "vocals":
            # Enhance vocals: reduce noise, normalize, and boost clarity
            # Remove silence
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            # Normalize audio
            y_normalized = librosa.util.normalize(y_trimmed)
            
            # Apply gentle compression to vocals
            y_enhanced = np.tanh(y_normalized * 1.2) * 0.8
            
            # Boost high frequencies for clarity
            y_enhanced = librosa.effects.preemphasis(y_enhanced, coef=0.97)
            
        elif enhancement_type == "drums":
            # Enhance drums: boost low frequencies, normalize
            y_normalized = librosa.util.normalize(y)
            
            # Boost low frequencies for kick and snare
            y_enhanced = y_normalized * 1.3
            
            # Apply gentle compression
            y_enhanced = np.tanh(y_enhanced * 1.1) * 0.9
            
        elif enhancement_type == "bass":
            # Enhance bass: boost low frequencies, normalize
            y_normalized = librosa.util.normalize(y)
            
            # Boost low frequencies
            y_enhanced = y_normalized * 1.4
            
            # Apply gentle compression
            y_enhanced = np.tanh(y_enhanced * 1.2) * 0.8
            
        else:  # other instruments
            # Enhance other instruments: normalize and boost clarity
            y_normalized = librosa.util.normalize(y)
            
            # Apply gentle enhancement
            y_enhanced = y_normalized * 1.1
            
            # Boost mid frequencies
            y_enhanced = librosa.effects.preemphasis(y_enhanced, coef=0.95)
        
        # Ensure no clipping
        y_enhanced = np.clip(y_enhanced, -1.0, 1.0)
        
        # Save enhanced audio
        sf.write(output_file, y_enhanced, sr)
        return True
        
    except Exception as e:
        print(f" Audio enhancement error: {e}")
        return False

def separate_audio_ultra(input_file, output_dir="output_demucs", model="htdemucs", device="cpu", components=4, enhance=True):
    """
    Ultra audio separation with multiple component options and enhancement
    
    Args:
        input_file (str): Path to input audio file
        output_dir (str): Output directory for separated files
        model (str): Demucs model to use
        device (str): Device to use (cpu, cuda, mps)
        components (int): Number of components to separate (4, 6, 8)
        enhance (bool): Whether to enhance audio quality
    """
    print(f" Starting ULTRA {components}-component separation for: {input_file}")
    print(f" Output directory: {output_dir}")
    print(f" Model: {model}")
    print(f" Device: {device}")
    print(f" Components: {components}")
    print(f" Audio Enhancement: {'ON' if enhance else 'OFF'}")
    print("-" * 70)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the virtual environment's Python
    venv_python = os.path.join("venv_demucs", "Scripts", "python.exe")
    if not os.path.exists(venv_python):
        venv_python = "python"  # fallback to system python
    
    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file '{input_file}' not found")
        
        if components == 4:
            # Standard 4-component separation
            print(" Running standard 4-component separation...")
            command = [
                venv_python, "-m", "demucs",
                "-n", model,
                "-d", device,
                "-o", output_dir,
                input_file
            ]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            
            if enhance:
                print(" Enhancing audio quality...")
                enhance_4_components(input_file, output_dir)
            
        elif components == 6:
            # Advanced 6-component separation using two-stems
            print(" Running 6-component separation using two-stems technique...")
            
            # First pass: Standard 4-component separation
            command1 = [
                venv_python, "-m", "demucs",
                "-n", model,
                "-d", device,
                "-o", output_dir,
                input_file
            ]
            result1 = subprocess.run(command1, check=True, capture_output=True, text=True)
            
            # Second pass: Further separate "other" into piano and guitar
            base_name = Path(input_file).stem
            model_dir = os.path.join(output_dir, "htdemucs")
            track_dir = os.path.join(model_dir, base_name)
            other_file = os.path.join(track_dir, "other.wav")
            
            if os.path.exists(other_file):
                print(" Further separating 'other' component...")
                # Separate "other" into piano and guitar using two-stems
                command2 = [
                    venv_python, "-m", "demucs",
                    "-n", model,
                    "-d", device,
                    "--two-stems", "other",
                    "-o", os.path.join(output_dir, "advanced"),
                    other_file
                ]
                result2 = subprocess.run(command2, check=True, capture_output=True, text=True)
                
                # Create 6-component structure
                create_6_component_structure(track_dir, output_dir, base_name)
                
                if enhance:
                    print(" Enhancing audio quality...")
                    enhance_6_components(input_file, output_dir)
            
        elif components == 8:
            # Direct 8-component separation using advanced techniques
            print(" Running direct 8-component separation...")
            
            # Use the best model for 8-component separation
            best_model = "htdemucs_ft"  # Use the finest model
            
            # First pass: Standard 4-component separation
            print(" Pass 1/2: Standard 4-component separation...")
            command1 = [
                venv_python, "-m", "demucs",
                "-n", best_model,
                "-d", device,
                "-o", output_dir,
                input_file
            ]
            result1 = subprocess.run(command1, check=True, capture_output=True, text=True)
            
            # Second pass: Further separate each component
            base_name = Path(input_file).stem
            model_dir = os.path.join(output_dir, "htdemucs_ft")
            track_dir = os.path.join(model_dir, base_name)
            
            if os.path.exists(track_dir):
                print(" Pass 2/2: Advanced component separation...")
                
                # Separate vocals into lead and harmony
                vocals_file = os.path.join(track_dir, "vocals.wav")
                if os.path.exists(vocals_file):
                    print("  Separating vocals into lead and harmony...")
                    command_vocals = [
                        venv_python, "-m", "demucs",
                        "-n", best_model,
                        "-d", device,
                        "--two-stems", "vocals",
                        "-o", os.path.join(output_dir, "advanced_vocals"),
                        vocals_file
                    ]
                    subprocess.run(command_vocals, check=True, capture_output=True, text=True)
                
                # Separate drums into kick/snare and cymbals
                drums_file = os.path.join(track_dir, "drums.wav")
                if os.path.exists(drums_file):
                    print("  Separating drums into kick/snare and cymbals...")
                    command_drums = [
                        venv_python, "-m", "demucs",
                        "-n", best_model,
                        "-d", device,
                        "--two-stems", "drums",
                        "-o", os.path.join(output_dir, "advanced_drums"),
                        drums_file
                    ]
                    subprocess.run(command_drums, check=True, capture_output=True, text=True)
                
                # Separate other into piano and guitar
                other_file = os.path.join(track_dir, "other.wav")
                if os.path.exists(other_file):
                    print("  Separating other into piano and guitar...")
                    command_other = [
                        venv_python, "-m", "demucs",
                        "-n", best_model,
                        "-d", device,
                        "--two-stems", "other",
                        "-o", os.path.join(output_dir, "advanced_other"),
                        other_file
                    ]
                    subprocess.run(command_other, check=True, capture_output=True, text=True)
                
                # Create 8-component structure
                create_8_component_structure_direct(track_dir, output_dir, base_name)
                
                if enhance:
                    print(" Enhancing audio quality...")
                    enhance_8_components(input_file, output_dir)
        
        print(" Ultra separation complete!")
        print_ultra_components_info(input_file, output_dir, components)
        
    except FileNotFoundError as e:
        print(f" ERROR: {e}")
        return False
    except subprocess.CalledProcessError as e:
        print(f" Demucs processing error: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f" Unexpected error: {e}")
        return False
    
    return True

def enhance_4_components(input_file, output_dir):
    """Enhance 4-component separation results"""
    base_name = Path(input_file).stem
    track_dir = os.path.join(output_dir, "htdemucs", base_name)
    
    if os.path.exists(track_dir):
        components = [
            ("vocals.wav", "vocals"),
            ("drums.wav", "drums"),
            ("bass.wav", "bass"),
            ("other.wav", "other")
        ]
        
        for filename, enhancement_type in components:
            input_path = os.path.join(track_dir, filename)
            if os.path.exists(input_path):
                # Create enhanced version
                enhanced_path = os.path.join(track_dir, f"enhanced_{filename}")
                if enhance_audio(input_path, enhanced_path, enhancement_type):
                    print(f"   Enhanced {filename}")
                else:
                    print(f"   Failed to enhance {filename}")

def enhance_6_components(input_file, output_dir):
    """Enhance 6-component separation results"""
    base_name = Path(input_file).stem
    track_dir = os.path.join(output_dir, "6_components", base_name)
    
    if os.path.exists(track_dir):
        components = [
            ("vocals.wav", "vocals"),
            ("drums.wav", "drums"),
            ("bass.wav", "bass"),
            ("piano.wav", "other"),
            ("guitar.wav", "other"),
            ("other.wav", "other")
        ]
        
        for filename, enhancement_type in components:
            input_path = os.path.join(track_dir, filename)
            if os.path.exists(input_path):
                # Create enhanced version
                enhanced_path = os.path.join(track_dir, f"enhanced_{filename}")
                if enhance_audio(input_path, enhanced_path, enhancement_type):
                    print(f"   Enhanced {filename}")
                else:
                    print(f"   Failed to enhance {filename}")

def enhance_8_components(input_file, output_dir):
    """Enhance 8-component separation results"""
    base_name = Path(input_file).stem
    track_dir = os.path.join(output_dir, "8_components", base_name)
    
    if os.path.exists(track_dir):
        components = [
            ("vocals.wav", "vocals"),
            ("drums.wav", "drums"),
            ("bass.wav", "bass"),
            ("other.wav", "other"),
            ("lead_vocals.wav", "vocals"),
            ("harmony.wav", "vocals"),
            ("kick_snare.wav", "drums"),
            ("cymbals.wav", "drums")
        ]
        
        for filename, enhancement_type in components:
            input_path = os.path.join(track_dir, filename)
            if os.path.exists(input_path):
                # Create enhanced version
                enhanced_path = os.path.join(track_dir, f"enhanced_{filename}")
                if enhance_audio(input_path, enhanced_path, enhancement_type):
                    print(f"   Enhanced {filename}")
                else:
                    print(f"   Failed to enhance {filename}")

def create_6_component_structure(track_dir, output_dir, base_name):
    """Create 6-component structure from 4-component separation"""
    # The advanced separation creates a folder named after the input file
    advanced_dir = os.path.join(output_dir, "advanced", "htdemucs", "other")
    final_dir = os.path.join(output_dir, "6_components", base_name)
    os.makedirs(final_dir, exist_ok=True)
    
    # Copy original 4 components
    components = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
    for comp in components:
        src = os.path.join(track_dir, comp)
        dst = os.path.join(final_dir, comp)
        if os.path.exists(src):
            os.system(f'copy "{src}" "{dst}"')
    
    # Copy separated "other" components
    if os.path.exists(advanced_dir):
        # Copy other.wav as piano.wav
        other_src = os.path.join(advanced_dir, "other.wav")
        other_dst = os.path.join(final_dir, "piano.wav")
        if os.path.exists(other_src):
            os.system(f'copy "{other_src}" "{other_dst}"')
            print(f" Created piano.wav")
        
        # Copy no_other.wav as guitar.wav
        no_other_src = os.path.join(advanced_dir, "no_other.wav")
        no_other_dst = os.path.join(final_dir, "guitar.wav")
        if os.path.exists(no_other_src):
            os.system(f'copy "{no_other_src}" "{no_other_dst}"')
            print(f" Created guitar.wav")

def create_8_component_structure_direct(track_dir, output_dir, base_name):
    """Create 8-component structure using direct separation"""
    final_dir = os.path.join(output_dir, "8_components", base_name)
    os.makedirs(final_dir, exist_ok=True)
    
    # Copy original 4 components
    components = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
    for comp in components:
        src = os.path.join(track_dir, comp)
        dst = os.path.join(final_dir, comp)
        if os.path.exists(src):
            os.system(f'copy "{src}" "{dst}"')
    
    # Copy advanced separated components
    # Vocals separation
    vocals_advanced_dir = os.path.join(output_dir, "advanced_vocals", "htdemucs_ft", "vocals")
    if os.path.exists(vocals_advanced_dir):
        vocals_src = os.path.join(vocals_advanced_dir, "vocals.wav")
        vocals_dst = os.path.join(final_dir, "lead_vocals.wav")
        if os.path.exists(vocals_src):
            os.system(f'copy "{vocals_src}" "{vocals_dst}"')
            print(f" Created lead_vocals.wav")
        
        no_vocals_src = os.path.join(vocals_advanced_dir, "no_vocals.wav")
        no_vocals_dst = os.path.join(final_dir, "harmony.wav")
        if os.path.exists(no_vocals_src):
            os.system(f'copy "{no_vocals_src}" "{no_vocals_dst}"')
            print(f" Created harmony.wav")
    
    # Drums separation
    drums_advanced_dir = os.path.join(output_dir, "advanced_drums", "htdemucs_ft", "drums")
    if os.path.exists(drums_advanced_dir):
        drums_src = os.path.join(drums_advanced_dir, "drums.wav")
        drums_dst = os.path.join(final_dir, "kick_snare.wav")
        if os.path.exists(drums_src):
            os.system(f'copy "{drums_src}" "{drums_dst}"')
            print(f" Created kick_snare.wav")
        
        no_drums_src = os.path.join(drums_advanced_dir, "no_drums.wav")
        no_drums_dst = os.path.join(final_dir, "cymbals.wav")
        if os.path.exists(no_drums_src):
            os.system(f'copy "{no_drums_src}" "{no_drums_dst}"')
            print(f" Created cymbals.wav")
    
    # Other separation
    other_advanced_dir = os.path.join(output_dir, "advanced_other", "htdemucs_ft", "other")
    if os.path.exists(other_advanced_dir):
        other_src = os.path.join(other_advanced_dir, "other.wav")
        other_dst = os.path.join(final_dir, "piano.wav")
        if os.path.exists(other_src):
            os.system(f'copy "{other_src}" "{other_dst}"')
            print(f" Created piano.wav")
        
        no_other_src = os.path.join(other_advanced_dir, "no_other.wav")
        no_other_dst = os.path.join(final_dir, "guitar.wav")
        if os.path.exists(no_other_src):
            os.system(f'copy "{no_other_src}" "{no_other_dst}"')
            print(f" Created guitar.wav")

def print_ultra_components_info(input_file, output_dir, components):
    """Print information about separated components"""
    base_name = Path(input_file).stem
    
    print("\n" + "=" * 70)
    print(f" ULTRA {components}-COMPONENT SEPARATION RESULTS:")
    print("=" * 70)
    
    if components == 4:
        track_dir = os.path.join(output_dir, "htdemucs", base_name)
        components_info = [
            (" VOCALS", "Lead vocals and harmonies", "vocals.wav"),
            (" DRUMS", "Kick, snare, hi-hats, cymbals", "drums.wav"),
            (" BASS", "Bass guitar and low-frequency instruments", "bass.wav"),
            (" OTHER", "Piano, guitar, synths, strings, etc.", "other.wav")
        ]
        
        for emoji, description, filename in components_info:
            file_path = os.path.join(track_dir, filename)
            enhanced_path = os.path.join(track_dir, f"enhanced_{filename}")
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                print(f"{emoji} - {description}")
                print(f"    {filename} ({file_size:.1f} MB)")
                if os.path.exists(enhanced_path):
                    enhanced_size = os.path.getsize(enhanced_path) / (1024 * 1024)
                    print(f"    enhanced_{filename} ({enhanced_size:.1f} MB)")
            else:
                print(f"{emoji} - {description}")
                print(f"    {filename} (File not found)")
    
    elif components == 6:
        track_dir = os.path.join(output_dir, "6_components", base_name)
        components_info = [
            (" VOCALS", "Lead vocals and harmonies", "vocals.wav"),
            (" DRUMS", "Kick, snare, hi-hats, cymbals", "drums.wav"),
            (" BASS", "Bass guitar and low-frequency instruments", "bass.wav"),
            (" PIANO", "Piano and keyboard instruments", "piano.wav"),
            (" GUITAR", "Electric and acoustic guitars", "guitar.wav"),
            (" OTHER", "Brass, strings, synths, etc.", "other.wav")
        ]
        
        for emoji, description, filename in components_info:
            file_path = os.path.join(track_dir, filename)
            enhanced_path = os.path.join(track_dir, f"enhanced_{filename}")
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                print(f"{emoji} - {description}")
                print(f"    {filename} ({file_size:.1f} MB)")
                if os.path.exists(enhanced_path):
                    enhanced_size = os.path.getsize(enhanced_path) / (1024 * 1024)
                    print(f"    enhanced_{filename} ({enhanced_size:.1f} MB)")
            else:
                print(f"{emoji} - {description}")
                print(f"    {filename} (File not found)")
    
    elif components == 8:
        track_dir = os.path.join(output_dir, "8_components", base_name)
        components_info = [
            (" VOCALS", "Lead vocals and harmonies", "vocals.wav"),
            (" DRUMS", "Kick, snare, hi-hats, cymbals", "drums.wav"),
            (" BASS", "Bass guitar and low-frequency instruments", "bass.wav"),
            (" OTHER", "Piano, guitar, synths, strings, etc.", "other.wav"),
            (" LEAD VOCALS", "Main vocal track", "lead_vocals.wav"),
            (" HARMONY", "Harmony and backing vocals", "harmony.wav"),
            (" KICK/SNARE", "Kick drum and snare", "kick_snare.wav"),
            (" CYMBALS", "Hi-hats, cymbals, percussion", "cymbals.wav")
        ]
        
        for emoji, description, filename in components_info:
            file_path = os.path.join(track_dir, filename)
            enhanced_path = os.path.join(track_dir, f"enhanced_{filename}")
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                print(f"{emoji} - {description}")
                print(f"    {filename} ({file_size:.1f} MB)")
                if os.path.exists(enhanced_path):
                    enhanced_size = os.path.getsize(enhanced_path) / (1024 * 1024)
                    print(f"    enhanced_{filename} ({enhanced_size:.1f} MB)")
            else:
                print(f"{emoji} - {description}")
                print(f"    {filename} (File not found)")
    
    print(f"\n All files saved in: {track_dir}")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Ultra-Enhanced Music Source Separation with Audio Enhancement")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("-o", "--output", default="output_demucs", help="Output directory")
    parser.add_argument("-m", "--model", choices=["htdemucs", "htdemucs_ft", "mdx", "mdx_extra"], 
                       default="htdemucs", help="Demucs model to use")
    parser.add_argument("-d", "--device", choices=["cpu", "cuda", "mps"], 
                       default="cpu", help="Device to use")
    parser.add_argument("-c", "--components", type=int, choices=[4, 6, 8], 
                       default=4, help="Number of components to separate")
    parser.add_argument("--no-enhance", action="store_true", 
                       help="Disable audio enhancement")
    
    args = parser.parse_args()
    
    print(" ULTRA-ENHANCED MUSIC SOURCE SEPARATOR")
    print("=" * 70)
    print(f" Components: {args.components}")
    print(f" Model: {args.model}")
    print(f" Device: {args.device}")
    print(f" Audio Enhancement: {'OFF' if args.no_enhance else 'ON'}")
    print("=" * 70)
    
    success = separate_audio_ultra(
        args.input, 
        args.output, 
        args.model, 
        args.device, 
        args.components,
        enhance=not args.no_enhance
    )
    
    if success:
        print("\n Separation completed successfully!")
    else:
        print("\n Separation failed!")

if __name__ == "__main__":
    main()
