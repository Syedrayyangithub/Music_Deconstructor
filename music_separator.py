import sys
import os
import subprocess
import glob
import argparse
from pathlib import Path
from typing import Callable, Optional
import librosa
import soundfile as sf
import shutil
import numpy as np
from pydub import AudioSegment # <-- NEW IMPORT

# A global dictionary to keep track of active subprocesses by a unique job ID
ACTIVE_PROCESSES = {}

# -----------------------------------------------------------------
#  NEW FUSION FUNCTIONS
# -----------------------------------------------------------------

def get_bpm(audio_file) -> float:
    """Loads an audio file and estimates its BPM."""
    try:
        print(f"   Analysing BPM for {audio_file}...")
        y, sr = librosa.load(audio_file, sr=None)
        # Use a larger hop_length for faster analysis
        onset_env = librosa.onset.onset_detect(y=y, sr=sr, hop_length=1024)
        bpm = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)["bpm"]
        print(f"   Detected BPM: {bpm}")
        return float(bpm)
    except Exception as e:
        print(f"   Could not detect BPM: {e}. Defaulting to 120.")
        return 120.0

def time_stretch_audio(input_file, output_file, target_bpm: float):
    """
    Time-stretches an audio file to a target BPM.
    Returns True on success, False on failure.
    """
    try:
        print(f"   Stretching {Path(input_file).name} to {target_bpm} BPM...")
        y, sr = librosa.load(input_file, sr=None)
        source_bpm = get_bpm(input_file)
        
        if source_bpm == 0:
            print("   Source BPM is 0, cannot stretch.")
            return False

        # The 'rate' is the factor by which to speed up or slow down
        # e.g., target=120, source=60 -> rate=2.0 (speed up)
        # e.g., target=90, source=120 -> rate=0.75 (slow down)
        stretch_rate = target_bpm / source_bpm

        if np.abs(stretch_rate - 1.0) < 0.01:
            print("   File is already at target BPM. Copying.")
            shutil.copy(input_file, output_file)
            return True

        y_stretched = librosa.effects.time_stretch(y, rate=stretch_rate)
        
        sf.write(output_file, y_stretched, sr)
        print(f"   Successfully stretched and saved to {output_file}")
        return True
    except Exception as e:
        print(f"   Error during time-stretch: {e}")
        return False

def fuse_stems(stem_paths: list, output_file: str):
    """
    Mixes a list of audio file paths into a single output file.
    All stems must be .wav files.
    """
    try:
        print(f"   Fusing {len(stem_paths)} stems...")
        
        # Load the first stem as the base
        base = AudioSegment.from_wav(stem_paths[0])

        # Overlay the rest of the stems
        for path in stem_paths[1:]:
            stem_to_overlay = AudioSegment.from_wav(path)
            # We assume all stems are the same length after stretching
            base = base.overlay(stem_to_overlay)
        
        # Export the final mixed file
        base.export(output_file, format="mp3", bitrate="320k") # Export as high-quality MP3
        print(f"   Fusion complete! Saved to {output_file}")
        return True
    except Exception as e:
        print(f"   Error during stem fusion: {e}")
        return False

# -----------------------------------------------------------------
#  EXISTING SEPARATION & ENHANCEMENT FUNCTIONS (WITH FIXES)
# -----------------------------------------------------------------

def enhance_audio(input_file, output_file, enhancement_type="vocals", silence_threshold_db: int = 30):
    try:
        y, sr = librosa.load(input_file, sr=None)
        y_trimmed, index = librosa.effects.trim(y, top_db=silence_threshold_db)

        if y_trimmed.size == 0:
            print(f"   Skipping empty file: {Path(input_file).name}")
            return False
        
        if enhancement_type == "vocals":
            y_normalized = librosa.util.normalize(y_trimmed)
            y_enhanced = np.tanh(y_normalized * 1.2) * 0.8
            y_enhanced = librosa.effects.preemphasis(y_enhanced, coef=0.97)
        elif enhancement_type == "drums":
            y_normalized = librosa.util.normalize(y_trimmed)
            y_enhanced = y_normalized * 1.3
            y_enhanced = np.tanh(y_enhanced * 1.1) * 0.9
        elif enhancement_type == "bass":
            y_normalized = librosa.util.normalize(y_trimmed)
            y_enhanced = y_normalized * 1.4
            y_enhanced = np.tanh(y_enhanced * 1.2) * 0.8
        else:  # other instruments
            y_normalized = librosa.util.normalize(y_trimmed)
            y_enhanced = y_normalized * 1.1
            y_enhanced = librosa.effects.preemphasis(y_enhanced, coef=0.95)
        
        y_enhanced = np.clip(y_enhanced, -1.0, 1.0)
        sf.write(output_file, y_enhanced, sr)
        return True
    except Exception as e:
        print(f" Audio enhancement error: {e}")
        return False

def _get_enhancement_map():
    return {
        "vocals": "vocals", "drums": "drums", "bass": "bass", "other": "other",
        "piano": "other", "guitar": "other", "lead_vocals": "vocals",
        "harmony": "vocals", "kick_snare": "drums", "cymbals": "drums"
    }

def _run_demucs(venv_python, input_file, output_dir, model, device, two_stems_target=None, job_id: Optional[str] = None, progress_callback: Optional[Callable[[str], None]] = None):
    command = [
        venv_python, "-m", "demucs",
        "-n", model, "-d", device, "-o", output_dir,
    ]
    if two_stems_target:
        command.extend(["--two-stems", two_stems_target])
    
    command.append(input_file)

    print(f" Running Demucs on '{Path(input_file).name}'...")
    proc = None
    try:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        
        if job_id:
            ACTIVE_PROCESSES[job_id] = proc

        if proc.stderr:
            for line in iter(proc.stderr.readline, ''):
                line = line.strip()
                if progress_callback:
                    yield progress_callback(line)
                else:
                    print(line.strip())
        
        stdout, stderr = proc.communicate()

        if proc.returncode != 0 and proc.returncode != -9:
            print(f"Demucs Error: {stderr}")
            raise subprocess.CalledProcessError(proc.returncode, command, output=stdout, stderr=stderr)
    finally:
        if job_id and job_id in ACTIVE_PROCESSES:
            del ACTIVE_PROCESSES[job_id]
        if proc and proc.poll() is None:
            proc.kill()

def separate_audio_ultra(input_file, output_dir="output_demucs", model="htdemucs", device="cpu", components=4, enhance=True, silence_threshold_db: int = 30, progress_callback: Optional[Callable[[str], None]] = None):
    print(f" Starting ULTRA {components}-component separation for: {input_file}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    venv_python = sys.executable
    if "venv_demucs" not in venv_python:
        print("Warning: Not running from 'venv_demucs'. Using system python.")
    
    try:
        if not os.path.exists(input_file):
            if progress_callback: yield progress_callback(f"ERROR: Input file '{input_file}' not found")
            return
            
        job_id = Path(input_file).name
        
        if components == 4:
            yield from _separate_4_components(venv_python, input_file, output_dir, model, device, enhance, silence_threshold_db, job_id, progress_callback=progress_callback)
        elif components == 6:
            yield from _separate_6_components(venv_python, input_file, output_dir, model, device, enhance, silence_threshold_db, job_id, progress_callback=progress_callback)
        elif components == 8:
            yield from _separate_8_components(venv_python, input_file, output_dir, device, enhance, silence_threshold_db, job_id, progress_callback=progress_callback)
        
        print(" Ultra separation complete!")
        used_model = "htdemucs_ft" if components == 8 else model
        print_ultra_components_info(input_file, output_dir, components, used_model)
        
    except FileNotFoundError as e:
        if progress_callback: yield progress_callback(f"ERROR: {e}")
    except subprocess.CalledProcessError as e:
        if e.returncode != -9:
            error_message = f"Demucs processing error: {e.stderr}"
            if progress_callback: yield progress_callback(f"ERROR: {error_message}")
    except Exception as e:
        if progress_callback: yield progress_callback(f"ERROR: An unexpected error occurred: {e}")

def _separate_4_components(venv_python, input_file, output_dir, model, device, enhance, silence_threshold_db, job_id: str, progress_callback: Optional[Callable[[str], None]] = None):
    yield from _run_demucs(venv_python, input_file, output_dir, model, device, job_id=job_id, progress_callback=progress_callback)
    if enhance:
        print(" Enhancing audio quality...")
        enhance_4_components(input_file, output_dir, model, silence_threshold_db)

def _separate_6_components(venv_python, input_file, output_dir, model, device, enhance, silence_threshold_db, job_id: str, progress_callback: Optional[Callable[[str], None]] = None):
    if progress_callback: yield progress_callback("Running 6-component separation (Pass 1/2)...")
    
    yield from _run_demucs(venv_python, input_file, output_dir, model, device, job_id=job_id, progress_callback=progress_callback)

    base_name = Path(input_file).stem
    track_dir = os.path.join(output_dir, model, base_name)
    other_file = os.path.join(track_dir, "other.wav")
    
    if os.path.exists(other_file):
        if progress_callback: yield progress_callback("Further separating 'other' component (Pass 2/2)...")
        advanced_output_dir = os.path.join(output_dir, "advanced")
        yield from _run_demucs(venv_python, other_file, advanced_output_dir, model, device, "other", job_id=f"{job_id}-other", progress_callback=progress_callback)
        
        create_6_component_structure(track_dir, output_dir, base_name, model)
        
        if enhance:
            if progress_callback: yield progress_callback("Enhancing audio quality...")
            enhance_6_components(input_file, output_dir, silence_threshold_db)

def _separate_8_components(venv_python, input_file, output_dir, device, enhance, silence_threshold_db, job_id: str, progress_callback: Optional[Callable[[str], None]] = None):
    best_model = "htdemucs_ft"
    print(f" Overriding model to '{best_model}' for highest quality 8-component separation.")
    
    print(" Pass 1/2: Standard 4-component separation...")
    yield from _run_demucs(venv_python, input_file, output_dir, best_model, device, job_id=job_id, progress_callback=progress_callback)
    
    base_name = Path(input_file).stem
    track_dir = os.path.join(output_dir, best_model, base_name)
    
    if not os.path.exists(track_dir):
        print(f" Warning: Initial separation directory not found at '{track_dir}'. Skipping advanced separation.")
        return

    if progress_callback: yield progress_callback("Pass 2/2: Advanced component separation...")
    stems_to_separate = {
        "vocals": "advanced_vocals", "drums": "advanced_drums", "other": "advanced_other"
    }
    for stem, adv_dir in stems_to_separate.items():
        stem_file = os.path.join(track_dir, f"{stem}.wav")
        if os.path.exists(stem_file):
            yield from _run_demucs(venv_python, stem_file, os.path.join(output_dir, adv_dir), best_model, device, stem, job_id=f"{job_id}-{stem}", progress_callback=progress_callback)

    create_8_component_structure_direct(track_dir, output_dir, base_name, best_model)
    if enhance:
        if progress_callback: yield progress_callback("Enhancing audio quality...")
        enhance_8_components(input_file, output_dir, silence_threshold_db)

def enhance_4_components(input_file, output_dir, model="htdemucs", silence_threshold_db: int = 30):
    base_name = Path(input_file).stem
    track_dir = os.path.join(output_dir, model, base_name)
    
    enhancement_map = _get_enhancement_map()
    if not os.path.exists(track_dir):
        return

    for stem_name in ["vocals", "drums", "bass", "other"]:
        filename = f"{stem_name}.wav"
        input_path = os.path.join(track_dir, filename)
        if os.path.exists(input_path):
            temp_enhanced_path = os.path.join(track_dir, f"temp_enhanced_{filename}")
            enhancement_type = enhancement_map.get(stem_name, "other")
            if enhance_audio(input_path, temp_enhanced_path, enhancement_type, silence_threshold_db):
                print(f"   Enhanced {filename}")
                os.remove(input_path)
                os.rename(temp_enhanced_path, input_path)
            else:
                print(f"   Skipped empty {filename} (after trimming), original file kept.")
                if os.path.exists(temp_enhanced_path):
                    os.remove(temp_enhanced_path)

def enhance_6_components(input_file, output_dir, silence_threshold_db: int = 30):
    base_name = Path(input_file).stem
    track_dir = os.path.join(output_dir, "6_components", base_name)
    
    enhancement_map = _get_enhancement_map()
    if not os.path.exists(track_dir):
        return

    for stem_name in ["vocals", "drums", "bass", "piano", "guitar", "other"]:
        filename = f"{stem_name}.wav"
        input_path = os.path.join(track_dir, filename)
        if os.path.exists(input_path):
            temp_enhanced_path = os.path.join(track_dir, f"temp_enhanced_{filename}")
            enhancement_type = enhancement_map.get(stem_name, "other")
            if enhance_audio(input_path, temp_enhanced_path, enhancement_type, silence_threshold_db):
                print(f"   Enhanced {filename}")
                os.remove(input_path)
                os.rename(temp_enhanced_path, input_path)
            else:
                print(f"   Skipped empty {filename} (after trimming), original file kept.")
                if os.path.exists(temp_enhanced_path):
                    os.remove(temp_enhanced_path)

def enhance_8_components(input_file, output_dir, silence_threshold_db: int = 30):
    base_name = Path(input_file).stem
    track_dir = os.path.join(output_dir, "8_components", base_name)
    
    enhancement_map = _get_enhancement_map()
    if not os.path.exists(track_dir):
        return

    stems_to_enhance = [
        "vocals", "drums", "bass", "other",
        "lead_vocals", "harmony", "kick_snare", "cymbals", "piano", "guitar"
    ]

    for stem_name in stems_to_enhance:
        filename = f"{stem_name}.wav"
        input_path = os.path.join(track_dir, filename)
        if os.path.exists(input_path):
            temp_enhanced_path = os.path.join(track_dir, f"temp_enhanced_{filename}")
            enhancement_type = enhancement_map.get(stem_name, "other")
            if enhance_audio(input_path, temp_enhanced_path, enhancement_type, silence_threshold_db):
                print(f"   Enhanced {filename}")
                os.remove(input_path)
                os.rename(temp_enhanced_path, input_path)
            else:
                print(f"   Skipped empty {filename} (after trimming), original file kept.")
                if os.path.exists(temp_enhanced_path):
                    os.remove(temp_enhanced_path)

def create_6_component_structure(track_dir, output_dir, base_name, model="htdemucs"):
    advanced_dir = os.path.join(output_dir, "advanced", model, "other")
    final_dir = os.path.join(output_dir, "6_components", base_name)
    os.makedirs(final_dir, exist_ok=True)
    
    components = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
    for comp in components:
        src = os.path.join(track_dir, comp)
        dst = os.path.join(final_dir, comp)
        if os.path.exists(src):
            shutil.copy(src, dst)
    
    if os.path.exists(advanced_dir):
        other_src = os.path.join(advanced_dir, "other.wav")
        other_dst = os.path.join(final_dir, "piano.wav")
        if os.path.exists(other_src):
            shutil.copy(other_src, other_dst)
            print(f" Created piano.wav")
        
        no_other_src = os.path.join(advanced_dir, "no_other.wav")
        no_other_dst = os.path.join(final_dir, "guitar.wav")
        if os.path.exists(no_other_src):
            shutil.copy(no_other_src, no_other_dst)
            print(f" Created guitar.wav")

def _copy_and_log(src_path, dst_path):
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f" Created {Path(dst_path).name}")

def create_8_component_structure_direct(track_dir, output_dir, base_name, model_for_advanced_stems):
    final_dir = os.path.join(output_dir, "8_components", base_name)
    os.makedirs(final_dir, exist_ok=True)
    
    file_map = [
        (track_dir, "vocals.wav", "vocals.wav"),
        (track_dir, "drums.wav", "drums.wav"),
        (track_dir, "bass.wav", "bass.wav"),
        (track_dir, "other.wav", "other.wav"),
        (os.path.join(output_dir, "advanced_vocals", model_for_advanced_stems, "vocals"), "vocals.wav", "lead_vocals.wav"),
        (os.path.join(output_dir, "advanced_vocals", model_for_advanced_stems, "vocals"), "no_vocals.wav", "harmony.wav"),
        (os.path.join(output_dir, "advanced_drums", model_for_advanced_stems, "drums"), "drums.wav", "kick_snare.wav"),
        (os.path.join(output_dir, "advanced_drums", model_for_advanced_stems, "drums"), "no_drums.wav", "cymbals.wav"),
        (os.path.join(output_dir, "advanced_other", model_for_advanced_stems, "other"), "other.wav", "piano.wav"),
        (os.path.join(output_dir, "advanced_other", model_for_advanced_stems, "other"), "no_other.wav", "guitar.wav"),
    ]

    for src_dir, src_filename, dst_filename in file_map:
        src_path = os.path.join(src_dir, src_filename)
        dst_path = os.path.join(final_dir, dst_filename)
        _copy_and_log(src_path, dst_path)

def get_separation_results(input_file, output_dir, components, model):
    base_name = Path(input_file).stem
    results = []
    
    if components == 4:
        relative_dir = os.path.join(model, base_name)
        track_dir = os.path.join(output_dir, relative_dir)
        files_to_check = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
    elif components == 6:
        relative_dir = os.path.join("6_components", base_name)
        track_dir = os.path.join(output_dir, relative_dir)
        files_to_check = ["vocals.wav", "drums.wav", "bass.wav", "piano.wav", "guitar.wav", "other.wav"]
    elif components == 8:
        relative_dir = os.path.join("8_components", base_name)
        track_dir = os.path.join(output_dir, relative_dir)
        files_to_check = [
            "vocals.wav", "drums.wav", "bass.wav", "other.wav",
            "lead_vocals.wav", "harmony.wav", "kick_snare.wav", "cymbals.wav", 
            "piano.wav", "guitar.wav"
        ]
    else:
        return []

    for filename in files_to_check:
        if os.path.exists(os.path.join(track_dir, filename)):
            results.append({
                "name": filename.replace('_', ' ').replace('.wav', '').title(),
                "path": os.path.join(relative_dir, filename).replace('\\', '/')
            })
            
    return results


def _print_component_info(track_dir, emoji, description, filename):
    file_path = os.path.join(track_dir, filename)
    
    print(f"{emoji} - {description}")
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        print(f"     {filename} ({file_size:.1f} MB)")
    else:
        print(f"     {filename} (File not found)")

def print_ultra_components_info(input_file, output_dir, components, model):
    base_name = Path(input_file).stem
    
    print("\n" + "=" * 70)
    print(f" ULTRA {components}-COMPONENT SEPARATION RESULTS:")
    print("=" * 70)
    
    track_dir = ""
    components_info = []

    if components == 4:
        track_dir = os.path.join(output_dir, model, base_name)
        components_info = [
            (" VOCALS", "Lead vocals and harmonies", "vocals.wav"),
            (" DRUMS", "Kick, snare, hi-hats, cymbals", "drums.wav"),
            (" BASS", "Bass guitar and low-frequency instruments", "bass.wav"),
            (" OTHER", "Piano, guitar, synths, strings, etc.", "other.wav")
        ]
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
            (" CYMBALS", "Hi-hats, cymbals, percussion", "cymbals.wav"),
            (" PIANO", "Piano and keyboard instruments", "piano.wav"),
            (" GUITAR", "Electric and acoustic guitars", "guitar.wav")
        ]
    
    if track_dir:
        for emoji, description, filename in components_info:
            _print_component_info(track_dir, emoji, description, filename)
        print(f"\n All files saved in: {track_dir}")
    else:
        print("   Error: Invalid component number.")
        
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Ultra-Enhanced Music Source Separation with Audio Enhancement")
    main.parser = parser

    parser.add_argument("input", help="Input audio file")
    # ... (rest of main function is unchanged and correct) ...
    
    args = parser.parse_args()
    
    # ... (rest of main function is unchanged and correct) ...
    
    for _ in separate_audio_ultra(
        args.input, 
        args.output, 
        args.model, 
        args.device, 
        args.components,
        enhance=not args.no_enhance,
        silence_threshold_db=args.silence_threshold,
        progress_callback=print
    ):
        pass
    
    print("\n Separation process finished.")

main.parser = None

if __name__ == "__main__":
    main()