import os
import subprocess

# --- IMPORTANT ---
# Make sure to place your audio file in the same folder as this script.
input_file = "7years.mp3"  # <--- Make sure this matches your file name

# ---
print(f"Starting separation for: {input_file} ...")

# Add "vocals" after "--two-stems". This is required by your version of demucs.
command = f"python -m demucs --two-stems vocals -o output_demucs \"{input_file}\""

# Run the command
try:
    if not os.path.exists(input_file):
        raise FileNotFoundError()

    subprocess.run(command, shell=True, check=True)
    print("--------------------------------------------------")
    print("✅ Separation complete!")
    print("Check the 'output_demucs' folder for your files.")
    print("--------------------------------------------------")

except FileNotFoundError:
    print(f"\nERROR: The input file '{input_file}' was not found.")
    print("Please make sure the audio file is in the same directory as this script.")

except subprocess.CalledProcessError as e:
    print(f"\nAn error occurred while running Demucs: {e}")
    print("Please ensure Demucs is installed correctly in your virtual environment.")