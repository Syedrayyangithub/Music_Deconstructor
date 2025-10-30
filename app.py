import os
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, Response, session, current_app, flash, jsonify
from werkzeug.utils import secure_filename
import uuid
from pathlib import Path
import shutil

# Import the core separation function from your script
from music_separator import (
    separate_audio_ultra, get_separation_results, ACTIVE_PROCESSES,
    get_bpm, time_stretch_audio, fuse_stems # <-- NEW IMPORTS
)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output_demucs'
TEMP_FOLDER = 'temp_fusion' # <-- NEW FOLDER FOR STRETCHED FILES
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a', 'aac'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER # <-- NEW
app.secret_key = 'super_secret_key'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True) # <-- NEW

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # ... (This function is unchanged and correct) ...
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return "Invalid file", 400

    original_filename = secure_filename(file.filename)
    unique_filename = f"{uuid.uuid4()}_{original_filename}"
    input_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(input_path)

    components = int(request.form.get('components'))
    model = request.form.get('model')
    enhance = request.form.get('enhance') == 'on'
    silence_threshold_db = int(request.form.get('silence_threshold_db', 30))
    
    session['last_file'] = {
        "filename": unique_filename,
        "components": components,
        "model": model,
        "silence_threshold_db": silence_threshold_db
    }

    output_dir_for_later = current_app.config['OUTPUT_FOLDER']

    def generate_progress(output_dir):
        try:
            for progress_line in separate_audio_ultra(
                input_path,
                output_dir=output_dir,
                model=model,
                components=components,
                enhance=enhance,
                silence_threshold_db=silence_threshold_db,
                progress_callback=lambda line: f"data: {line}\n\n"
            ):
                if progress_line:
                    yield progress_line
            
            # Send back the unique filename AND the original filename
            yield f"data: SEPARATION_COMPLETE::{unique_filename}::{original_filename}\n\n"
            
        except Exception as e:
            yield f"data: ERROR::{str(e)}\n\n"

    return Response(generate_progress(output_dir_for_later), mimetype='text/event-stream')

@app.route('/results-for-file', methods=['POST'])
def results_for_file():
    # ... (This function is unchanged and correct) ...
    data = request.get_json()
    unique_filename = data.get('unique_filename')
    original_filename = data.get('original_filename')
    
    input_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
    
    results = get_separation_results(
        input_path,
        current_app.config['OUTPUT_FOLDER'],
        int(data.get('components')),
        data.get('model')
    )
    
    # We pass the unique_filename to the partial template for the data attributes
    rendered_html = render_template(
        'results_partial.html', 
        results=results, 
        filename=original_filename, 
        unique_filename=unique_filename,
        model=data.get('model'),
        components=int(data.get('components'))
    )
    return jsonify({'html': rendered_html})

# -----------------------------------------------------------------
#  NEW FUSION ROUTE
# -----------------------------------------------------------------
@app.route('/fuse', methods=['POST'])
def fuse_tracks():
    data = request.get_json()
    fusion_map = data.get('fusion_map') # e.g., {"vocals": "uuid_song_a", "drums": "uuid_song_b", ...}
    master_tempo_song_id = data.get('master_tempo_song_id') # e.g., "uuid_song_a"
    model_map = data.get('model_map') # e.g., {"uuid_song_a": "htdemucs", ...}
    components_map = data.get('components_map') # e.g., {"uuid_song_a": 4, ...}

    # 1. Determine the Master BPM
    master_song_input_path = os.path.join(current_app.config['UPLOAD_FOLDER'], master_tempo_song_id)
    if not os.path.exists(master_song_input_path):
        return jsonify({"error": "Master tempo song not found"}), 404
        
    master_bpm = get_bpm(master_song_input_path)
    if master_bpm == 0:
        return jsonify({"error": "Could not detect master BPM"}), 400

    print(f"--- STARTING FUSION ---")
    print(f"Master Tempo: {master_bpm} BPM (from {master_tempo_song_id})")

    stems_to_fuse = [] # This will hold paths to the *stretched* stems
    temp_dir = current_app.config['TEMP_FOLDER']
    
    # 2. Iterate, Stretch, and Collect Stems
    for stem_name, song_id in fusion_map.items():
        if not song_id: # Skip if user selected "None" for a stem
            continue
            
        print(f"Processing stem: {stem_name} from song: {song_id}")
        
        # Find the original (non-stretched) stem file
        model = model_map.get(song_id)
        components = components_map.get(song_id)
        
        # This is a bit of a hack to re-create the stem path.
        # We call get_separation_results just to find the one file we need.
        input_path = os.path.join(current_app.config['UPLOAD_FOLDER'], song_id)
        all_stems = get_separation_results(input_path, current_app.config['OUTPUT_FOLDER'], components, model)
        
        original_stem_path = None
        for s in all_stems:
            # s['name'] is "Vocals", stem_name is "vocals"
            if s['name'].lower() == stem_name:
                original_stem_path = os.path.join(current_app.config['OUTPUT_FOLDER'], s['path'])
                break
        
        if not original_stem_path or not os.path.exists(original_stem_path):
            print(f"  Could not find original stem for {stem_name} from {song_id}")
            continue

        # 3. Time-Stretch if necessary
        stretched_stem_path = os.path.join(temp_dir, f"{song_id}_{stem_name}.wav")
        
        if song_id == master_tempo_song_id:
            print("  At master tempo. Copying.")
            shutil.copy(original_stem_path, stretched_stem_path)
        else:
            time_stretch_audio(original_stem_path, stretched_stem_path, master_bpm)
        
        stems_to_fuse.append(stretched_stem_path)

    if not stems_to_fuse:
        return jsonify({"error": "No stems were selected for fusion"}), 400

    # 4. Fuse the stretched stems
    fused_filename = f"fused_{uuid.uuid4()}.mp3"
    fused_output_path_relative = os.path.join("fused_tracks", fused_filename)
    fused_output_path_full = os.path.join(current_app.config['OUTPUT_FOLDER'], fused_output_path_relative)
    
    os.makedirs(os.path.dirname(fused_output_path_full), exist_ok=True)
    
    if not fuse_stems(stems_to_fuse, fused_output_path_full):
        return jsonify({"error": "Failed to fuse stems"}), 500

    # 5. Clean up temporary stretched files
    for p in stems_to_fuse:
        os.remove(p)

    print(f"--- FUSION COMPLETE ---")
    
    return jsonify({
        "success": True, 
        "path": fused_output_path_relative.replace('\\', '/'),
        "filename": "Your Fused Track"
    })

# -----------------------------------------------------------------
#  EXISTING ROUTES
# -----------------------------------------------------------------

@app.route('/download/<path:filepath>')
def download_file(filepath):
    return send_from_directory(current_app.config['OUTPUT_FOLDER'], filepath, as_attachment=True)

@app.route('/play/<path:filepath>')
def play_file(filepath):
    """Serves an audio file for in-browser playback."""
    # We now serve from OUTPUT_FOLDER, which contains all stems AND fused tracks
    return send_from_directory(current_app.config['OUTPUT_FOLDER'], filepath)

@app.route('/clear-files', methods=['POST'])
def clear_files():
    upload_folder = current_app.config['UPLOAD_FOLDER']
    output_folder = current_app.config['OUTPUT_FOLDER']
    temp_folder = current_app.config['TEMP_FOLDER'] # <-- NEW
    
    folders_to_clear = [upload_folder, output_folder, temp_folder] # <-- NEW
    
    for folder in folders_to_clear:
        if not os.path.exists(folder):
            continue
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Failed to delete {item_path}. Reason: {e}")
                flash(f"Error deleting some files: {e}", "error")
    
    # Re-create dirs
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(TEMP_FOLDER, exist_ok=True)

    session.pop('last_file', None)
    flash("All temporary files and outputs have been cleared.", "success")
    return redirect(url_for('index'))

@app.route('/cancel', methods=['POST'])
def cancel_process():
    # ... (This function is unchanged and correct) ...
    last_file_info = session.get('last_file')
    if not last_file_info:
        return {"status": "error", "message": "No active job found in session."}, 404

    job_id = last_file_info['filename']
    process_to_kill = ACTIVE_PROCESSES.get(job_id)

    if process_to_kill and process_to_kill.poll() is None:
        process_to_kill.kill()
        return {"status": "success", "message": f"Process {job_id} cancelled."}, 200
    
    return {"status": "error", "message": "Process not found or already finished."}, 404

if __name__ == '__main__':
    app.run(debug=True, threaded=True)