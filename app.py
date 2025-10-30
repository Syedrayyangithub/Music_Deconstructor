import os
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, Response, session, current_app, flash, jsonify
from werkzeug.utils import secure_filename
import uuid
from pathlib import Path
import shutil

# Import the core separation function from your script
from music_separator import separate_audio_ultra, get_separation_results, ACTIVE_PROCESSES

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output_demucs'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'flac', 'm4a', 'aac'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.secret_key = 'super_secret_key' # Needed for session

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
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

            yield f"data: SEPARATION_COMPLETE::{unique_filename}\n\n"
            
        except Exception as e:
            yield f"data: ERROR::{str(e)}\n\n"

    return Response(generate_progress(output_dir_for_later), mimetype='text/event-stream')

@app.route('/results')
def results_page():
    last_file_info = session.get('last_file')
    if not last_file_info:
        return redirect(url_for('index'))

    unique_filename = last_file_info['filename']
    input_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
    
    results = get_separation_results(
        input_path,
        current_app.config['OUTPUT_FOLDER'],
        last_file_info['components'],
        last_file_info['model']
    )
    original_filename = unique_filename.split('_', 1)[-1]
    return render_template('index.html', results=results, filename=original_filename)

@app.route('/results-for-file', methods=['POST'])
def results_for_file():
    """API endpoint to get the rendered HTML for a single file's results."""
    data = request.get_json()
    unique_filename = data.get('unique_filename')
    
    input_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
    
    results = get_separation_results(
        input_path,
        current_app.config['OUTPUT_FOLDER'],
        int(data.get('components')), # Convert string from JS to int
        data.get('model')
    )
    original_filename = unique_filename.split('_', 1)[-1]
    rendered_html = render_template('results_partial.html', results=results, filename=original_filename)
    return jsonify({'html': rendered_html})

@app.route('/download/<path:filepath>')
def download_file(filepath):
    return send_from_directory(current_app.config['OUTPUT_FOLDER'], filepath, as_attachment=True)

@app.route('/play/<path:filepath>')
def play_file(filepath):
    """Serves an audio file for in-browser playback."""
    return send_from_directory(current_app.config['OUTPUT_FOLDER'], filepath)

@app.route('/clear-files', methods=['POST'])
def clear_files():
    """Deletes all files from the upload and output directories."""
    upload_folder = current_app.config['UPLOAD_FOLDER']
    output_folder = current_app.config['OUTPUT_FOLDER']
    
    folders_to_clear = [upload_folder, output_folder]
    
    for folder in folders_to_clear:
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

    session.pop('last_file', None)
    flash("All temporary files and outputs have been cleared.", "success")
    return redirect(url_for('index'))

@app.route('/cancel', methods=['POST'])
def cancel_process():
    """Attempts to cancel a running Demucs process."""
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