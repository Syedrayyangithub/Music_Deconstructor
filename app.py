import os
from flask import Flask, request, render_template, send_from_directory, redirect, url_for, Response, session, current_app
from werkzeug.utils import secure_filename
import uuid
from pathlib import Path

# Import the core separation function from your script
from music_separator import separate_audio_ultra, get_separation_results

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
    silence_threshold_db = int(request.form.get('silence_threshold_db', 30)) # Default to 30 if not provided
    
    session['last_file'] = {
        "filename": unique_filename,
        "components": components,
        "model": model,
        "silence_threshold_db": silence_threshold_db # Store in session
    }

    # --- THIS IS THE FIX ---
    # 1. Get ALL context-bound variables *now*.
    results_url_for_later = url_for('results_page', _external=True)
    output_dir_for_later = current_app.config['OUTPUT_FOLDER']

    # 2. Create the generator function that *accepts* the variables.
    def generate_progress(results_url, output_dir):
        try:
            for progress_line in separate_audio_ultra(
                input_path,
                output_dir=output_dir,  # Use the passed-in variable
                model=model,
                components=components,
                enhance=enhance,
                silence_threshold_db=silence_threshold_db, # Pass the new parameter
                progress_callback=lambda line: f"data: {line}\n\n"
            ):
                if progress_line:
                    yield progress_line

            # Use the pre-generated URL string
            yield f"data: SEPARATION_COMPLETE::{results_url}\n\n"
            
        except Exception as e:
            yield f"data: ERROR::{str(e)}\n\n"

    # 3. Pass ALL variables into the generator when creating the Response.
    return Response(generate_progress(results_url_for_later, output_dir_for_later), mimetype='text/event-stream')

@app.route('/results')
def results_page():
    last_file_info = session.get('last_file')
    if not last_file_info:
        return redirect(url_for('index'))

    unique_filename = last_file_info['filename']
    # We MUST use current_app here because this route *has* context.
    input_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
    
    results = get_separation_results(
        input_path,
        current_app.config['OUTPUT_FOLDER'],
        last_file_info['components'],
        last_file_info['model']
    )
    original_filename = unique_filename.split('_', 1)[-1]
    return render_template('index.html', results=results, filename=original_filename)

@app.route('/download/<path:filepath>')
def download_file(filepath):
    return send_from_directory(current_app.config['OUTPUT_FOLDER'], filepath, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)