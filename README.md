# AI Music Deconstructor

A powerful web application and command-line tool for separating music into its instrumental stems using state-of-the-art AI models from [Demucs](https://github.com/facebookresearch/demucs). This tool allows users to upload an audio file and deconstruct it into various components like vocals, drums, bass, and more.

The project features a user-friendly web interface built with Flask that provides real-time progress updates, along with a flexible Python script for backend processing.

## Features

- **Multiple Stem Separation Options:**
  - **4 Stems:** Vocals, Drums, Bass, Other
  - **6 Stems:** Vocals, Drums, Bass, Piano, Guitar, Other (via multi-pass separation)
  - **8 Stems:** Advanced separation into Lead Vocals, Harmony, Kick/Snare, Cymbals, and more.
- **Choice of AI Models:** Select from different Demucs models (`htdemucs`, `mdx_extra`, `htdemucs_ft`) to balance speed and quality.
- **Optional Audio Enhancement:** Automatically applies post-processing effects (normalization, compression, EQ) to enhance the clarity and quality of the separated stems.
- **Configurable Silence Trimming:** Intelligently removes leading/trailing silence from stems to reduce file size. The sensitivity of the trimming is adjustable.
- **Dynamic Web Interface:**
  - Modern, responsive UI with drag-and-drop file uploads.
  - Real-time progress bar and log powered by Server-Sent Events (SSE).
  - **Cancel Button:** Stop the separation process at any time.
  - **Clear Files Button:** Easily delete all uploaded and generated files to free up space.
- **Dual Usage:** Can be run as a web application or as a standalone command-line tool.

## Technology Stack

- **Backend:** Python, Flask
- **AI Model:** Demucs
- **Audio Processing:** Librosa, SoundFile, NumPy
- **Frontend:** HTML5, CSS3, JavaScript (Fetch API, SSE)

## Project Structure

```
music-separator/
├── app.py                  # Flask web server and API routes
├── music_separator.py      # Core logic for audio separation and enhancement (can be run standalone)
├── templates/
│   └── index.html          # Frontend HTML, CSS, and JavaScript
├── uploads/                # Directory for user-uploaded audio files
└── output_demucs/          # Directory for the separated output stems
```

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd music-separator
    ```

2.  **Create a Virtual Environment:**
    It is highly recommended to use a virtual environment.
    ```bash
    python -m venv venv_demucs
    source venv_demucs/bin/activate  # On Windows, use `venv_demucs\Scripts\activate`
    ```

3.  **Install Dependencies:**
    First, install PyTorch according to your system's specifications (CPU or GPU). You can find the correct command on the [PyTorch website](https://pytorch.org/get-started/locally/).

    Then, install Demucs and other required Python packages:
    ```bash
    pip install -U demucs
    pip install flask librosa soundfile numpy
    ```

## Usage

### 1. Web Application

To run the user-friendly web interface, execute the `app.py` script:

```bash
python app.py
```

Navigate to `http://127.0.0.1:5000` in your web browser.

1.  Drag and drop or select an audio file.
2.  Choose the desired number of stems, separation model, and enhancement options.
3.  Click "Deconstruct" to start the process.
4.  Monitor the progress in real-time. You can cancel the process at any point.
5.  Once complete, you will be redirected to a results page with download links for each stem.

### 2. Command-Line Interface (CLI)

You can also use `music_separator.py` directly from your terminal for more advanced or scripted use.

**Basic Usage (4 stems):**
```bash
python music_separator.py "path/to/your/song.mp3"
```

**Advanced Usage (6 stems, high-quality model, no enhancement):**
```bash
python music_separator.py "path/to/your/song.mp3" -c 6 -m mdx_extra --no-enhance
```

**Full List of CLI Options:**

| Argument              | Alias | Description                                                  | Default      |
| --------------------- | ----- | ------------------------------------------------------------ | ------------ |
| `input`               |       | Input audio file path.                                       | (Required)   |
| `--output`            | `-o`  | Output directory for separated files.                        | `output_demucs` |
| `--model`             | `-m`  | Demucs model to use (`htdemucs`, `htdemucs_ft`, `mdx_extra`). | `htdemucs`   |
| `--device`            | `-d`  | Device to use (`cpu`, `cuda`).                               | `cpu`        |
| `--components`        | `-c`  | Number of stems to separate (`4`, `6`, `8`).                 | `4`          |
| `--no-enhance`        |       | Disable the audio enhancement post-processing step.          | (flag)       |
| `--silence-threshold` |       | dB threshold for silence trimming (e.g., `20` for less, `40` for more). | `30`         |

---

This project provides a comprehensive and flexible solution for music source separation, suitable for both casual users via its web UI and advanced users through its command-line capabilities.