/**
 * Main application class to manage the music deconstruction and fusion UI.
 * Encapsulates all state and DOM interactions.
 */
class MusicProcessorApp {
    
    // --- *** 1. SETUP *** ---

    /**
     * @constructor
     * Gathers all DOM elements and sets initial state.
     */
    constructor() {
        this.ALLOWED_TYPES = [
            'audio/mpeg', // mp3
            'audio/wav',  // wav
            'audio/flac', // flac
            'audio/x-m4a',// m4a
            'audio/mp4'    // m4a can also be mp4
        ];
        
        // --- State Variables ---
        this.fileQueue = [];
        this.isProcessing = false;
        this.activePlayers = [];
        this.processedSongs = {};
        this.fusedPlayer = null;
        this.abortController = null;

        // --- DOM Element Cache ---
        this.dom = {
            uploadForm: document.getElementById('upload-form'),
            fileInput: document.getElementById('file-input'),
            fileUploadArea: document.getElementById('file-upload-area'),
            fileUploadError: document.getElementById('file-upload-error'),
            filenameDisplay: document.getElementById('filename-display'),
            submitBtn: document.getElementById('submit-btn'),
            clearBtn: document.getElementById('clear-btn'),
            cancelBtn: document.getElementById('cancel-btn'),
            processingSection: document.getElementById('processing-section'),
            progressBar: document.getElementById('progressBar'),
            progressLog: document.getElementById('progress-log'),
            fileQueueList: document.getElementById('file-queue'),
            allResultsContainer: document.getElementById('all-results-container'),
            fusionPad: document.getElementById('fusion-pad'),
            fusionMixer: document.getElementById('fusion-mixer'), // Renamed from grid
            fuseBtn: document.getElementById('fuse-btn'),
            fuseProgressLog: document.getElementById('fuse-progress-log'),
            fusedResultContainer: document.getElementById('fused-result-container'),
            masterTempoSelect: document.getElementById('master-tempo-select')
        };
    }

    /**
     * Initializes all event listeners for the application.
     */
    init() {
        // Bind `this` context for all event handlers
        this.dom.fileUploadArea.addEventListener('click', () => this.dom.fileInput.click());
        this.dom.fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // Drag & Drop Listeners
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            this.dom.fileUploadArea.addEventListener(eventName, this.preventDefaults.bind(this), false);
        });
        ['dragenter', 'dragover'].forEach(eventName => {
            this.dom.fileUploadArea.addEventListener(eventName, this.highlightDropzone.bind(this), false);
        });
        ['dragleave', 'drop'].forEach(eventName => {
            this.dom.fileUploadArea.addEventListener(eventName, this.unhighlightDropzone.bind(this), false);
        });
        this.dom.fileUploadArea.addEventListener('drop', this.handleFileDrop.bind(this), false);
        
        // Form & Button Listeners
        this.dom.uploadForm.addEventListener('submit', this.handleFormSubmit.bind(this));
        this.dom.fuseBtn.addEventListener('click', this.handleFuseSubmit.bind(this));
        
        // Add event listener for Mute/Solo buttons (delegated)
        this.dom.fusionMixer.addEventListener('click', this.handleChannelButtons.bind(this));
    }

    // --- *** 2. FILE HANDLING & UI *** ---

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    highlightDropzone() {
        this.dom.fileUploadArea.classList.add('drag-over');
    }

    unhighlightDropzone() {
        this.dom.fileUploadArea.classList.remove('drag-over');
    }

    handleFileSelect(e) {
        this.addFilesToQueue([...e.target.files]);
        e.target.value = ''; // Reset file input
    }

    handleFileDrop(e) {
        this.addFilesToQueue([...e.dataTransfer.files]);
    }

    /**
     * Validates and adds files to the internal queue.
     * @param {File[]} files - An array of File objects.
     */
    addFilesToQueue(files) {
        this.dom.fileUploadError.style.display = 'none';
        this.dom.fileUploadError.textContent = '';
        
        const validFiles = files.filter(file => {
            if (this.ALLOWED_TYPES.includes(file.type)) {
                return true;
            }
            this.dom.fileUploadError.textContent = `Invalid file type: ${file.name}. Only audio files are allowed.`;
            this.dom.fileUploadError.style.display = 'block';
            return false;
        });

        this.fileQueue.push(...validFiles);
        this.updateQueueUI();
    }

    updateQueueUI() {
        this.dom.fileQueueList.innerHTML = '';
        this.fileQueue.forEach((file, index) => {
            const li = document.createElement('li');
            li.textContent = file.name;
            if (this.isProcessing && index === 0) li.className = 'processing';
            this.dom.fileQueueList.appendChild(li);
        });
        this.dom.filenameDisplay.textContent = `${this.fileQueue.length} file(s) in queue.`;
    }

    /**
     * Resets the UI to its initial or post-processing state.
     * @param {string} message - A final message to log.
     * @param {boolean} isError - Whether the message is an error.
     */
    resetUI(message, isError = false) {
        this.dom.submitBtn.disabled = false;
        this.dom.submitBtn.textContent = 'Deconstruct';
        this.dom.clearBtn.disabled = false;
        this.dom.cancelBtn.disabled = true;
        
        if (!isError) {
            this.dom.processingSection.classList.remove('is-visible');
        }

        this.dom.progressLog.textContent += `\n\n${message}`;
        this.dom.progressLog.scrollTop = this.dom.progressLog.scrollHeight;
        this.isProcessing = false;
        
        // Don't reset fusion pad if processing was successful
        if (isError) {
            this.dom.fusionPad.classList.remove('is-visible');
            if (this.fusedPlayer) {
                this.fusedPlayer.destroy();
                this.fusedPlayer = null;
            }
            this.dom.fusedResultContainer.classList.remove('is-visible');
            this.dom.fusedResultContainer.innerHTML = '';
        }
    }

    // --- *** 3. CORE PROCESSING (DECONSTRUCT) *** ---

    handleFormSubmit(event) {
        event.preventDefault();
        if (this.fileQueue.length === 0 || this.isProcessing) return;
        
        this.isProcessing = true;
        this.dom.submitBtn.disabled = true;
        this.dom.clearBtn.disabled = true;
        this.dom.cancelBtn.disabled = false;
        this.dom.submitBtn.textContent = 'Processing...';
        this.dom.allResultsContainer.innerHTML = '';

        // Reset state for a new batch
        this.processedSongs = {};
        this.dom.fusionPad.classList.remove('is-visible');
        if (this.fusedPlayer) {
            this.fusedPlayer.destroy();
            this.fusedPlayer = null;
        }
        this.dom.fusedResultContainer.classList.remove('is-visible');
        this.dom.fusedResultContainer.innerHTML = '';
        
        this.activePlayers.forEach(player => {
            if (player) player.destroy();
        });
        this.activePlayers = [];
        
        this.processQueue();
    }

    async processQueue() {
        if (this.fileQueue.length === 0) {
            this.resetUI("All files processed.", false);
            this.updateQueueUI();
            this.showFusionPad(); // Show the pad when queue is done
            return;
        }

        const currentFile = this.fileQueue[0];
        this.updateQueueUI();

        // Show processing section
        this.dom.processingSection.classList.add('is-visible');
        this.dom.progressLog.textContent = `Initializing for ${currentFile.name}...`;
        this.dom.progressBar.style.width = '0%';

        const formData = new FormData(this.dom.uploadForm);
        formData.set('file', currentFile); // Set the current file from queue

        this.abortController = new AbortController();
        this.dom.cancelBtn.onclick = () => {
            this.abortController.abort();
            // We also send a request to the backend to kill the process
            fetch('/cancel', { method: 'POST' });
        };

        let uniqueFilename = '';
        let originalFilename = '';

        try {
            const response = await fetch('/process', { 
                method: 'POST', 
                body: formData, 
                signal: this.abortController.signal 
            });
            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n\n');

                for (const line of lines) {
                    if (line.startsWith('data:')) {
                        const data = line.substring(5).trim();
                        if (data.startsWith('SEPARATION_COMPLETE::')) {
                            const parts = data.split('::');
                            uniqueFilename = parts[1];
                            originalFilename = parts[2];
                            break; // Exit inner loop
                        } else if (data.startsWith('ERROR::')) {
                            throw new Error(data.split('::')[1]);
                        } else {
                            this.dom.progressLog.textContent += `\n${data}`;
                            this.dom.progressLog.scrollTop = this.dom.progressLog.scrollHeight;
                            const match = data.match(/(\d+)%/);
                            if (match) this.dom.progressBar.style.width = `${match[1]}%`;
                        }
                    }
                }
                if (uniqueFilename) break; // Exit outer loop
            }

            const resultsResponse = await fetch('/results-for-file', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    unique_filename: uniqueFilename,
                    original_filename: originalFilename,
                    components: formData.get('components'),
                    model: formData.get('model')
                })
            });

            if (!resultsResponse.ok) {
                throw new Error(`Fetching results failed with status: ${resultsResponse.status}`);
            }

            const resultsData = await resultsResponse.json();
            
            // Create a "Song Card" wrapper
            const songCard = document.createElement('div');
            songCard.className = 'song-result-card animated-section';
            songCard.innerHTML = resultsData.html; // The partial now goes *inside* the card
            this.dom.allResultsContainer.appendChild(songCard);
            // Trigger the animation
            setTimeout(() => songCard.classList.add('is-visible'), 10); 
            
            // Store metadata for this processed song
            this.processedSongs[uniqueFilename] = {
                original_filename: originalFilename,
                model: formData.get('model'),
                components: parseInt(formData.get('components'))
            };
            
            this.initializeWaveforms();

        } catch (err) {
            if (err.name === 'AbortError') {
                this.resetUI("Processing cancelled by user.", false);
                this.fileQueue = []; // Clear queue on cancel
                this.updateQueueUI();
                return;
            } else {
                this.resetUI(`ERROR: ${err.message}`, true);
            }
        }

        // Successfully processed, remove from queue and continue
        this.fileQueue.shift();
        this.processQueue(); // Recursive call to process next item
    }

    // --- *** 4. FUSION PAD LOGIC *** ---
    
    /**
     * Dynamically builds and shows the Fusion Pad as a mixer.
     */
    showFusionPad() {
        if (Object.keys(this.processedSongs).length < 1) {
            return; // Don't show the pad if no songs were processed
        }

        // 1. Get component setting from the first processed song
        const firstSongId = Object.keys(this.processedSongs)[0];
        const components = this.processedSongs[firstSongId].components;

        // 2. Define stem lists based on component count
        let stems = [];
        if (components === 4) {
            stems = ["vocals", "drums", "bass", "other"];
        } else if (components === 6) {
            stems = ["vocals", "drums", "bass", "piano", "guitar", "other"];
        } else if (components === 8) {
            stems = [
                "vocals", "drums", "bass", "other",
                "lead_vocals", "harmony", "kick_snare", "cymbals", "piano", "guitar"
            ];
        }

        // 3. Clear and build the fusion grid
        this.dom.fusionMixer.innerHTML = '';
        
        stems.forEach(stem => {
            const prettyName = stem.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()); 

            const channelStrip = document.createElement('div');
            channelStrip.className = 'channel-strip';
            channelStrip.dataset.stem = stem; // "vocals", "kick_snare", etc.
            
            // Note: The volume fader value is 1.0 (full volume)
            channelStrip.innerHTML = `
                <h4>${prettyName}</h4>
                <select class="fusion-select" data-role="source-select">
                    </select>
                <div class="fader-container">
                    <label>Volume</label>
                    <input type="range" class="volume-fader" data-role="volume-fader" min="0" max="1.5" step="0.01" value="1.0">
                </div>
                <div class="channel-buttons">
                    <button class="channel-btn mute-btn" data-role="mute-btn">M</button>
                    <button class="channel-btn solo-btn" data-role="solo-btn">S</button>
                </div>
            `;
            this.dom.fusionMixer.appendChild(channelStrip);
        });

        // 4. Populate the dropdowns
        this.dom.fusionPad.classList.add('is-visible');
        this.dom.fuseProgressLog.style.display = 'none';
        this.dom.fuseProgressLog.textContent = '';
        
        this.dom.masterTempoSelect.innerHTML = '';
        const stemSelects = document.querySelectorAll('.fusion-select[data-role="source-select"]'); 
        stemSelects.forEach(sel => sel.innerHTML = '');

        const noneOption = document.createElement('option');
        noneOption.value = '';
        noneOption.textContent = 'None (Silence)';

        for (const [unique_id, songData] of Object.entries(this.processedSongs)) {
            // Add to master tempo select
            const tempoOption = document.createElement('option');
            tempoOption.value = unique_id;
            tempoOption.textContent = `Tempo: ${songData.original_filename}`;
            this.dom.masterTempoSelect.appendChild(tempoOption);

            // Add to each stem select
            stemSelects.forEach(select => {
                const stemOption = document.createElement('option');
                stemOption.value = unique_id;
                const channel = select.closest('.channel-strip');
                const labelText = channel.querySelector('h4').textContent;
                stemOption.textContent = `${labelText}: ${songData.original_filename}`;
                select.appendChild(stemOption.cloneNode(true));
            });
        }
        
        // Add "None" option
        stemSelects.forEach(sel => sel.appendChild(noneOption.cloneNode(true)));
    }

    /**
     * Handles Mute/Solo button logic
     */
    handleChannelButtons(event) {
        const target = event.target;
        if (!target.matches('[data-role="mute-btn"], [data-role="solo-btn"]')) return;

        target.classList.toggle('active');
        
        // Solo Logic: If any solo is active, mute all non-solo'd tracks
        const soloButtons = this.dom.fusionMixer.querySelectorAll('[data-role="solo-btn"]');
        const anySoloActive = [...soloButtons].some(btn => btn.classList.contains('active'));

        this.dom.fusionMixer.querySelectorAll('.channel-strip').forEach(strip => {
            const muteBtn = strip.querySelector('[data-role="mute-btn"]');
            const soloBtn = strip.querySelector('[data-role="solo-btn"]');
            
            if (anySoloActive) {
                if (soloBtn.classList.contains('active')) {
                    // This track is solo'd, unmute it
                    muteBtn.classList.remove('active');
                } else {
                    // Another track is solo'd, mute this one
                    muteBtn.classList.add('active');
                }
            } else {
                // No solo is active, return mute buttons to their explicit state
                // (do nothing, user's last mute click is correct)
            }
        });
    }

    /**
     * Gathers data from the new mixer UI to send to the backend.
     */
    async handleFuseSubmit(event) {
        event.preventDefault();
        this.dom.fuseBtn.disabled = true;
        this.dom.fuseBtn.textContent = 'Fusing...';
        this.dom.fuseProgressLog.style.display = 'block';
        this.dom.fuseProgressLog.textContent = 'Initializing fusion... (This may take a moment)';
        
        if (this.fusedPlayer) {
            this.fusedPlayer.destroy();
            this.fusedPlayer = null;
        }
        this.dom.fusedResultContainer.innerHTML = '';
        this.dom.fusedResultContainer.classList.remove('is-visible');

        try {
            // Build complex payload from mixer
            const fusion_map = {};
            const model_map = {};
            const components_map = {};
            
            document.querySelectorAll('.channel-strip').forEach(strip => {
                const stem = strip.dataset.stem;
                const sourceSelect = strip.querySelector('[data-role="source-select"]');
                const volumeFader = strip.querySelector('[data-role="volume-fader"]');
                const muteBtn = strip.querySelector('[data-role="mute-btn"]');

                fusion_map[stem] = {
                    song_id: sourceSelect.value, // e.g., "uuid_of_song_a"
                    volume: parseFloat(volumeFader.value), // e.g., 1.0
                    is_muted: muteBtn.classList.contains('active') // e.g., false
                };
            });

            for (const [id, data] of Object.entries(this.processedSongs)) {
                model_map[id] = data.model;
                components_map[id] = data.components;
            }

            const payload = {
                fusion_map: fusion_map,
                model_map: model_map,
                components_map: components_map,
                master_tempo_song_id: this.dom.masterTempoSelect.value
            };
            
            console.log("Sending complex payload to /fuse:", payload);

            const response = await fetch('/fuse', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || `Server error: ${response.status}`);
            }

            const result = await response.json();
            
            if (result.success) {
                this.dom.fuseProgressLog.textContent = 'Fusion complete! Loading player...';
                // Create a "Song Card" for the fused result
                const fusedHtml = `
                    <div class="song-result-card">
                        <h2>Your Fused Track</h2>
                        <ul>
                            <li>
                                <div class="stem-player" data-audio-path="/play/${result.path}">
                                    <button class="play-pause-btn" disabled>
                                        <svg class="play-icon" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                                        <svg class="pause-icon" style="display:none;" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="4" width="4" height="16"></rect><rect x="14" y="4" width="4" height="16"></rect></svg>
                                    </button>
                                    <div class="stem-details">
                                        <span class="stem-name">${result.filename}</span>
                                        <div class="waveform-container"></div>
                                    </div>
                                </div>
                                <a href="/download/${result.path}" class="download-btn">Download</a>
                            </li>
                        </ul>
                    </div>
                `;
                this.dom.fusedResultContainer.innerHTML = fusedHtml;
                this.dom.fusedResultContainer.classList.add('is-visible');
                
                const newPlayerEl = this.dom.fusedResultContainer.querySelector('.stem-player');
                this.initializeSingleWaveform(newPlayerEl, true); // 'true' marks it as the main fused player
                
                this.dom.fuseBtn.disabled = false;
                this.dom.fuseBtn.textContent = 'Fuse Tracks';
                this.dom.fuseProgressLog.style.display = 'none';
            } else {
                throw new Error(result.error || "Unknown error during fusion.");
            }

        } catch (err) {
            this.dom.fuseProgressLog.textContent = `ERROR: ${err.message}`;
            this.dom.fuseBtn.disabled = false;
            this.dom.fuseBtn.textContent = 'Fuse Tracks';
        }
    }


    // --- *** 5. WAVESURFER LOGIC *** ---

    /**
     * Finds all uninitialized players and calls initializeSingleWaveform.
     */
    initializeWaveforms() {
        // We query the document, not a specific container,
        // to initialize all new players at once.
        const players = document.querySelectorAll('.stem-player:not(.initialized)');
        players.forEach(player => {
            this.initializeSingleWaveform(player, false);
        });
    }

    /**
     * Initializes a single wavesurfer instance.
     * @param {HTMLElement} playerEl - The .stem-player element.
     * @param {boolean} isFusedPlayer - Whether this is the main fused player.
     */
    initializeSingleWaveform(playerEl, isFusedPlayer) {
        playerEl.classList.add('initialized');
        const audioPath = playerEl.dataset.audioPath;
        const waveformContainer = playerEl.querySelector('.waveform-container');
        const playBtn = playerEl.querySelector('.play-pause-btn');
        const playIcon = playBtn.querySelector('.play-icon');
        const pauseIcon = playBtn.querySelector('.pause-icon');

        if (!waveformContainer) {
            console.error("Waveform container not found for", audioPath);
            return;
        }

        const waveSurfer = WaveSurfer.create({
            container: waveformContainer,
            waveColor: '#555',
            progressColor: 'var(--primary-color)',
            cursorColor: 'var(--primary-color)',
            barWidth: 2, barRadius: 3, height: 40, responsive: true,
        });

        waveSurfer.load(audioPath);
        
        if (isFusedPlayer) {
            this.fusedPlayer = waveSurfer;
        } else {
            this.activePlayers.push(waveSurfer);
        }

        waveSurfer.on('ready', () => { playBtn.disabled = false; });
        waveSurfer.on('error', (e) => {
            console.error('Wavesurfer error:', e);
            if (e.name !== 'AbortError') {
                waveformContainer.innerHTML = `<span style="color: var(--error-color);">Error loading waveform</span>`;
            }
        });

        playBtn.addEventListener('click', () => {
            // Pause all *other* players
            const playersToPause = isFusedPlayer 
                ? this.activePlayers 
                : [this.fusedPlayer, ...this.activePlayers.filter(p => p !== waveSurfer)];
            
            playersToPause.forEach(p => {
                if (p && p.isPlaying()) p.pause();
            });
            waveSurfer.playPause();
        });

        waveSurfer.on('play', () => {
            playIcon.style.display = 'none';
            pauseIcon.style.display = 'block';
        });
        waveSurfer.on('pause', () => {
            playIcon.style.display = 'block';
            pauseIcon.style.display = 'none';
        });
        waveSurfer.on('finish', () => {
            waveSurfer.seekTo(0);
            playIcon.style.display = 'block';
            pauseIcon.style.display = 'none';
        });
    }
}

// --- *** 6. APP INITIALIZATION *** ---
document.addEventListener('DOMContentLoaded', () => {
    const app = new MusicProcessorApp();
    app.init();
});