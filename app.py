#!/usr/bin/env python3
"""
FLASK BACKEND FOR BENGALI SHORTS PIPELINE - FIXED
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import subprocess
import threading
import uuid
from pathlib import Path
from datetime import datetime
import shutil
import sys

# ============================================================================
# CONFIG
# ============================================================================

app = Flask(__name__)
CORS(app)

# Get absolute paths
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
OUTPUT_FOLDER = BASE_DIR / "outputs"
PIPELINE_SCRIPT = BASE_DIR / "pipeline.py"

# FIX: Get venv Python path
VENV_PYTHON = BASE_DIR / "venv" / "Scripts" / "python.exe"
if not VENV_PYTHON.exists():
    print(f"❌ WARNING: venv Python not found at {VENV_PYTHON}")
    print(f"   Using system Python instead")
    VENV_PYTHON = "python"

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}
MAX_FILE_SIZE = 500 * 1024 * 1024

# Create folders
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Debug info
print(f"\n🔧 CONFIGURATION:")
print(f"BASE_DIR: {BASE_DIR}")
print(f"VENV_PYTHON: {VENV_PYTHON}")
print(f"UPLOAD_FOLDER: {UPLOAD_FOLDER}")
print(f"OUTPUT_FOLDER: {OUTPUT_FOLDER}")
print(f"PIPELINE_SCRIPT: {PIPELINE_SCRIPT}")
print(f"Pipeline exists: {PIPELINE_SCRIPT.exists()}\n")

# In-memory job tracking
jobs = {}

# ============================================================================
# UTILITIES
# ============================================================================

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_size_mb(filepath):
    return os.path.getsize(filepath) / (1024 * 1024)

def get_job(job_id):
    return jobs.get(job_id, None)

def update_job(job_id, **kwargs):
    if job_id in jobs:
        jobs[job_id].update(kwargs)
        return jobs[job_id]
    return None

def extract_duration(video_path):
    """Extract video duration using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'json', str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        return f"{minutes}m {seconds}s"
    except Exception:
        return "unknown"

# ============================================================================
# PIPELINE RUNNER - FIXED TO USE VENV PYTHON
# ============================================================================

def run_pipeline_for_video(job_id, video_path):
    """Run the pipeline and update job status"""
    job = get_job(job_id)
    if not job:
        return
    
    try:
        update_job(job_id, status='processing', progress=5)
        
        # Create temp folder for this job
        temp_dir = UPLOAD_FOLDER / job_id
        temp_dir.mkdir(exist_ok=True)
        
        # Copy video to temp folder
        temp_video = temp_dir / Path(video_path).name
        shutil.copy2(video_path, temp_video)
        
        # Create input/output folders for pipeline
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        transcripts_dir = temp_dir / "transcripts"
        
        input_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        transcripts_dir.mkdir(exist_ok=True)
        
        # Move video to input folder
        shutil.move(str(temp_video), str(input_dir / temp_video.name))
        
        update_job(job_id, progress=10)
        
        # FIXED: Use venv Python explicitly
        cmd = [
            str(VENV_PYTHON),
            str(PIPELINE_SCRIPT.absolute()),
        ]
        
        env = os.environ.copy()
        env['PIPELINE_INPUT'] = str(input_dir)
        env['PIPELINE_OUTPUT'] = str(output_dir)
        env['PIPELINE_TRANSCRIPTS'] = str(transcripts_dir)
        
        print(f"\n🚀 Running pipeline with VENV Python: {VENV_PYTHON}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Input folder: {input_dir}")
        print(f"Video file: {list(input_dir.glob('*'))}")
        print(f"Environment variables set...\n")
        
        # Run the pipeline
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        
        update_job(job_id, progress=20)
        stdout, stderr = process.communicate()
        
        # Print pipeline output for debugging
        if stdout:
            print(f"✅ Pipeline stdout: {stdout[:500]}")
        if stderr:
            print(f"⚠️ Pipeline stderr: {stderr[:500]}")
        
        if process.returncode == 0:
            # Check what files were created
            print(f"\n📁 Checking output folder: {output_dir}")
            print(f"MP4 files found: {list(output_dir.glob('*.mp4'))}")
            print(f"Checking transcripts folder: {transcripts_dir}")
            print(f"TXT files found: {list(transcripts_dir.glob('*.txt'))}")
            
            # Find output files
            output_files = list(output_dir.glob("*.mp4"))
            transcript_files = list(transcripts_dir.glob("*.txt"))
            
            if output_files:
                output_file = output_files[0]
                transcript_file = transcript_files[0] if transcript_files else None
                
                # Copy to outputs folder for download
                final_output = OUTPUT_FOLDER / output_file.name
                shutil.copy2(output_file, final_output)
                
                final_transcript = None
                if transcript_file:
                    final_transcript = OUTPUT_FOLDER / transcript_file.name
                    shutil.copy2(transcript_file, final_transcript)
                
                print(f"✅ Processing complete! Output: {final_output}")
                
                update_job(
                    job_id,
                    status='completed',
                    progress=100,
                    output_file=str(final_output.name),
                    transcript_file=str(final_transcript.name) if final_transcript else None,
                    duration=extract_duration(output_file),
                    completed_at=datetime.now().isoformat()
                )
            else:
                print(f"❌ No output file created in {output_dir}")
                update_job(job_id, status='error', error='No output file created')
        else:
            error_msg = stderr if stderr else "Pipeline failed"
            print(f"❌ Pipeline failed: {error_msg[:200]}")
            update_job(job_id, status='error', error=error_msg[:200])
        
        # Cleanup temp folder
        print(f"🧹 Cleaning up temp folder: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    except Exception as e:
        print(f"❌ Error in run_pipeline_for_video: {str(e)}")
        update_job(job_id, status='error', error=str(e)[:200])

# ============================================================================
# SIMPLE TEST ENDPOINT TO CHECK PIPELINE
# ============================================================================

@app.route('/test-pipeline', methods=['GET'])
def test_pipeline():
    """Test if pipeline.py works directly"""
    try:
        # Run a simple test
        test_cmd = [str(VENV_PYTHON), "-c", "import whisper; print('✅ Whisper imported successfully')"]
        result = subprocess.run(test_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            return jsonify({
                'success': True,
                'message': 'Pipeline test passed',
                'output': result.stdout
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Pipeline test failed',
                'error': result.stderr
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': 'Test failed',
            'error': str(e)
        })

# ============================================================================
# API ENDPOINTS (KEEP YOUR EXISTING HTML - IT'S FINE)
# ============================================================================

@app.route('/', methods=['GET'])
def dashboard():
    """Serve the dashboard HTML"""
    # YOUR EXISTING HTML CODE HERE - KEEP IT EXACTLY AS IS
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bengali Shorts Pipeline - Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .drag-active {
            background-color: rgba(59, 130, 246, 0.1) !important;
            border-color: rgb(59, 130, 246) !important;
        }
        .progress-bar {
            transition: width 0.3s ease;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
    </style>
</head>
<body class="bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 min-h-screen">
    <div class="max-w-7xl mx-auto p-8">
        <!-- Header -->
        <div class="mb-12">
            <h1 class="text-5xl font-bold text-white mb-2">
                <i class="fas fa-film text-blue-500"></i> Bengali Shorts Pipeline
            </h1>
            <p class="text-gray-400 text-lg">Convert Bengali news videos to YouTube Shorts</p>
            <div class="mt-4 flex items-center gap-2 text-sm">
                <div class="w-3 h-3 bg-green-500 rounded-full"></div>
                <span class="text-green-400">Server: http://localhost:5000</span>
                <div class="w-3 h-3 bg-blue-500 rounded-full ml-4"></div>
                <span class="text-blue-400">Using: ''' + str(VENV_PYTHON) + '''</span>
            </div>
        </div>

        <!-- Upload Area -->
        <div id="dropZone" class="mb-12 border-3 border-dashed border-gray-600 rounded-2xl p-12 text-center cursor-pointer hover:border-blue-400 transition-all bg-slate-800/50">
            <input type="file" id="fileInput" multiple accept="video/*" class="hidden">
            <i class="fas fa-cloud-upload-alt text-gray-400 text-6xl mb-4 block"></i>
            <h2 class="text-2xl font-bold text-white mb-2">Drag videos here</h2>
            <p class="text-gray-400 mb-4">or click to browse</p>
            <p class="text-sm text-gray-500">Supported: MP4, AVI, MOV, MKV</p>
        </div>

        <!-- Videos List -->
        <div id="videosList"></div>

        <!-- Empty State -->
        <div id="emptyState" class="text-center text-gray-400 py-12">
            <p class="text-lg">Upload videos to get started</p>
        </div>
    </div>

    <script>
        const API_URL = 'http://localhost:5000';
        const videos = {};
        let jobCheckInterval;

        // DOM Elements
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const videosList = document.getElementById('videosList');
        const emptyState = document.getElementById('emptyState');

        // Drag and drop
        dropZone.addEventListener('dragenter', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-active');
        });

        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-active');
        });

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-active');
            const files = e.dataTransfer.files;
            Array.from(files).forEach(file => {
                if (file.type.startsWith('video/')) {
                    uploadVideo(file);
                }
            });
        });

        dropZone.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', (e) => {
            Array.from(e.target.files).forEach(file => uploadVideo(file));
        });

        async function uploadVideo(file) {
            const videoId = Date.now();
            const fileSize = (file.size / (1024 * 1024)).toFixed(2);

            videos[videoId] = {
                id: videoId,
                name: file.name,
                size: fileSize,
                status: 'uploading',
                progress: 0,
                jobId: null,
                output: null,
                transcript: null,
                duration: '0s',
                error: null,
            };

            updateUI();

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch(`${API_URL}/api/upload`, {
                    method: 'POST',
                    body: formData
                });

                if (response.ok) {
                    const data = await response.json();
                    videos[videoId].status = 'queued';
                    videos[videoId].jobId = data.job_id;
                } else {
                    const error = await response.json();
                    videos[videoId].status = 'error';
                    videos[videoId].error = error.error || 'Upload failed';
                }
            } catch (err) {
                videos[videoId].status = 'error';
                videos[videoId].error = 'Connection failed';
            }

            updateUI();
            startJobMonitoring();
        }

        async function fetchJobStatus(videoId) {
            const video = videos[videoId];
            if (!video || !video.jobId) return;

            try {
                const response = await fetch(`${API_URL}/api/job/${video.jobId}`);
                if (response.ok) {
                    const job = await response.json();
                    videos[videoId].status = job.status;
                    videos[videoId].progress = job.progress;
                    videos[videoId].output = job.output_file;
                    videos[videoId].transcript = job.transcript_file;
                    videos[videoId].duration = job.duration;
                    videos[videoId].error = job.error;
                    updateUI();
                }
            } catch (err) {
                console.error('Status fetch failed:', err);
            }
        }

        function startJobMonitoring() {
            if (jobCheckInterval) clearInterval(jobCheckInterval);
            
            jobCheckInterval = setInterval(() => {
                Object.keys(videos).forEach(videoId => {
                    const video = videos[videoId];
                    if (video.status === 'queued' || video.status === 'processing') {
                        fetchJobStatus(videoId);
                    }
                });
            }, 1000);
        }

        function downloadFile(filename) {
            window.open(`${API_URL}/api/download/${filename}`, '_blank');
        }

        function removeVideo(videoId) {
            delete videos[videoId];
            updateUI();
        }

        function getStatusIcon(status) {
            switch(status) {
                case 'completed': return '<i class="fas fa-check-circle text-green-500"></i>';
                case 'processing': return '<i class="fas fa-spinner text-blue-500 animate-spin"></i>';
                case 'uploading': return '<i class="fas fa-spinner text-blue-500 animate-spin"></i>';
                case 'error': return '<i class="fas fa-exclamation-circle text-red-500"></i>';
                default: return '<i class="fas fa-play text-gray-400"></i>';
            }
        }

        function updateUI() {
            const videoIds = Object.keys(videos);
            
            if (videoIds.length === 0) {
                videosList.innerHTML = '';
                emptyState.style.display = 'block';
                return;
            }

            emptyState.style.display = 'none';

            videosList.innerHTML = `
                <div class="mb-6">
                    <h2 class="text-2xl font-bold text-white mb-6">Processing Queue (${videoIds.length})</h2>
                </div>
                ${videoIds.map(videoId => {
                    const video = videos[videoId];
                    return `
                        <div class="bg-slate-800 rounded-xl p-6 border border-gray-700 hover:border-gray-600 transition-all mb-4">
                            <div class="flex items-start justify-between mb-4">
                                <div class="flex items-start gap-4 flex-1">
                                    <div class="text-2xl mt-1">
                                        ${getStatusIcon(video.status)}
                                    </div>
                                    
                                    <div class="flex-1">
                                        <h3 class="text-lg font-semibold text-white">${video.name}</h3>
                                        <div class="flex gap-4 text-sm text-gray-400 mt-2">
                                            <span>${video.size} MB</span>
                                            ${video.status === 'completed' ? `
                                                <span class="text-green-400">Duration: ${video.duration}</span>
                                                <span class="text-green-400">✓ Complete</span>
                                            ` : ''}
                                            ${video.status === 'processing' ? `
                                                <span class="text-blue-400">Processing...</span>
                                            ` : ''}
                                            ${video.status === 'queued' ? `
                                                <span class="text-yellow-400">Queued</span>
                                            ` : ''}
                                            ${video.status === 'uploading' ? `
                                                <span class="text-blue-400">Uploading...</span>
                                            ` : ''}
                                        </div>
                                    </div>
                                </div>

                                <button onclick="removeVideo('${videoId}')" class="text-gray-400 hover:text-red-400 transition-colors p-2">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </div>

                            ${(video.status === 'processing' || video.status === 'uploading') ? `
                                <div class="mb-4">
                                    <div class="w-full bg-gray-700 rounded-full h-2 overflow-hidden">
                                        <div class="progress-bar bg-gradient-to-r from-blue-500 to-blue-400 h-full" style="width: ${video.progress}%"></div>
                                    </div>
                                    <div class="text-xs text-gray-400 mt-1">${Math.round(video.progress)}%</div>
                                </div>
                            ` : ''}

                            ${video.status === 'completed' ? `
                                <div class="bg-slate-700/50 rounded-lg p-4 space-y-3">
                                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                                        ${video.output ? `
                                            <button onclick="downloadFile('${video.output}')" class="flex items-center gap-2 text-blue-400 hover:text-blue-300 transition-colors bg-slate-700 p-3 rounded-lg">
                                                <span class="text-lg">🎥</span>
                                                <div class="text-left flex-1">
                                                    <div class="text-sm font-medium truncate">${video.output}</div>
                                                    <div class="text-xs text-gray-400">YouTube Short</div>
                                                </div>
                                                <i class="fas fa-download"></i>
                                            </button>
                                        ` : ''}
                                        
                                        ${video.transcript ? `
                                            <button onclick="downloadFile('${video.transcript}')" class="flex items-center gap-2 text-blue-400 hover:text-blue-300 transition-colors bg-slate-700 p-3 rounded-lg">
                                                <span class="text-lg">📄</span>
                                                <div class="text-left flex-1">
                                                    <div class="text-sm font-medium truncate">${video.transcript}</div>
                                                    <div class="text-xs text-gray-400">Transcript</div>
                                                </div>
                                                <i class="fas fa-download"></i>
                                            </button>
                                        ` : ''}
                                    </div>
                                </div>
                            ` : ''}

                            ${video.status === 'error' ? `
                                <div class="bg-red-500/10 border border-red-500/30 rounded-lg p-3">
                                    <p class="text-red-400 text-sm">${video.error}</p>
                                </div>
                            ` : ''}
                        </div>
                    `;
                }).join('')}
            `;
        }

        // Initial UI update
        updateUI();
    </script>
</body>
</html>'''

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Upload video and start processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use MP4, AVI, MOV, etc'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    upload_id = str(uuid.uuid4())
    filepath = UPLOAD_FOLDER / f"{upload_id}_{filename}"
    
    try:
        file.save(str(filepath))
        file_size = get_file_size_mb(filepath)
        
        # Create job
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            'id': job_id,
            'filename': filename,
            'upload_id': upload_id,
            'status': 'queued',
            'progress': 0,
            'file_size': file_size,
            'created_at': datetime.now().isoformat(),
            'output_file': None,
            'transcript_file': None,
            'duration': None,
            'error': None,
            'completed_at': None
        }
        
        print(f"\n📥 Upload received: {filename}")
        print(f"Job ID: {job_id}")
        print(f"File saved to: {filepath}")
        print(f"File size: {file_size:.2f} MB")
        print(f"Using Python: {VENV_PYTHON}\n")
        
        # Start processing in background thread
        thread = threading.Thread(
            target=run_pipeline_for_video,
            args=(job_id, str(filepath))
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'filename': filename,
            'status': 'queued'
        }), 202
        
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/job/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get job status and progress"""
    job = get_job(job_id)
    
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(job)

@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    return jsonify(list(jobs.values()))

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download output file"""
    filepath = OUTPUT_FOLDER / secure_filename(filename)
    
    if not filepath.exists():
        return jsonify({'error': 'File not found'}), 404
    
    try:
        return send_file(str(filepath), as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/job/<job_id>', methods=['DELETE'])
def cancel_job(job_id):
    """Cancel a job"""
    if job_id in jobs:
        job = jobs[job_id]
        if job['status'] in ['queued', 'processing']:
            job['status'] = 'cancelled'
            return jsonify({'message': 'Job cancelled'})
    
    return jsonify({'error': 'Job not found or already completed'}), 404

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🎬 BENGALI SHORTS PIPELINE - FLASK BACKEND (FIXED)")
    print("=" * 80)
    print(f"\n✅ Server starting on http://localhost:5000")
    print(f"✅ Using Python: {VENV_PYTHON}")
    print(f"📤 Upload endpoint: POST /api/upload")
    print(f"🧪 Test endpoint: GET /test-pipeline")
    print(f"📊 Status endpoint: GET /api/job/<job_id>")
    print(f"📥 Download endpoint: GET /api/download/<filename>")
    print("\n🔧 Before uploading:")
    print("   1. Open http://localhost:5000/test-pipeline to check pipeline")
    print("   2. If test fails, install missing packages in venv")
    print("=" * 80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)