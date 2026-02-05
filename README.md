# 🎬 Bengali YouTube Shorts Pipeline

Automated pipeline to convert long-form Bengali news videos (2-10 min) into YouTube Shorts (45-55s) using Faster-Whisper + Gemini AI.

**Performance**: ~2 minutes (GPU) | ~15-20 minutes (CPU)

---

## ✨ Features

- 🎯 **AI-Powered Selection** - Gemini API picks the best coherent 45-55s section
- 🗣️ **Bengali Transcription** - Faster-Whisper (small model) for accurate Bengali speech-to-text
- 📐 **Vertical Format** - Auto-converts to 9:16 (1080x1920) YouTube Shorts format
- 🌐 **Web UI** - Drag-and-drop upload with real-time progress tracking
- 🖥️ **CLI Support** - Direct pipeline.py processing for batch jobs
- 📝 **Auto Transcripts** - Generated .txt files showing selected segments

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/yourusername/bengali-shorts-pipeline.git
cd bengali-shorts-pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Add your Gemini API key (get free from https://makersuite.google.com/app/apikeys)
```

### 3. Run

**Web UI (Recommended):**
```bash
python app.py
# Visit http://localhost:5000
```

**Command Line:**
```bash
cp your_video.mp4 input/
python pipeline.py
# Results in output/
```

---

## 🏗️ How It Works

```
INPUT VIDEO (2-10 min)
    ↓
[Faster-Whisper] → Bengali speech-to-text with timestamps
    ↓
[Gemini API] → Analyzes segments, selects best 45-55s section
    ↓
[FFmpeg] → Cuts video to selected timeframe
    ↓
[FFmpeg] → Converts to 9:16 vertical format
    ↓
OUTPUT → YouTube Short (mp4) + Transcript (txt)
```

---

## 📊 Performance

| Stage | GPU (RTX 3080) | CPU (i7) |
|-------|---|---|
| Transcription | ~10s | 60-90s |
| Gemini Analysis | ~1-2s | ~1-2s |
| Video Cutting | ~3-5s | ~10-15s |
| Vertical Conversion | ~10-15s | ~30-60s |
| **TOTAL** | **~2 min** | **15-20 min** |

---

## 🔧 Configuration

```bash
# .env (required)
GEMINI_API_KEY=AIza...

# Optional
PIPELINE_INPUT=./input
PIPELINE_OUTPUT=./output
DEVICE=cuda              # or 'cpu'
COMPUTE_TYPE=float16     # or 'int8'
```

---

## 📁 Project Structure

```
bengali-shorts-pipeline/
├── pipeline.py           # Core processing engine
├── app.py                # Flask web server
├── requirements.txt      # Dependencies
├── .env                  # Secrets (don't commit!)
├── .env.example          # Template
├── .gitignore            # Git rules
├── uploads/              # Temp uploaded files
├── outputs/              # Final shorts
└── venv/                 # Python environment
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| **"No valid segments"** | Try video with clearer Bengali audio |
| **"API key error"** | Check GEMINI_API_KEY in .env |
| **"ffmpeg not found"** | Install: `sudo apt-get install ffmpeg` |
| **"GPU not detected"** | Install CUDA 11.8 for GPU acceleration |
| **"Video too short"** | Use videos ≥60 seconds |

---

## 📦 Requirements

```
Python 3.11+
faster-whisper>=0.9.1
google-generativeai>=0.3.0
flask>=2.3.0
flask-cors>=4.0.0
werkzeug>=2.3.0
requests>=2.31.0
```

---

## 🚀 Deployment

### Local
```bash
python app.py  # http://localhost:5000
```

### Railway.app
1. Connect GitHub repo
2. Set `GEMINI_API_KEY` env var
3. Deploy → Get public URL

---

## 📝 Web API

```
POST /api/upload           # Upload video
GET /api/job/{job_id}      # Check status
GET /api/download/{file}   # Download result
GET /api/jobs              # List all jobs
DELETE /api/job/{job_id}   # Cancel job
```

---

## 📄 License

MIT

---

## 🤝 Contributing

1. Fork repo
2. Create feature branch (`git checkout -b feature/name`)
3. Commit changes
4. Push & open PR

---

**Made for Bengali content creators** 🇧🇩

**Last Updated**: February 2026 | v1.0.0