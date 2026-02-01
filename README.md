# Bengali News Shorts Automation 🎬

Automatically transform long-form Bengali news videos (2-10 minutes) into engaging YouTube Shorts (55 seconds) using AI.

## What It Does

1. **Upload** a Bengali news video
2. **Transcribe** using Whisper AI (converts speech to Bengali text)
3. **Analyze** with Gemini API (selects coherent segments)
4. **Cut & Stitch** using FFmpeg (creates vertical video)
5. **Download** your YouTube-ready Short (9:16 format, 45-60 seconds)

## Features

✅ Automatic Bengali speech-to-text transcription  
✅ AI-powered segment selection (picks most engaging parts)  
✅ Automatic vertical video formatting (9:16 aspect ratio)  
✅ Generates transcript showing selected segments  
✅ Web interface - no command line needed  

## How to Use

1. Open the web interface at `http://localhost:5000`
2. Upload a Bengali news video (MP4, up to 500MB)
3. Click "Create Short"
4. Wait 2-3 minutes for processing
5. Download your YouTube Short from the output folder

## Requirements

- Python 3.11+
- FFmpeg (included in Docker)
- Gemini API key (free from Google)
- Whisper model (auto-downloaded on first run)

## Installation & Setup

### Option 1: Docker (Easiest)
```bash
docker build -t bengali-shorts .
docker run -p 5000:5000 -e GEMINI_API_KEY=your_key_here bengali-shorts
```

### Option 2: Local Python
```bash
pip install -r requirements.txt
python app.py
```

Then visit: `http://localhost:5000`

## Environment Variables

```
GEMINI_API_KEY=your_google_gemini_api_key
```

Get a free API key from: https://ai.google.dev/

## Project Structure

```
bengali-news-shorts/
├── app.py                 # Flask web interface
├── pipeline.py            # Main processing pipeline
├── debug_step_by_step.py  # Debugging tool
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── input/                 # Upload videos here
├── output/                # Final shorts created here
├── transcripts/           # Generated transcripts saved here
├── temp/                  # Temporary processing files (auto-cleaned)
├── logs/                  # Processing logs
└── ffmpeg-master-latest-win64-gpl/  # FFmpeg binaries
```

## Technical Stack

- **Whisper (OpenAI)** - Bengali speech-to-text transcription
- **Gemini API (Google)** - AI segment selection & coherence validation
- **FFmpeg** - Video cutting, stitching, format conversion
- **Flask** - Web interface
- **Python 3.11** - Core programming language

## How It Works (Behind the Scenes)

1. **Transcription** → Whisper converts Bengali audio to text with timestamps
2. **Segment Extraction** → Extract 8-10 second segments from transcript
3. **Analysis** → Gemini AI analyzes which segments are most coherent
4. **Selection** → Picks 2-4 segments that form a logical story
5. **Video Cutting** → FFmpeg extracts selected time ranges from original video
6. **Stitching** → Combines multiple segments into one video
7. **Vertical Format** → Converts to 9:16 aspect ratio (YouTube Shorts standard)
8. **Output** → Creates final MP4 file + transcript file

## Processing Time

- Small video (2-3 min): ~2-3 minutes
- Medium video (5-7 min): ~5-7 minutes
- Large video (10 min): ~10-12 minutes

(First run takes extra time to download Whisper model)

## Troubleshooting

**"No valid segments found"**
- Video may not be clear Bengali speech
- Try a different video with clearer audio

**"API key error"**
- Make sure GEMINI_API_KEY is set correctly
- Get a free key from: https://ai.google.dev/

**Video won't upload**
- Check file is MP4 format
- Video size should be under 500MB
- Try a shorter video first

## Limitations

- Input video must be in Bengali language
- Audio should be clear and audible
- Recommended video length: 3-7 minutes
- Maximum file size: 500MB (can be changed)

## Future Improvements

- Support for other languages (Hindi, Urdu, etc.)
- Automatic caption generation
- Multiple output formats (Instagram Reels, TikTok)
- Batch processing (multiple videos at once)
- Custom segment length selection

## API Costs

- **Whisper**: Handled locally (no cost)
- **Gemini API**: First 15 requests/minute free, then minimal cost
- **FFmpeg**: Free and open source

For most use cases, this stays within Google's free tier.

## License

This project is for personal and educational use.

## Support

Check the debug logs in the `logs/` folder if something goes wrong.

---

**Made for Bengali content creators** 🇧🇩
