#!/usr/bin/env python3
"""
DEBUG PIPELINE - Find exactly where it fails
Place this in your PROJECT ROOT folder (same as pipeline.py)
"""

import subprocess
import json
import sys
from pathlib import Path
import whisper
from datetime import datetime

print("🎬 BENGALI SHORTS DEBUGGER")
print("="*60)

# Setup paths - ADJUST THESE to match your setup
PROJECT_ROOT = Path(__file__).parent
print(f"Project root: {PROJECT_ROOT}")

FFMPEG = PROJECT_ROOT / "ffmpeg-master-latest-win64-gpl" / "bin" / "ffmpeg.exe"
FFPROBE = PROJECT_ROOT / "ffmpeg-master-latest-win64-gpl" / "bin" / "ffprobe.exe"

# Check if paths exist
print(f"\n🔧 Checking paths:")
print(f"   FFmpeg: {'✅' if FFMPEG.exists() else '❌'} {FFMPEG}")
print(f"   FFprobe: {'✅' if FFPROBE.exists() else '❌'} {FFPROBE}")

if not FFMPEG.exists() or not FFPROBE.exists():
    print("❌ FFmpeg/FFprobe not found!")
    sys.exit(1)

# Create directories
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
TEMP_DIR = PROJECT_ROOT / "temp"

print(f"\n📁 Checking folders:")
print(f"   Input: {'✅' if INPUT_DIR.exists() else '❌'} {INPUT_DIR}")
print(f"   Output: {'✅' if OUTPUT_DIR.exists() else '❌'} {OUTPUT_DIR}")

if not INPUT_DIR.exists():
    INPUT_DIR.mkdir()
    print("   Created input folder")

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir()
    print("   Created output folder")

if not TEMP_DIR.exists():
    TEMP_DIR.mkdir()
    print("   Created temp folder")

# Step 1: Find video
print(f"\n[1/7] Finding video in {INPUT_DIR}...")
videos = list(INPUT_DIR.glob("*.mp4")) + list(INPUT_DIR.glob("*.MP4"))
print(f"   Found: {len(videos)} video(s)")

for v in videos:
    size_mb = v.stat().st_size / (1024*1024)
    print(f"   • {v.name} ({size_mb:.1f} MB)")

if not videos:
    print("❌ No videos in input/ folder!")
    print("\n💡 Place your VID.mp4 in the 'input' folder")
    sys.exit(1)

video = videos[0]
print(f"✅ Using: {video.name}")

# Step 2: Get duration
print("\n[2/7] Getting video duration...")
cmd = [str(FFPROBE), '-v', 'error', '-show_entries', 'format=duration',
       '-of', 'json', str(video)]

result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode != 0:
    print(f"❌ FFprobe failed: {result.stderr}")
    sys.exit(1)

try:
    data = json.loads(result.stdout)
    duration = float(data['format']['duration'])
    print(f"✅ Duration: {duration:.1f}s ({duration/60:.1f} min)")
except:
    print(f"❌ JSON parse error: {result.stdout}")
    sys.exit(1)

if duration < 30:
    print(f"⚠️ Video is short ({duration:.1f}s), but continuing...")

# Step 3: Load Whisper
print("\n[3/7] Loading Whisper model...")
try:
    # Try 'base' first (faster), fall back to 'tiny' if needed
    print("   Loading 'base' model...")
    model = whisper.load_model("base")
    print("✅ Whisper 'base' loaded successfully")
except Exception as e:
    print(f"   'base' failed: {e}")
    print("   Trying 'tiny' model...")
    try:
        model = whisper.load_model("tiny")
        print("✅ Whisper 'tiny' loaded successfully")
    except Exception as e2:
        print(f"❌ Whisper load failed: {e2}")
        sys.exit(1)

# Step 4: Transcribe FIRST 60 SECONDS only (for testing)
print("\n[4/7] Transcribing FIRST 60 SECONDS...")
try:
    # Extract first 60 seconds of audio
    audio_path = TEMP_DIR / "test_audio.wav"
    print(f"   Extracting audio (0-60s)...")
    
    extract_cmd = [
        str(FFMPEG), '-y',
        '-i', str(video),
        '-t', '60',  # Only first 60 seconds for testing
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', '16000', '-ac', '1',
        '-af', 'volume=3.0',  # Boost volume
        str(audio_path)
    ]
    
    extract_result = subprocess.run(extract_cmd, capture_output=True, text=True)
    if extract_result.returncode != 0:
        print(f"   ⚠️ Audio extraction failed: {extract_result.stderr[:100]}")
        print("   Trying direct transcription from video...")
        audio_path = video
    
    # Transcribe
    print("   Transcribing...")
    result = model.transcribe(
        str(audio_path),
        language="bn",
        task="transcribe",
        fp16=False,
        temperature=0.0
    )
    
    # Clean up audio file
    if audio_path != video and audio_path.exists():
        audio_path.unlink()
    
    segments = result.get("segments", [])
    print(f"✅ Got {len(segments)} segments")
    
    if not segments:
        print("❌ No segments found!")
        print("\n💡 Possible reasons:")
        print("1. Audio is too quiet/noisy")
        print("2. Not Bengali speech")
        print("3. Whisper model issue")
        sys.exit(1)
    
    # Show segments
    print("\n   Segments found:")
    for i, seg in enumerate(segments[:5]):  # Show first 5
        text = seg['text'].strip()
        if text:
            print(f"   {i+1}. [{seg['start']:.1f}s-{seg['end']:.1f}s] {text}")
    
    # Check if segments have Bengali text
    print("\n   Checking for Bengali text...")
    bengali_segments = []
    for i, seg in enumerate(segments):
        text = seg['text'].strip()
        if text:
            bengali_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
            total_chars = len(text)
            if total_chars > 0:
                bengali_ratio = bengali_chars / total_chars
                if bengali_ratio > 0.3:  # At least 30% Bengali
                    bengali_segments.append((i, text, bengali_ratio))
    
    if bengali_segments:
        print(f"✅ Found {len(bengali_segments)} segments with Bengali text")
        for idx, text, ratio in bengali_segments[:3]:
            print(f"   Segment {idx+1}: {ratio:.0%} Bengali - '{text[:50]}...'")
    else:
        print("⚠️ No Bengali text detected!")
        print("   Showing what WAS detected:")
        for i, seg in enumerate(segments[:3]):
            text = seg['text'].strip()
            if text:
                print(f"   Segment {i+1}: '{text}'")
        
except Exception as e:
    print(f"❌ Transcription failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ BASIC TESTS PASSED!")
print("="*60)
print("\nYour setup is working. The issue might be in your main pipeline code.")
print("\n💡 Next steps:")
print("1. Run your main pipeline with:")
print('   python pipeline.py 2>&1 | tee debug_output.txt')
print("2. Share the debug_output.txt")
print("\nOr try this quick fix first:")
print('''
# Add this to the TOP of your pipeline.py main() function:
import logging
logging.basicConfig(level=logging.DEBUG)
''')
