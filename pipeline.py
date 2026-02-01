#!/usr/bin/env python3
"""
FINAL OPTIMIZED BENGALI YOUTUBE SHORTS PIPELINE
- Uses Whisper "small" (confirmed to work best for Bengali)
- FIXED: Segment selection now targets 45-60s shorts (not 27s)
- Gemini integration with improved prompt
- FIXED: Now uses environment variables from Flask for folder paths
- Copy & paste ready - just set API key and run
"""

import os
import sys
import subprocess
import json
import shutil
import whisper
import re
import requests
import locale
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import io

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Load .env file if exists
ENV_FILE = Path(__file__).parent / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().strip().split('\n'):
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")

# Set API key directly if not in .env
if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = "AIzaSyAr47FEJa_Y6T5Pwf-VfTT_o06mrrxT50M"

# Force UTF-8 encoding for all operations
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    pass

# Force UTF-8 encoding for Windows (skip in Colab)
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except:
    pass

# ============================================================================
# CONFIG - USE ENVIRONMENT VARIABLES FROM FLASK
# ============================================================================

PROJECT_ROOT = Path(__file__).parent

# Use environment variables if provided by Flask (otherwise use defaults)
input_folder = os.environ.get("PIPELINE_INPUT", PROJECT_ROOT / "input")
output_folder = os.environ.get("PIPELINE_OUTPUT", PROJECT_ROOT / "output")
temp_folder = os.environ.get("PIPELINE_TEMP", PROJECT_ROOT / "temp")
transcripts_folder = os.environ.get("PIPELINE_TRANSCRIPTS", PROJECT_ROOT / "transcripts")

FOLDERS = {
    "input": Path(input_folder),
    "output": Path(output_folder),
    "temp": Path(temp_folder),
    "transcripts": Path(transcripts_folder),
}

# FIXED: Changed from 55s to min/max range for flexibility
TARGET_DURATION_MIN = 45  # YouTube Shorts minimum
TARGET_DURATION_MAX = 60  # YouTube Shorts maximum
WHISPER_MODEL = "small"
ENCODE_PRESET = "veryfast"

print("\n" + "=" * 80)
print("BENGALI YOUTUBE SHORTS PIPELINE - FIXED SEGMENT SELECTION")
print("=" * 80)
print(f"\n🔧 PIPELINE CONFIGURATION:")
print(f"   Input folder: {FOLDERS['input']}")
print(f"   Output folder: {FOLDERS['output']}")
print(f"   Temp folder: {FOLDERS['temp']}")
print(f"   Transcripts folder: {FOLDERS['transcripts']}\n")

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Segment:
    start: float
    end: float
    text: str
    importance: float = 0.0
    
    @property
    def duration(self) -> float:
        return self.end - self.start

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_read_file(filepath):
    """Read a file with multiple encoding attempts"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'utf-16']
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    # If all encodings fail, return binary content
    with open(filepath, 'rb') as f:
        return f.read().decode('utf-8', errors='ignore')

def safe_write_file(filepath, content):
    """Write a file with UTF-8 encoding"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

# ============================================================================
# SINGLETON MODEL LOADER
# ============================================================================

# Global model cache (loads once, reuses forever)
_whisper_model = None

def get_whisper_model():
    """Load Whisper once and cache it"""
    global _whisper_model
    if _whisper_model is None:
        print("[1.2/6] Loading Whisper model (first run only)...")
        _whisper_model = whisper.load_model(WHISPER_MODEL)
        print("      ✅ Model cached in memory\n")
    else:
        print("[1.2/6] Using cached Whisper model\n")
    return _whisper_model

# ============================================================================
# AUDIO PREPROCESSING
# ============================================================================

def preprocess_audio_to_16khz_mono(video_path: Path, output_wav: Path) -> bool:
    """Convert video audio to 16kHz mono WAV for faster transcription"""
    # Try to find ffmpeg in PATH first
    ffmpeg_cmd = "ffmpeg"
    
    cmd = [
        ffmpeg_cmd, '-y', '-loglevel', 'error',
        '-i', str(video_path),
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        str(output_wav)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except:
        return False

# ============================================================================
# FFMPEG UTILITIES
# ============================================================================

def run_ffmpeg(cmd: List[str], timeout: int = 120) -> Tuple[bool, str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return result.returncode == 0, result.stderr if result.returncode != 0 else "OK"
    except Exception as e:
        return False, str(e)

def get_duration(video_path: Path) -> float:
    try:
        ffprobe_cmd = "ffprobe"
        cmd = [ffprobe_cmd, '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(video_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except:
        return 0

def cut_clip(video_path: Path, start: float, end: float, output_path: Path, video_duration: float) -> bool:
    duration = end - start
    if duration > 120 or duration < 1:
        return False
    
    ffmpeg_cmd = "ffmpeg"
    cmd = [
        ffmpeg_cmd, '-y', '-loglevel', 'error',
        '-ss', str(max(0, start - 0.5)), '-i', str(video_path),
        '-t', str(duration + 1), '-c:v', 'libx264', '-preset', ENCODE_PRESET,
        '-crf', '28', '-c:a', 'aac', '-b:a', '128k', str(output_path)
    ]
    success, _ = run_ffmpeg(cmd, timeout=120)
    return success

def stitch_clips(clip_paths: List[Path], output_path: Path) -> bool:
    if len(clip_paths) == 1:
        try:
            shutil.copy2(clip_paths[0], output_path)
            return True
        except:
            return False
    
    concat_file = FOLDERS["temp"] / f"concat_{os.getpid()}.txt"
    try:
        with open(concat_file, 'w', encoding='utf-8') as f:
            for clip in clip_paths:
                f.write(f"file '{clip.resolve()}'\n")
        
        ffmpeg_cmd = "ffmpeg"
        cmd = [ffmpeg_cmd, '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', str(concat_file), '-c', 'copy', str(output_path)]
        success, _ = run_ffmpeg(cmd, timeout=120)
        concat_file.unlink(missing_ok=True)
        return success
    except:
        return False

def make_vertical(input_path: Path, output_path: Path) -> bool:
    ffmpeg_cmd = "ffmpeg"
    cmd = [
        ffmpeg_cmd, '-y', '-loglevel', 'error', '-i', str(input_path),
        '-vf', 'scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black',
        '-c:a', 'copy', '-movflags', '+faststart', str(output_path)
    ]
    success, _ = run_ffmpeg(cmd, timeout=120)
    return success

# ============================================================================
# TRANSCRIPTION
# ============================================================================

def transcribe(video_path: Path) -> List[Segment]:
    print("[1.5/6] Preprocessing audio to 16kHz mono...")
    
    audio_path = FOLDERS["temp"] / f"audio_{os.getpid()}.wav"
    
    if preprocess_audio_to_16khz_mono(video_path, audio_path):
        print("      ✅ Audio preprocessed\n")
        transcribe_source = audio_path
    else:
        print("      ⚠️ Audio preprocessing skipped\n")
        transcribe_source = video_path
    
    print("[2/6] Transcribing with Whisper...")
    
    try:
        model = get_whisper_model()
        result = model.transcribe(str(transcribe_source), language="bn", fp16=False, verbose=False)
        segments = []
        
        for seg in result.get("segments", []):
            text = seg["text"].strip()
            if len(text) >= 3:
                s = Segment(start=seg["start"], end=seg["end"], text=text)
                segments.append(s)
    
    except Exception as e:
        print(f"      ❌ Transcription error: {e}")
        try:
            # Retry with fresh model
            model = whisper.load_model(WHISPER_MODEL)
            result = model.transcribe(str(transcribe_source), language="bn", fp16=False, verbose=False)
            segments = []
            for seg in result.get("segments", []):
                text = seg["text"].strip()
                if len(text) >= 3:
                    s = Segment(start=seg["start"], end=seg["end"], text=text)
                    segments.append(s)
        except:
            segments = []
    
    if audio_path.exists():
        try:
            audio_path.unlink()
        except:
            pass
    
    print(f"      ✅ {len(segments)} segments found\n")
    return segments

# ============================================================================
# GEMINI ANALYSIS - IMPROVED PROMPT
# ============================================================================

def analyze_with_gemini(segments: List[Segment], video_name: str) -> Tuple[bool, List[int]]:
    """Use Gemini to select segments targeting 45-60s YouTube Short"""
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        print("[3/6] Gemini analysis: ⚠️ No API key - using smart selection\n")
        return False, []
    
    if not segments:
        return False, []
    
    print("[3/6] Gemini analyzing segments...")
    
    try:
        # Build segment text with timing info
        segment_text = "\n".join([
            f"[{i+1}] ({s.start:.1f}s-{s.end:.1f}s, {s.duration:.0f}s) {s.text[:80]}"
            for i, s in enumerate(segments[:40])  # Show more segments for better selection
        ])
        
        # IMPROVED PROMPT - targets 45-60s, not arbitrary length
        prompt = f"""You are a YouTube Shorts expert. Analyze this Bengali news video and select segments for a YouTube Short.

VIDEO: {video_name}
TOTAL_SEGMENTS: {len(segments)}

SEGMENTS WITH TIMING:
{segment_text}

INSTRUCTIONS:
1. Select segments that discuss the SAME news topic (coherent narrative)
2. Total duration MUST be 45-60 seconds (YouTube Shorts standard)
3. Prefer segments with clear content (skip filler/ads)
4. If video is shorter than 45s, use ALL meaningful segments
5. Prioritize segments from different time points for variety
6. Never select overlapping segments

Return ONLY this JSON (no markdown):
{{"selected_indices": [0, 2, 4], "total_duration": 52, "topic": "Topic name", "reason": "Why these segments"}}"""
        
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,  # Lower = more deterministic
                    "maxOutputTokens": 500
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            try:
                data = response.json()
                text = data['candidates'][0]['content']['parts'][0]['text'].strip()
                text = re.sub(r'```json|```', '', text).strip()
                result = json.loads(text)
                indices = result.get("selected_indices", [])
                topic = result.get("topic", "Bengali News")
                reason = result.get("reason", "")
                
                if indices:
                    total = sum(segments[i].duration for i in indices if i < len(segments))
                    print(f"      ✅ Gemini selected {len(indices)} segments")
                    print(f"         Topic: {topic}")
                    print(f"         Duration: {total:.0f}s")
                    if reason:
                        print(f"         Reason: {reason}")
                    print()
                    return True, indices
            except (json.JSONDecodeError, KeyError, IndexError) as je:
                print(f"      ⚠️ Gemini response parse error - using smart fallback\n")
                return False, []
        else:
            print(f"      ⚠️ Gemini API error ({response.status_code}) - using smart fallback\n")
            return False, []
        
    except requests.Timeout:
        print(f"      ⚠️ Gemini timeout - using smart fallback\n")
        return False, []
    except Exception as e:
        print(f"      ⚠️ Gemini error: {str(e)[:60]} - using smart fallback\n")
        return False, []

# ============================================================================
# SMART SEGMENT SELECTION (when Gemini unavailable)
# ============================================================================

def smart_select_segments(segments: List[Segment]) -> List[int]:
    """
    Select segments to reach 45-60 seconds (YouTube Short duration)
    Strategy: Pick longer, consecutive segments for coherence (avoid choppiness)
    """
    if not segments:
        return []
    
    # Score each segment based on quality
    scored = []
    for i, seg in enumerate(segments):
        score = 0
        text_lower = seg.text.lower()
        
        # Skip obvious ads/CTAs
        if any(skip in text_lower for skip in ['subscribe', 'channel', 'like', 'bell', 'click']):
            score = -999  # Heavily penalize
        else:
            # Prefer longer segments (less choppy)
            if seg.duration >= 5:
                score += 3.0  # Strongly prefer 5+ second segments
            elif seg.duration >= 3:
                score += 2.0
            elif seg.duration >= 2:
                score += 1.0
            
            # Prefer segments with more content
            word_count = len(seg.text.split())
            if word_count >= 10:
                score += 2.0
            elif word_count >= 5:
                score += 1.0
            
            # Penalty for very short segments (causes choppiness)
            if seg.duration < 1.5:
                score -= 1.0
        
        scored.append((score, i, seg))
    
    # Sort by score (highest first)
    scored.sort(reverse=True, key=lambda x: x[0])
    
    # Strategy: Pick consecutive high-scoring segments (avoids choppiness)
    selected_indices = []
    total_duration = 0
    
    # Get all good segments (score >= 1.0)
    good_segments = [(idx, seg) for score, idx, seg in scored if score >= 1.0]
    
    # If we have enough good segments, pick consecutively from video
    if len(good_segments) >= 3:
        # Sort by original position (not score)
        good_segments.sort(key=lambda x: x[0])
        
        # Add segments consecutively until we hit target
        for idx, seg in good_segments:
            selected_indices.append(idx)
            total_duration += seg.duration
            
            if total_duration >= TARGET_DURATION_MIN:
                break
    else:
        # Fallback: just add all good segments
        selected_indices = [idx for idx, seg in good_segments]
        total_duration = sum(seg.duration for idx, seg in good_segments)
    
    # If still under minimum, add more (even shorter ones)
    if total_duration < TARGET_DURATION_MIN:
        for score, idx, seg in scored:
            if idx not in selected_indices and score >= 0:
                selected_indices.append(idx)
                total_duration += seg.duration
                if total_duration >= TARGET_DURATION_MIN:
                    break
    
    # Sort by original order (chronological - no jumping around)
    selected_indices.sort()
    
    total_duration = sum(segments[i].duration for i in selected_indices if i < len(segments))
    print(f"      💡 Smart selection: {len(selected_indices)} segments, {total_duration:.0f}s (longer, less choppy)\n")
    return selected_indices

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_video(video_path: Path) -> Optional[Path]:
    print(f"\n{'='*80}")
    print(f"Processing: {video_path.name}")
    print(f"{'='*80}\n")
    
    start_time = datetime.now()
    
    # Get duration
    duration = get_duration(video_path)
    if duration < 30:
        print("❌ Video too short (< 30s)\n")
        return None
    
    print(f"[0/6] Video duration: {duration/60:.1f} minutes\n")
    
    # Transcribe
    segments = transcribe(video_path)
    if not segments:
        print("❌ No segments transcribed\n")
        return None
    
    # Try Gemini first, fall back to smart selection
    success, indices = analyze_with_gemini(segments, video_path.stem)
    
    if not indices:
        print("[3/6] Using smart selection (45-60s target)...")
        indices = smart_select_segments(segments)
    
    # Get selected segments in chronological order
    selected = [segments[i] for i in indices if i < len(segments)]
    selected = sorted(selected, key=lambda x: x.start)
    
    if not selected:
        print("❌ No segments selected\n")
        return None
    
    total_duration = sum(s.duration for s in selected)
    
    # Check duration
    if total_duration < TARGET_DURATION_MIN:
        print(f"⚠️ Duration {total_duration:.0f}s < {TARGET_DURATION_MIN}s minimum")
        print(f"   Adding more segments...\n")
        
        # Add more segments to reach minimum
        remaining = [s for s in segments if s not in selected]
        for seg in remaining:
            selected.append(seg)
            total_duration += seg.duration
            if total_duration >= TARGET_DURATION_MIN:
                break
        
        selected = sorted(selected, key=lambda x: x.start)
    
    print(f"[4/6] Cutting {len(selected)} clips (total: {total_duration:.0f}s)...")
    
    # Cut clips
    clip_paths = []
    for i, seg in enumerate(selected):
        clip_path = FOLDERS["temp"] / f"clip_{i}.mp4"
        if cut_clip(video_path, seg.start, seg.end, clip_path, duration):
            clip_paths.append(clip_path)
        else:
            print(f"      ⚠️ Failed to cut clip {i+1}")
    
    if not clip_paths:
        print("❌ No clips created\n")
        return None
    
    print(f"      ✅ {len(clip_paths)} clips ready\n")
    
    # Stitch
    if len(clip_paths) > 1:
        print("[5/6] Stitching clips...")
        stitched = FOLDERS["temp"] / f"stitched_{os.getpid()}.mp4"
        if not stitch_clips(clip_paths, stitched):
            print("      ❌ Stitching failed\n")
            return None
        working_path = stitched
        print("      ✅ Stitched\n")
    else:
        working_path = clip_paths[0]
        print("[5/6] Single clip - no stitching needed\n")
    
    # Vertical
    print("[6/6] Converting to vertical (9:16)...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = FOLDERS["output"] / f"{video_path.stem}_{timestamp}_SHORT.mp4"
    
    if not make_vertical(working_path, output_path):
        print("      ❌ Vertical conversion failed\n")
        return None
    
    if not output_path.exists():
        print("      ❌ Output file not created\n")
        return None
    
    size_mb = output_path.stat().st_size / (1024*1024)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"      ✅ {size_mb:.1f} MB\n")
    
    # Transcript - Use safe_write_file
    transcript_path = FOLDERS["transcripts"] / f"{output_path.stem}_TRANSCRIPT.txt"
    transcript_content = f"SOURCE: {video_path.name}\n"
    transcript_content += f"DURATION: {total_duration:.0f}s\n"
    transcript_content += f"SEGMENTS: {len(selected)}\n"
    transcript_content += f"CREATED: {datetime.now()}\n\n"
    
    for i, seg in enumerate(selected, 1):
        transcript_content += f"[{i}] {seg.start:.1f}s-{seg.end:.1f}s ({seg.duration:.0f}s)\n{seg.text}\n\n"
    
    safe_write_file(transcript_path, transcript_content)
    
    # Cleanup
    for f in FOLDERS["temp"].glob("*"):
        try:
            f.unlink(missing_ok=True)
        except:
            pass
    
    print(f"✅ SUCCESS in {elapsed:.0f}s")
    print(f"   Output: {output_path.name}")
    print(f"   Duration: {total_duration:.0f}s")
    print(f"   Transcript: {transcript_path.name}\n")
    
    return output_path

# ============================================================================
# MAIN
# ============================================================================

def main():
    # Create all necessary folders
    print("\n📁 CREATING FOLDERS:")
    for name, folder in FOLDERS.items():
        try:
            folder.mkdir(parents=True, exist_ok=True)
            print(f"   {name}: {folder} {'(exists)' if folder.exists() else ''}")
        except Exception as e:
            print(f"   ❌ {name}: Failed to create {folder} - {e}")
    
    print()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        print(f"✅ Gemini API key loaded\n")
    else:
        print(f"⚠️ No Gemini API key (will use smart selection)\n")
    
    # Get unique videos from input folder
    video_files = {}
    input_folder = FOLDERS["input"]
    
    # Check if input folder exists and has videos
    if not input_folder.exists():
        print(f"❌ Input folder not found: {input_folder}")
        return
    
    for vid in list(input_folder.glob("*.mp4")) + list(input_folder.glob("*.MP4")):
        key = (vid.name, vid.stat().st_size)
        if key not in video_files:
            video_files[key] = vid
    
    videos = list(video_files.values())
    
    if not videos:
        print(f"❌ No videos in {input_folder}\n")
        return
    
    print(f"Found {len(videos)} unique video(s)\n")
    
    results = []
    for video in videos:
        try:
            output = process_video(video)
            if output:
                results.append(output)
        except Exception as e:
            print(f"❌ Error processing {video.name}: {e}\n")
    
    print("=" * 80)
    if results:
        print(f"✅ DONE! Created {len(results)} short(s)")
        for r in results:
            print(f"   📺 {r.name}")
        print(f"\n📁 Output folder: {FOLDERS['output']}")
        print(f"📄 Transcripts folder: {FOLDERS['transcripts']}\n")
    else:
        print("❌ No shorts created\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ Stopped\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
        import traceback
        traceback.print_exc()