#!/usr/bin/env python3
"""
FINAL PRODUCTION BENGALI YOUTUBE SHORTS PIPELINE
- FASTER-WHISPER: 3x faster transcription
- SINGLE CONTINUOUS SECTION: Better quality than stitching
- GEMINI ENHANCED: Smart 45-55s section selection
- Optimized for web hosting (Railway/Render/AWS)
"""

import os
import sys
import subprocess
import json
import shutil
import re
import requests
import locale
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import io

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

ENV_FILE = Path(__file__).parent / ".env"
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().strip().split('\n'):
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ[key.strip()] = value.strip().strip('"').strip("'")

if "GEMINI_API_KEY" not in os.environ:
    print("⚠️ GEMINI_API_KEY not found! Will use simple selection method")

try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    pass

try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except:
    pass

# ============================================================================
# CONFIG
# ============================================================================

PROJECT_ROOT = Path(__file__).parent

input_folder = os.environ.get("PIPELINE_INPUT", PROJECT_ROOT / "input")
output_folder = os.environ.get("PIPELINE_OUTPUT", PROJECT_ROOT / "output")
temp_folder = os.environ.get("PIPELINE_TEMP", PROJECT_ROOT / "temp")

FOLDERS = {
    "input": Path(input_folder),
    "output": Path(output_folder),
    "temp": Path(temp_folder),
}

TARGET_DURATION_MIN = 45
TARGET_DURATION_MAX = 55  # Shorter for better YouTube Shorts retention
ENCODE_PRESET = "veryfast"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

print("\n" + "=" * 80)
print("BENGALI YOUTUBE SHORTS PIPELINE - SINGLE SECTION MODE")
print("=" * 80)
print(f"\n🔧 CONFIGURATION:")
print(f"   Target: {TARGET_DURATION_MIN}-{TARGET_DURATION_MAX}s single continuous section")
print(f"   Input folder: {FOLDERS['input']}")
print(f"   Output folder: {FOLDERS['output']}")
print(f"   Temp folder: {FOLDERS['temp']}")
if GEMINI_API_KEY:
    print(f"   ✅ Gemini API: Available")
else:
    print(f"   ⚠️ Gemini API: Not available (using simple selection)\n")

# ============================================================================
# GLOBAL SESSION STATE
# ============================================================================

_SESSION_STATE = {
    "faster_whisper_model": None,
    "model_loaded": False,
}

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Segment:
    start: float
    end: float
    text: str
    
    @property
    def duration(self) -> float:
        return self.end - self.start

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_write_file(filepath, content):
    """Write a file with UTF-8 encoding"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

# ============================================================================
# FASTER-WHISPER MODEL LOADER
# ============================================================================

def get_faster_whisper_model():
    """Load Faster-Whisper once per session"""
    global _SESSION_STATE
    
    if _SESSION_STATE["faster_whisper_model"] is not None:
        return _SESSION_STATE["faster_whisper_model"]
    
    print("[1.2/6] Loading Faster-Whisper...")
    
    try:
        from faster_whisper import WhisperModel
        
        # Determine device
        device = "cuda" if sys.platform != "win32" and not os.environ.get("RAILWAY_ENVIRONMENT") else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        model = WhisperModel(
            "small",
            device=device,
            compute_type=compute_type,
            num_workers=2,
            download_root="/tmp/whisper_cache"
        )
        
        _SESSION_STATE["faster_whisper_model"] = model
        _SESSION_STATE["model_loaded"] = True
        
        print("      ✅ Faster-Whisper ready\n")
        return model
        
    except ImportError:
        print("      ❌ faster-whisper not installed!")
        print("      Run: pip install faster-whisper\n")
        raise
    except Exception as e:
        print(f"      ❌ Failed to load Faster-Whisper: {e}\n")
        raise

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

def cut_clip(video_path: Path, start: float, end: float, output_path: Path) -> bool:
    """Cut a single continuous section"""
    duration = end - start
    if duration > 120 or duration < 1:
        return False
    
    ffmpeg_cmd = "ffmpeg"
    
    cmd = [
        ffmpeg_cmd, '-y', '-loglevel', 'error',
        '-ss', str(start), '-i', str(video_path),
        '-t', str(duration),
        '-c:v', 'libx264', '-preset', ENCODE_PRESET, '-crf', '23',
        '-c:a', 'aac', '-b:a', '192k',
        '-avoid_negative_ts', 'make_zero',
        str(output_path)
    ]
    
    success, error = run_ffmpeg(cmd, timeout=120)
    if not success:
        print(f"      FFmpeg error: {error[:100]}")
    return success

def make_vertical(input_path: Path, output_path: Path) -> bool:
    """Convert to 9:16 vertical format"""
    ffmpeg_cmd = "ffmpeg"
    cmd = [
        ffmpeg_cmd, '-y', '-loglevel', 'error', '-i', str(input_path),
        '-vf', 'scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black',
        '-c:v', 'libx264', '-preset', ENCODE_PRESET, '-crf', '23',
        '-c:a', 'aac', '-b:a', '192k',
        '-movflags', '+faststart',
        str(output_path)
    ]
    success, _ = run_ffmpeg(cmd, timeout=120)
    return success

# ============================================================================
# TRANSCRIPTION WITH FASTER-WHISPER
# ============================================================================

def transcribe_fast(video_path: Path) -> List[Segment]:
    """
    Transcribe using Faster-Whisper with VAD filtering
    """
    print("[2/6] Transcribing with Faster-Whisper...")
    
    try:
        model = get_faster_whisper_model()
        
        # Transcribe with VAD filtering for better segment detection
        segments_info, info = model.transcribe(
            str(video_path),
            language="bn",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=1000,
                speech_pad_ms=500,
                threshold=0.5
            )
        )
        
        segments = []
        for seg in segments_info:
            text = seg.text.strip()
            if len(text) >= 3 and seg.end - seg.start >= 2:  # Minimum 2 seconds
                segments.append(Segment(
                    start=max(0, seg.start - 0.3),  # Start a bit earlier
                    end=seg.end + 0.3,              # End a bit later
                    text=text
                ))
        
        print(f"      ✅ Found {len(segments)} speech segments\n")
        return segments
        
    except Exception as e:
        print(f"      ❌ Transcription error: {e}\n")
        return []

# ============================================================================
# SIMPLE SEGMENT SELECTION - FIND BEST SINGLE CONTINUOUS SECTION
# ============================================================================

def find_best_single_section(segments: List[Segment], video_duration: float) -> Tuple[float, float]:
    """
    Find the best single continuous section of 45-55 seconds.
    Returns (start_time, end_time)
    """
    if not segments:
        return (0, 0)
    
    print("[3/6] Finding best single continuous section...")
    
    # Sort by start time
    segments = sorted(segments, key=lambda x: x.start)
    
    best_start = 0
    best_end = 0
    best_score = -1
    
    # Look for good sections in the video
    for i in range(len(segments)):
        current_start = segments[i].start
        current_end = current_start
        
        # Try to build a section starting from this segment
        for j in range(i, len(segments)):
            segment_end = segments[j].end
            
            # Check if adding this segment keeps us under max duration
            if segment_end - current_start <= TARGET_DURATION_MAX:
                current_end = segment_end
                
                # Calculate score for this section
                section_duration = current_end - current_start
                
                if TARGET_DURATION_MIN <= section_duration <= TARGET_DURATION_MAX:
                    # Good duration range - score it
                    score = section_duration
                    
                    # Bonus for being in middle of video (not beginning/end)
                    mid_point = (current_start + current_end) / 2
                    if video_duration * 0.2 <= mid_point <= video_duration * 0.8:
                        score += 10
                    
                    # Bonus for including more segments (coherence)
                    num_segments_included = j - i + 1
                    score += num_segments_included * 5
                    
                    if score > best_score:
                        best_score = score
                        best_start = current_start
                        best_end = current_end
    
    # If no perfect section found, find the best available
    if best_score == -1:
        # Try to find the longest coherent section
        for i in range(len(segments)):
            for j in range(i, len(segments)):
                section_start = segments[i].start
                section_end = segments[j].end
                section_duration = section_end - section_start
                
                if section_duration >= 30:  # At least 30 seconds
                    if section_duration > best_score:
                        best_score = section_duration
                        best_start = section_start
                        best_end = section_end
    
    # Ensure we don't exceed video duration
    best_end = min(best_end, video_duration)
    best_start = max(0, best_start)
    
    duration = best_end - best_start
    
    if duration >= TARGET_DURATION_MIN:
        print(f"      ✅ Found {duration:.0f}s section: {best_start:.0f}s-{best_end:.0f}s\n")
        return (best_start, best_end)
    else:
        print(f"      ⚠️ Best section only {duration:.0f}s (target: {TARGET_DURATION_MIN}s)\n")
        # Try to extend the section
        extension_needed = TARGET_DURATION_MIN - duration
        if extension_needed > 0:
            # Extend end time
            best_end = min(best_end + extension_needed, video_duration)
            duration = best_end - best_start
            
            if duration >= TARGET_DURATION_MIN:
                print(f"      ✅ Extended to {duration:.0f}s\n")
                return (best_start, best_end)
    
    return (0, 0)

# ============================================================================
# GEMINI ENHANCED SELECTION
# ============================================================================

def analyze_with_gemini(segments: List[Segment], video_name: str, video_duration: float) -> Tuple[bool, float, float]:
    """Use Gemini to find the best single continuous section"""
    if not GEMINI_API_KEY or not segments:
        return False, 0, 0
    
    print("[3/6] Using Gemini to find coherent section...")
    
    try:
        # Group segments into chunks for analysis
        chunk_size = 15  # seconds
        chunks = []
        current_chunk = []
        current_chunk_end = 0
        
        for seg in segments:
            if not current_chunk or seg.start - current_chunk_end <= 5:
                current_chunk.append(seg)
                current_chunk_end = seg.end
            else:
                if current_chunk:
                    chunk_text = " ".join([s.text for s in current_chunk])
                    chunk_start = current_chunk[0].start
                    chunk_end = current_chunk[-1].end
                    chunks.append((chunk_start, chunk_end, chunk_text))
                current_chunk = [seg]
                current_chunk_end = seg.end
        
        if current_chunk:
            chunk_text = " ".join([s.text for s in current_chunk])
            chunk_start = current_chunk[0].start
            chunk_end = current_chunk[-1].end
            chunks.append((chunk_start, chunk_end, chunk_text))
        
        # Prepare prompt for Gemini
        chunk_descriptions = []
        for i, (start, end, text) in enumerate(chunks[:15]):  # Limit to first 15 chunks
            duration = end - start
            if duration >= 10:  # Only show chunks of at least 10 seconds
                chunk_descriptions.append(f"[{i}] {start:.0f}s-{end:.0f}s ({duration:.0f}s): {text[:100]}...")
        
        prompt = f"""Select the SINGLE best 45-55 second section from this Bengali news video for a YouTube Short.

VIDEO: {video_name}
TOTAL DURATION: {video_duration:.0f} seconds

AVAILABLE SECTIONS:
{chr(10).join(chunk_descriptions)}

CRITERIA:
1. Choose ONE continuous section (don't combine multiple sections)
2. Duration MUST be 45-55 seconds
3. Should be coherent (complete thought/paragraph)
4. Avoid beginnings/ends with filler content
5. Prefer sections with clear news content
6. Section must be within the video duration

Return ONLY the JSON:
{{"start_seconds": 120, "end_seconds": 175, "reason": "This section discusses..."}}"""
        
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 200
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            text = data['candidates'][0]['content']['parts'][0]['text'].strip()
            text = re.sub(r'```json|```', '', text).strip()
            result = json.loads(text)
            
            start_time = result.get("start_seconds", 0)
            end_time = result.get("end_seconds", 0)
            reason = result.get("reason", "")
            
            duration = end_time - start_time
            
            # Validate Gemini's selection
            if (TARGET_DURATION_MIN <= duration <= TARGET_DURATION_MAX and 
                start_time >= 0 and end_time <= video_duration):
                print(f"      ✅ Gemini selected: {start_time:.0f}s-{end_time:.0f}s ({duration:.0f}s)")
                print(f"      Reason: {reason[:80]}...\n")
                return True, start_time, end_time
        
        print("      ⚠️ Gemini didn't find suitable section, using simple method\n")
        return False, 0, 0
        
    except Exception as e:
        print(f"      ❌ Gemini error: {str(e)[:100]}\n")
        return False, 0, 0

# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_video(video_path: Path) -> Optional[Path]:
    print(f"\n{'='*80}")
    print(f"Processing: {video_path.name}")
    print(f"{'='*80}\n")
    
    start_time = datetime.now()
    
    # Step 1: Get video duration
    duration = get_duration(video_path)
    if duration < 60:
        print("❌ Video too short (< 60s)\n")
        return None
    
    print(f"[1/6] Video duration: {duration/60:.1f} minutes\n")
    
    # Step 2: Transcribe
    segments = transcribe_fast(video_path)
    if not segments:
        print("❌ No speech segments found\n")
        return None
    
    # Step 3: Try Gemini first, then fallback to simple method
    gemini_success, section_start, section_end = analyze_with_gemini(segments, video_path.stem, duration)
    
    if not gemini_success:
        # Use simple method
        section_start, section_end = find_best_single_section(segments, duration)
    
    # Validate section
    if section_end - section_start < 30:
        print("❌ No suitable section found (minimum 30s)\n")
        return None
    
    # Ensure section is within bounds
    section_start = max(0, section_start)
    section_end = min(duration, section_end)
    section_duration = section_end - section_start
    
    print(f"[4/6] Selected section: {section_start:.0f}s - {section_end:.0f}s ({section_duration:.0f}s)\n")
    
    # Step 4: Cut the single section
    print("[5/6] Cutting section...")
    
    # Create temp output
    temp_output = FOLDERS["temp"] / f"cut_{os.getpid()}.mp4"
    
    success = cut_clip(video_path, section_start, section_end, temp_output)
    
    if not success or not temp_output.exists():
        print("❌ Failed to cut section\n")
        return None
    
    print(f"      ✅ Cut successful\n")
    
    # Step 5: Convert to vertical
    print("[6/6] Converting to vertical (9:16)...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_output = FOLDERS["output"] / f"{video_path.stem}_{timestamp}_SHORT.mp4"
    
    # First check if temp file has audio
    check_audio_cmd = ["ffprobe", "-i", str(temp_output), "-show_streams", "-select_streams", "a", "-loglevel", "error"]
    has_audio = subprocess.run(check_audio_cmd, capture_output=True).returncode == 0
    
    if not has_audio:
        print("      ⚠️ No audio detected in cut section, using original audio...")
        # Extract audio from original video for the same section
        audio_temp = FOLDERS["temp"] / f"audio_{os.getpid()}.aac"
        audio_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(section_start), "-i", str(video_path),
            "-t", str(section_duration),
            "-vn", "-c:a", "aac", "-b:a", "192k",
            str(audio_temp)
        ]
        subprocess.run(audio_cmd, capture_output=True)
        
        # Merge video with extracted audio
        merge_cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(temp_output),
            "-i", str(audio_temp),
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0",
            "-map", "1:a:0",
            str(FOLDERS["temp"] / f"with_audio_{os.getpid()}.mp4")
        ]
        subprocess.run(merge_cmd, capture_output=True)
        temp_output = FOLDERS["temp"] / f"with_audio_{os.getpid()}.mp4"
    
    # Now convert to vertical
    if not make_vertical(temp_output, final_output):
        print("      ❌ Vertical conversion failed\n")
        return None
    
    if not final_output.exists():
        print("      ❌ Output file not created\n")
        return None
    
    # Verify final output
    final_duration = get_duration(final_output)
    size_mb = final_output.stat().st_size / (1024*1024)
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"      ✅ {size_mb:.1f} MB, {final_duration:.1f}s duration\n")
    
    # Cleanup
    for f in FOLDERS["temp"].glob("*"):
        try:
            f.unlink(missing_ok=True)
        except:
            pass
    
    # Create transcript
    try:
        transcript_path = final_output.with_suffix('.txt')
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(f"Video: {video_path.name}\n")
            f.write(f"Section: {section_start:.1f}s - {section_end:.1f}s ({section_duration:.1f}s)\n")
            f.write(f"Created: {datetime.now()}\n\n")
            
            # Find segments in this section
            for seg in segments:
                if section_start <= seg.start <= section_end:
                    f.write(f"[{seg.start:.1f}s-{seg.end:.1f}s] {seg.text}\n")
        
        print(f"📝 Transcript saved\n")
    except:
        pass
    
    print(f"✅ SUCCESS in {elapsed:.0f}s")
    print(f"   Output: {final_output.name}")
    print(f"   Duration: {final_duration:.1f}s\n")
    
    return final_output

# ============================================================
# MAIN
# ============================================================

def main():
    print("\n📁 CREATING FOLDERS:")
    for name, folder in FOLDERS.items():
        try:
            folder.mkdir(parents=True, exist_ok=True)
            print(f"   {name}: {folder}")
        except Exception as e:
            print(f"   ❌ {name}: Failed - {e}")
    
    print()
    
    if GEMINI_API_KEY:
        print(f"✅ Gemini API key loaded\n")
    else:
        print(f"⚠️ Using simple selection method only\n")
    
    # Get videos
    input_folder = FOLDERS["input"]
    if not input_folder.exists():
        print(f"❌ Input folder not found: {input_folder}")
        return
    
    # Find all video files
    videos = []
    for ext in ['.mp4', '.MP4', '.mov', '.MOV', '.mkv', '.avi']:
        videos.extend(list(input_folder.glob(f"*{ext}")))
    
    # Remove duplicates by name
    unique_videos = {}
    for video in videos:
        key = (video.name.lower(), video.stat().st_size)
        if key not in unique_videos:
            unique_videos[key] = video
    
    videos = list(unique_videos.values())
    
    if not videos:
        print(f"❌ No videos found in {input_folder}")
        return
    
    print(f"Found {len(videos)} video(s)\n")
    
    # Process all videos
    results = []
    for video in videos:
        try:
            output = process_video(video)
            if output:
                results.append(output)
                
                # Check if output is good
                duration = get_duration(output)
                if duration < TARGET_DURATION_MIN:
                    print(f"⚠️ Warning: Output only {duration:.0f}s (less than target {TARGET_DURATION_MIN}s)")
        except Exception as e:
            print(f"❌ Error processing {video.name}: {e}\n")
    
    print("\n" + "=" * 80)
    if results:
        print(f"✅ DONE! Created {len(results)} short(s)")
        for r in results:
            duration = get_duration(r)
            print(f"   📺 {r.name} ({duration:.0f}s)")
        print(f"\n📁 Output folder: {FOLDERS['output']}\n")
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