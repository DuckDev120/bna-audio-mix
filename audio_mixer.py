"""
AudioMixer - Automated, Reusable YouTube Audio Pipeline
Extracts, slices, normalizes, crossfades, and masters multi-track mixes.
"""

import os
import re
import json
import argparse
import subprocess
import wave
import numpy as np

class AudioMixer:
    def __init__(self, cache_dir="cache", output_dir="output", target_sr=44100):
        self.cache_dir = cache_dir
        self.output_dir = output_dir
        self.target_sr = target_sr
        self.tracks = []
        self.overlays = []
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def parse_time(t):
        """Converts MM:SS or SS.S to seconds float."""
        if isinstance(t, (int, float)):
            return float(t)
        t = str(t).strip()
        if ":" in t:
            parts = t.split(":")
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        return float(t)

    @staticmethod
    def extract_video_id(url):
        """Extracts 11-char YouTube ID from URL or returns the string if already an ID."""
        if os.path.exists(url):
            return os.path.splitext(os.path.basename(url))[0]
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:shorts\/)([0-9A-Za-z_-]{11})'
        ]
        for p in patterns:
            m = re.search(p, url)
            if m:
                return m.group(1)
        if len(url) == 11 and re.match(r'^[0-9A-Za-z_-]{11}$', url):
            return url
        return url

    def get_or_download_audio(self, url):
        """Checks cache or downloads audio using yt-dlp."""
        if os.path.exists(url):
            if url.lower().endswith(".wav"):
                return url
            base_name = os.path.splitext(os.path.basename(url))[0]
            clean_name = re.sub(r'[^0-9a-zA-Z_-]', '_', base_name)
            cached_wav = os.path.join(self.cache_dir, f"local_{clean_name}.wav")
            if not os.path.exists(cached_wav):
                print(f"[*] Converting local file '{os.path.basename(url)}' to WAV...")
                cmd = ["ffmpeg", "-y", "-i", url, "-ar", str(self.target_sr), "-ac", "2", cached_wav]
                subprocess.run(cmd, check=True, capture_output=True)
            return cached_wav
            
        video_id = self.extract_video_id(url)
        cached_wav = os.path.join(self.cache_dir, f"{video_id}.wav")
        if os.path.exists(cached_wav):
            return cached_wav

        print(f"[*] Downloading audio for '{video_id}' from YouTube...")
        cmd = [
            "python", "-m", "yt_dlp",
            "--no-check-certificates",
            "--extractor-args", "youtube:player_client=android",
            "-f", "ba/b",
            "-x", "--audio-format", "wav",
            "-o", os.path.join(self.cache_dir, f"{video_id}.%(ext)s"),
            f"https://www.youtube.com/watch?v={video_id}" if len(video_id)==11 else url
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(cached_wav):
            # Fallback to direct URL if needed
            cmd[-1] = url
            subprocess.run(cmd, check=True)
            
        if not os.path.exists(cached_wav):
            raise FileNotFoundError(f"Failed to download audio for {url}")
            
        return cached_wav

    def add_track(self, url_or_path, start, end, title="", repeat=1, crossfade=2.0, boost=1.0, direct_drop=False, fade_in=0.0, fade_out=0.0):
        """
        Adds a track segment to the mixer.
        - url_or_path: YouTube URL, video ID, or local file path
        - start, end: Time in seconds or 'MM:SS' string
        - title: Optional human-readable name
        - repeat: Number of times to play consecutively
        - crossfade: Crossfade overlap with next track (default 2.0s)
        - boost: Manual loudness multiplier (default 1.0)
        - direct_drop: If True, incoming track enters without being muffled by crossfade fade-in
        - fade_in: Fade in duration in seconds for this slice
        - fade_out: Fade out duration in seconds for this slice
        """
        self.tracks.append({
            "url": url_or_path,
            "start": self.parse_time(start),
            "end": self.parse_time(end),
            "title": title or url_or_path,
            "repeat": int(repeat),
            "crossfade": float(crossfade),
            "boost": float(boost),
            "direct_drop": bool(direct_drop),
            "fade_in": float(fade_in),
            "fade_out": float(fade_out)
        })

    def add_overlay(self, url_or_path, start_at=0.0, volume=1.0, title=""):
        """
        Adds an audio layer to be mixed on top of the master at a specific timestamp.
        - url_or_path: Audio file (PTT, voice note, sound effect)
        - start_at: Timeline position in seconds or 'MM:SS' string
        - volume: Multiplier for overlay loudness
        - title: Optional name
        """
        self.overlays.append({
            "url": url_or_path,
            "start_at": self.parse_time(start_at),
            "volume": float(volume),
            "title": title or url_or_path
        })

    def load_audio_slice(self, file_path, start_sec, end_sec):
        """Loads and converts slice to stereo float64 array normalized to [-1, 1]."""
        with wave.open(file_path, 'rb') as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            start_frame = max(0, int(round(start_sec * sr)))
            end_frame = min(w.getnframes(), int(round(end_sec * sr)))
            
            w.setpos(start_frame)
            raw = w.readframes(end_frame - start_frame)
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
            if ch == 2:
                data = data.reshape(-1, 2)
            elif ch == 1:
                data = np.column_stack([data, data])
            else:
                data = data.reshape(-1, ch)[:, :2]
                
            data /= 32768.0
            
            # Resample if sample rate doesn't match target
            if sr != self.target_sr:
                # Basic linear interpolation for safety
                target_len = int(len(data) * self.target_sr / sr)
                orig_indices = np.linspace(0, len(data) - 1, target_len)
                data = np.column_stack([
                    np.interp(orig_indices, np.arange(len(data)), data[:, 0]),
                    np.interp(orig_indices, np.arange(len(data)), data[:, 1])
                ])
                
            return data

    @staticmethod
    def equal_power_crossfade(seg1, seg2, overlap_samples):
        """Smooth equal-power crossfade using sine/cosine curves."""
        overlap_samples = min(overlap_samples, len(seg1), len(seg2))
        if overlap_samples <= 0:
            return np.vstack([seg1, seg2])
            
        t = np.linspace(0, np.pi / 2, overlap_samples)[:, None]
        fade_out = np.cos(t)
        fade_in = np.sin(t)
        
        pre = seg1[:-overlap_samples]
        mid = seg1[-overlap_samples:] * fade_out + seg2[:overlap_samples] * fade_in
        post = seg2[overlap_samples:]
        return np.vstack([pre, mid, post])

    @staticmethod
    def apply_fade_in_out(audio, sr, in_sec=1.5, out_sec=2.0):
        in_samples = int(round(in_sec * sr))
        out_samples = int(round(out_sec * sr))
        
        if in_samples > 0 and len(audio) > in_samples:
            t_in = np.linspace(0, np.pi / 2, in_samples)[:, None]
            audio[:in_samples] *= np.sin(t_in)
            
        if out_samples > 0 and len(audio) > out_samples:
            t_out = np.linspace(0, np.pi / 2, out_samples)[:, None]
            audio[-out_samples:] *= np.cos(t_out)
            
        return audio

    def build(self, output_name="master_audio", fade_in=1.5, fade_out=2.0, normalize_levels=True):
        """
        Builds the complete master track:
        - Downloads / loads all audio
        - Slices and repeats
        - Normalizes loudness
        - Assembles with transitions
        - Exports MP3 (320kbps) & WAV (44.1kHz 16-bit)
        """
        if not self.tracks:
            raise ValueError("No tracks added to mixer!")

        print(f"\n[+] Preparing {len(self.tracks)} track items...")
        sr = self.target_sr
        segments = []
        
        for item in self.tracks:
            wav_path = self.get_or_download_audio(item["url"])
            raw_slice = self.load_audio_slice(wav_path, item["start"], item["end"])
            
            # Apply boost
            raw_slice *= item["boost"]
            
            # Auto-balance loudness if requested
            if normalize_levels:
                rms = np.sqrt(np.mean(raw_slice**2))
                peak = np.max(np.abs(raw_slice))
                target_rms = 0.18 # -15 dB RMS standard dance master target
                gain = target_rms / max(rms, 1e-4)
                if peak * gain > 0.98:
                    gain = 0.98 / peak
                raw_slice *= gain
                
            # Apply slice-level fade in / fade out if requested
            if item.get("fade_in", 0.0) > 0:
                fi_len = int(round(item["fade_in"] * sr))
                fi_len = min(fi_len, len(raw_slice))
                raw_slice[:fi_len] *= np.sin(np.linspace(0, np.pi / 2, fi_len))[:, None]
                
            if item.get("fade_out", 0.0) > 0:
                fo_len = int(round(item["fade_out"] * sr))
                fo_len = min(fo_len, len(raw_slice))
                raw_slice[-fo_len:] *= np.cos(np.linspace(0, np.pi / 2, fo_len))[:, None]
                
            for r in range(item["repeat"]):
                title_suffix = f" (פעם {r+1})" if item["repeat"] > 1 else ""
                segments.append({
                    "data": raw_slice.copy(),
                    "title": item["title"] + title_suffix,
                    "crossfade": item["crossfade"],
                    "direct_drop": item["direct_drop"]
                })

        print(f"[+] Assembling {len(segments)} segments with seamless transitions...")
        current = segments[0]["data"]
        timeline = [{
            "track": segments[0]["title"],
            "start": 0.0,
            "duration": len(current) / sr
        }]
        
        current_time = 0.0
        for i in range(1, len(segments)):
            prev_item = segments[i - 1]
            next_item = segments[i]
            next_data = next_item["data"]
            
            overlap_sec = next_item["crossfade"]
            overlap_samples = int(round(overlap_sec * sr))
            
            if next_item["direct_drop"]:
                # Direct drop transition (e.g. into gunshots or chorus):
                # Fade out outgoing track over 1.6s, next track enters with full punch!
                fade_len = int(round(1.6 * sr))
                fade_len = min(fade_len, len(current))
                fo = np.cos(np.linspace(0, np.pi/2, fade_len))[:, None]
                current[-fade_len:] *= fo
                
                # Anti-click micro fade on incoming (40ms)
                micro = int(round(0.04 * sr))
                fi = np.sin(np.linspace(0, np.pi/2, micro))[:, None]
                next_mod = next_data.copy()
                next_mod[:micro] *= fi
                
                trans_overlap = int(round(0.6 * sr))
                pre = current[:-trans_overlap]
                mid = current[-trans_overlap:] + next_mod[:trans_overlap]
                post = next_mod[trans_overlap:]
                current = np.vstack([pre, mid, post])
                actual_overlap_sec = trans_overlap / sr
            else:
                current = self.equal_power_crossfade(current, next_data, overlap_samples)
                actual_overlap_sec = overlap_sec
                
            current_time += (len(prev_item["data"]) / sr) - actual_overlap_sec
            timeline.append({
                "track": next_item["title"],
                "start": current_time,
                "duration": len(next_data) / sr
            })

        # Apply global in/out fades
        current = self.apply_fade_in_out(current, sr, in_sec=fade_in, out_sec=fade_out)
        
        # Apply audio overlays (e.g. voice notes, sound effects)
        if self.overlays:
            print(f"[+] Applying {len(self.overlays)} audio overlay(s)...")
            for ov in self.overlays:
                ov_path = self.get_or_download_audio(ov["url"])
                ov_data = self.load_audio_slice(ov_path, 0, 999999) * ov["volume"]
                start_sample = int(round(ov["start_at"] * sr))
                end_sample = start_sample + len(ov_data)
                
                if end_sample > len(current):
                    pad = np.zeros((end_sample - len(current), 2))
                    current = np.vstack([current, pad])
                    
                current[start_sample:end_sample] += ov_data
                dur_ov = len(ov_data) / sr
                print(f"    Overlaid '{ov['title']}' at {ov['start_at']:.1f}s - {ov['start_at'] + dur_ov:.1f}s")
                timeline.append({
                    "track": f"[Overlay] {ov['title']}",
                    "start": ov["start_at"],
                    "duration": dur_ov
                })

        # True Peak limiter (-0.3 dBFS)
        peak = np.max(np.abs(current))
        if peak > 0.965:
            current *= (0.965 / peak)
            
        # Export files
        wav_file = os.path.join(self.output_dir, f"{output_name}.wav")
        mp3_file = os.path.join(self.output_dir, f"{output_name}.mp3")
        
        # Write WAV
        int16_data = np.clip(current * 32767.0, -32768.0, 32767.0).astype(np.int16)
        with wave.open(wav_file, 'wb') as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(int16_data.tobytes())
            
        # Write MP3 (320kbps CBR)
        cmd = [
            "ffmpeg", "-y", "-i", wav_file,
            "-codec:a", "libmp3lame",
            "-b:a", "320k",
            "-ar", "44100",
            mp3_file
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Copy to root as well for immediate user accessibility
        root_mp3 = f"{output_name}.mp3"
        root_wav = f"{output_name}.wav"
        subprocess.run(["ffmpeg", "-y", "-i", wav_file, "-codec:a", "libmp3lame", "-b:a", "320k", root_mp3], capture_output=True)
        with wave.open(root_wav, 'wb') as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(int16_data.tobytes())

        total_sec = len(current) / sr
        print(f"\n[v] Master created successfully!")
        print(f"    MP3: {mp3_file}")
        print(f"    WAV: {wav_file}")
        print(f"    Total Duration: {int(total_sec//60):02d}:{total_sec%60:05.2f} ({total_sec:.2f}s)")
        
        print("\n--- Master Timeline ---")
        for item in timeline:
            m = int(item['start'] // 60)
            s = item['start'] % 60
            print(f"  [{m:02d}:{s:05.2f}] {item['track']} ({item['duration']:.1f}s)")
            
        return {
            "wav": wav_file,
            "mp3": mp3_file,
            "duration": total_sec,
            "timeline": timeline
        }

def load_config_and_run(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
        
    mixer = AudioMixer(
        cache_dir=cfg.get("cache_dir", "cache"),
        output_dir=cfg.get("output_dir", "output")
    )
    for t in cfg.get("tracks", []):
        mixer.add_track(
            url_or_path=t["url"],
            start=t["start"],
            end=t["end"],
            title=t.get("title", ""),
            repeat=t.get("repeat", 1),
            crossfade=t.get("crossfade", 2.0),
            boost=t.get("boost", 1.0),
            direct_drop=t.get("direct_drop", False),
            fade_in=t.get("fade_in", 0.0),
            fade_out=t.get("fade_out", 0.0)
        )
    for ov in cfg.get("overlays", []):
        mixer.add_overlay(
            url_or_path=ov["url"],
            start_at=ov.get("start_at", 0.0),
            volume=ov.get("volume", 1.0),
            title=ov.get("title", "")
        )
    return mixer.build(
        output_name=cfg.get("output_name", "master_audio"),
        fade_in=cfg.get("fade_in", 1.5),
        fade_out=cfg.get("fade_out", 2.0),
        normalize_levels=cfg.get("normalize", True)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Reusable Audio Mixer")
    parser.add_argument("--config", "-c", default="tracks.json", help="Path to tracks JSON configuration")
    args = parser.parse_args()
    
    if os.path.exists(args.config):
        load_config_and_run(args.config)
    else:
        print(f"Config file '{args.config}' not found. Please create one or provide path.")

