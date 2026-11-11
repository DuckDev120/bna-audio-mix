import os
import wave
import numpy as np
import subprocess

sr = 44100

def load_wav(path, start=None, end=None):
    with wave.open(path, 'rb') as w:
        w_sr = w.getframerate()
        ch = w.getnchannels()
        s_frame = 0 if start is None else int(round(start * w_sr))
        e_frame = w.getnframes() if end is None else int(round(end * w_sr))
        w.setpos(s_frame)
        raw = w.readframes(e_frame - s_frame)
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
        if ch == 1:
            data = np.column_stack([data, data])
        elif ch == 2:
            data = data.reshape(-1, 2)
        if w_sr != sr:
            num = int(len(data) * sr / w_sr)
            idx = np.linspace(0, len(data) - 1, num)
            data = np.column_stack([np.interp(idx, np.arange(len(data)), data[:, 0]),
                                    np.interp(idx, np.arange(len(data)), data[:, 1])])
        return data

def equal_power_crossfade(seg1, seg2, overlap_samples):
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

print("[+] Loading source tracks...")
# 1. Shilo Elia: 00:00 to 01:01.85 (ends on 'כל הדינים')
t1 = load_wav(r"c:\Users\Admin\Desktop\bna-audio-mix\cache\43AZA8gYh00.wav", 0.0, 61.85)

# Add stereo reverb throw on the last 0.5s of t1
tail_source = t1[-int(0.5 * sr):]
reverb_len = int(1.2 * sr)
reverb = np.zeros((reverb_len, 2), dtype=np.float64)
decay = np.exp(-np.linspace(0, 4.5, reverb_len))[:, None]
for i, d_ms in enumerate([25, 45, 75, 110, 160, 220, 290]):
    d = int(d_ms * 0.001 * sr)
    g = (0.65 ** (i * 0.7)) * 0.25
    reverb[d:d+len(tail_source), 0] += tail_source[:, 0] * g
    reverb[d+15:d+15+len(tail_source), 1] += tail_source[:, 1] * g
reverb *= decay

# 2. Studio Riser FX (1.6s high energy build-up)
riser_raw = load_wav(r"c:\Users\Admin\Desktop\bna-audio-mix\cache\Am4wYTiHHx8.wav")
riser_len = int(1.6 * sr)
riser = riser_raw[-riser_len:].copy()
# Exponential gain curve from 0.2 to 1.0
t_r = np.linspace(0, 1, riser_len)
riser_gain = 0.2 + 0.8 * (t_r ** 2.2)
riser *= riser_gain[:, None]
riser = (riser / np.max(np.abs(riser))) * 0.85

# 3. Sub-Bass Boom (dz6Lp_PyX_Q.wav)
boom_raw = load_wav(r"c:\Users\Admin\Desktop\bna-audio-mix\cache\dz6Lp_PyX_Q.wav")
boom_start = int(0.415 * sr)
boom_len = int(3.2 * sr)
boom = boom_raw[boom_start : boom_start + boom_len].copy()
boom = (boom / np.max(np.abs(boom))) * 0.65 # massive low-end body

# 4. Rabenu HaKadosh: 36.90s to 70.0s (starts exactly on downbeat 1 with 'רבי נחמן')
t3 = load_wav(r"c:\Users\Admin\Desktop\bna-audio-mix\cache\63R54TBR_kE.wav", 36.90, 70.0)

# 5. Boee LaSeret: 46.0s to 76.0s (moved to end)
t2 = load_wav(r"c:\Users\Admin\Desktop\bna-audio-mix\cache\mbVhZ9OUFMY.wav", 46.0, 76.0)

print("[+] Assembling Studio Transition between Song 1 and Song 3...")
# t1 finishes at T_cut
L1 = len(t1)
# Riser starts 0.35s before t1 finishes (starts building under the tail of 'הדינים')
riser_start = L1 - int(0.35 * sr)
# Riser peaks at drop_sample
drop_sample = riser_start + len(riser)

# Build transition block
part1_len = drop_sample + len(t3)
part1 = np.zeros((part1_len, 2), dtype=np.float64)

# Place t1
part1[:L1] += t1
# Place vocal reverb tail starting at L1
rev_end = min(part1_len, L1 + reverb_len)
part1[L1:rev_end] += reverb[:rev_end - L1]

# Place riser
part1[riser_start : riser_start + len(riser)] += riser

# Place t3 and Boom right on drop_sample (ZERO gap!)
part1[drop_sample : drop_sample + len(t3)] += t3
boom_end = min(part1_len, drop_sample + len(boom))
part1[drop_sample : boom_end] += boom[:boom_end - drop_sample]

print("[+] Crossfading into Track 2 (Boee LaSeret)...")
# Crossfade part1 with t2 over 2.0s
overlap_samples = int(2.0 * sr)
master = equal_power_crossfade(part1, t2, overlap_samples)

# Apply master fade-in (1.5s) and fade-out (2.0s)
fi_samples = int(1.5 * sr)
master[:fi_samples] *= np.sin(np.linspace(0, np.pi/2, fi_samples))[:, None]

fo_samples = int(2.0 * sr)
master[-fo_samples:] *= np.cos(np.linspace(0, np.pi/2, fo_samples))[:, None]

# True Peak mastering limiter (-0.3 dBFS / 0.965)
peak = np.max(np.abs(master))
print(f"Master pre-limit peak: {peak:.3f}")
if peak > 0.965:
    master *= (0.965 / peak)

output_dir = r"c:\Users\Admin\Desktop\bna-audio-mix\output"
os.makedirs(output_dir, exist_ok=True)
wav_out = os.path.join(output_dir, "master_audio_6.wav")
mp3_out = os.path.join(output_dir, "master_audio_6.mp3")
root_mp3 = r"c:\Users\Admin\Desktop\bna-audio-mix\master_audio_6.mp3"

int16_master = np.clip(master * 32767.0, -32768, 32767).astype(np.int16)
with wave.open(wav_out, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes(int16_master.tobytes())

print(f"[v] Saved master WAV: {wav_out}")

# Convert to MP3 320kbps
cmd = ["ffmpeg", "-y", "-i", wav_out, "-codec:a", "libmp3lame", "-b:a", "320k", "-ar", "44100", mp3_out]
subprocess.run(cmd, check=True, capture_output=True)
print(f"[v] Exported master MP3: {mp3_out}")

# Copy to root
with open(mp3_out, 'rb') as src_f, open(root_mp3, 'wb') as dst_f:
    dst_f.write(src_f.read())
print(f"[v] Updated root MP3: {root_mp3}")

total_dur = len(master) / sr
m = int(total_dur // 60)
s = total_dur % 60
print(f"\n[+] Master Complete! Total Duration: {m:02d}:{s:05.2f} ({total_dur:.2f}s)")
print(f"    - Song 1: 00:00.00 - {L1/sr:05.2f}s (61.85s)")
print(f"    - Riser: {(riser_start/sr):05.2f}s - {(drop_sample/sr):05.2f}s (1.6s, peaks at drop)")
print(f"    - DROP (Rabenu HaKadosh + Bass Boom): {(drop_sample/sr):05.2f}s (ZERO GAP!)")
print(f"    - Song 3 duration: {len(t3)/sr:.2f}s")
print(f"    - Transition to Song 2: {((drop_sample + len(t3) - overlap_samples)/sr):05.2f}s")
