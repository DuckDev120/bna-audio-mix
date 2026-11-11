import wave
import numpy as np

sr = 44100

def load_slice(path):
    with wave.open(path, 'rb') as w:
        w_sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
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

# Load Am4wYTiHHx8 (5.25s riser)
raw_riser = load_slice(r"c:\Users\Admin\Desktop\bna-audio-mix\cache\Am4wYTiHHx8.wav")

# Take the last 2.8 seconds
riser_dur = 2.8
riser_part = raw_riser[-int(riser_dur * sr):].copy()

# Exponential volume curve from -30dB to 0dB
t = np.linspace(0, 1, len(riser_part))
exp_gain = (10 ** ((t * 30 - 30) / 20))[:, None]
riser_part *= exp_gain

# Peak normalize to -0.5 dB
p = np.max(np.abs(riser_part))
if p > 0:
    riser_part = (riser_part / p) * 0.94

# Add 0.15s cinematic silence / vacuum breath at the very end
vacuum_silence = np.zeros((int(0.15 * sr), 2), dtype=np.float64)

# Create final riser file: riser + silence
final_riser = np.vstack([riser_part, vacuum_silence])

out_path = r"c:\Users\Admin\Desktop\bna-audio-mix\cache\studio_riser_cinematic.wav"
out_int16 = np.clip(final_riser * 32767.0, -32768, 32767).astype(np.int16)

with wave.open(out_path, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(out_int16.tobytes())

print(f"Created {out_path}: duration = {len(final_riser)/sr:.3f}s")
