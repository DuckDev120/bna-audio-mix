import os, sys, json, re, ssl, urllib.request, subprocess
from concurrent.futures import ThreadPoolExecutor

ctx = ssl._create_unverified_context()
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

base_output_dir = r"C:\Users\Admin\Desktop\bna-audio-mix\plylist"
os.makedirs(base_output_dir, exist_ok=True)

playlists = [
    {
        "id": "6RECDAgUSI1NMXsAPFB4re",
        "default_title": "פלייליסט יום הולדת רגוע"
    },
    {
        "id": "2NUXjUmp7G6XY5YacyaLej",
        "default_title": "יומולדת שירים הרקדה"
    }
]

def clean_filename(name):
    clean = re.sub(r'[\\/*?:"<>|]', '_', name).strip()
    return re.sub(r'\s+', ' ', clean)

def fetch_playlist_tracks(playlist_id, default_title):
    url = f"https://open.spotify.com/embed/playlist/{playlist_id}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, context=ctx) as resp:
        html = resp.read().decode('utf-8')
    m = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html)
    if not m:
        raise ValueError(f"Could not parse Spotify embed for playlist {playlist_id}")
    data = json.loads(m.group(1))
    entity = data['props']['pageProps']['state']['data']['entity']
    title = clean_filename(entity.get('title', default_title))
    track_list = entity.get('trackList', [])
    return title, track_list

def download_track(item):
    idx, title, artist, folder = item
    safe_title = clean_filename(title)
    safe_artist = clean_filename(artist)
    filename = f"{idx:02d} - {safe_title} - {safe_artist}.mp3"
    target_path = os.path.join(folder, filename)

    if os.path.exists(target_path) and os.path.getsize(target_path) > 100000:
        print(f"[SKIP] Already exists: {filename}")
        return True

    query = f"{title} {artist} audio"
    out_template = os.path.join(folder, f"{idx:02d} - {safe_title} - {safe_artist}.%(ext)s")
    
    cmd = [
        "python", "-m", "yt_dlp",
        "--no-check-certificates",
        "--extractor-args", "youtube:player_client=android",
        "--default-search", "ytsearch",
        "-f", "ba/b",
        "-x", "--audio-format", "mp3",
        "--audio-quality", "320k",
        "--embed-metadata",
        "-o", out_template,
        f"ytsearch1:{query}"
    ]
    
    print(f"[*] Downloading [{idx:02d}]: {title} - {artist}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"[V] Finished [{idx:02d}]: {filename}")
        return True
    else:
        # Fallback query without 'audio'
        cmd[-1] = f"ytsearch1:{title} {artist}"
        res2 = subprocess.run(cmd, capture_output=True, text=True)
        if res2.returncode == 0:
            print(f"[V] Finished (fallback) [{idx:02d}]: {filename}")
            return True
        else:
            print(f"[X] Failed [{idx:02d}]: {filename} -> {res.stderr[:200]}")
            return False

def main():
    for p in playlists:
        pid = p["id"]
        def_title = p["default_title"]
        print(f"\n=======================================================")
        print(f"Fetching Spotify playlist: {pid} ({def_title})...")
        title, tracks = fetch_playlist_tracks(pid, def_title)
        folder = os.path.join(base_output_dir, title)
        os.makedirs(folder, exist_ok=True)
        print(f"Target folder: {folder}")
        print(f"Total tracks to download: {len(tracks)}")
        print(f"=======================================================")

        tasks = []
        for i, t in enumerate(tracks, 1):
            tasks.append((i, t.get('title', ''), t.get('subtitle', ''), folder))

        # Download with 3 concurrent workers to respect bandwidth & rate limits
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(download_track, tasks))

        success_count = sum(1 for r in results if r)
        print(f"\n[DONE] Playlist '{title}': {success_count}/{len(tracks)} tracks downloaded successfully!")

if __name__ == "__main__":
    main()

