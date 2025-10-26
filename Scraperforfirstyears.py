# py testscrapforfirstyears.py  
# C:\Users\loowi\Downloads\4chat
# Fetches up to COMMENTS_PER_YEAR English comments per year (2010–2024) from sampled popular videos,
# excluding music videos, rotating through multiple API keys, with safety and fallback.
# Ensures a maximum of MAX_COMMENTS_PER_VIDEO comments per video and outputs valid NDJSON.

import json
import time
import random
from datetime import datetime, timezone, timedelta
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langdetect import detect, DetectorFactory

# Ensure deterministic language detection
DetectorFactory.seed = 0

# Configuration constants
API_KEYS = [
  # Where the API keys were placed - Removed according to Google's rules.
]
MAX_ROTATIONS = len(API_KEYS)
COMMENTS_PER_YEAR = 10000
VIDEO_SAMPLE_SIZE = 4000
MAX_COMMENTS_PER_VIDEO = 40  # limit per video
MIN_VIDEOS_REQUIRED = 2500     # minimum videos per year to warn
MIN_COMMENTS_REQUIRED = 10000  # minimum comments per year to warn
YEARS = list(range(2010, 2012))
MAX_SEARCH_PAGES = 5000
BACKOFF_INITIAL = 0.5
BACKOFF_MAX = 8
MIN_CANDIDATES = 2000        # fallback threshold per tier
MAX_TIER_IDS = 20000         # cap on total search candidates per tier
COMMENT_PAGE_SIZE = 100      # comments per API call

# Query terms for video sampling
QUERY_TERMS = ['news', 'report', 'interview']  # additional queries to widen sampling



# Rotation state
current_key_index = 0
rotation_attempts = 0
used_api_keys = set()

# Storage for metadata
VIDEO_TITLES = {}
VIDEO_PUBDATES = {}
VIDEO_SOURCES = {}  # map video_id -> (term, order)

# Initialize YouTube client
def get_youtube_client():
    api_key = API_KEYS[current_key_index]
    return build('youtube', 'v3', developerKey=api_key, cache_discovery=False)

youtube = get_youtube_client()

# Rotate API key on quota errors
def rotate_key():
    global current_key_index, youtube, rotation_attempts
    rotation_attempts += 1
    if rotation_attempts >= MAX_ROTATIONS:
        raise RuntimeError('All API keys exhausted.')
    current_key_index = (current_key_index + 1) % len(API_KEYS)
    youtube = get_youtube_client()
    print(f"🔁 Rotated to key index {current_key_index}", flush=True)
    return youtube

# Safe wrappers for API calls
def safe_search(**kwargs):
    """Safe wrapper for search.list: rotates key on any 403, retries on 5xx."""
    retries = 0
    while True:
        try:
            return youtube.search().list(**kwargs).execute()
        except HttpError as e:
            status = getattr(e.resp, 'status', None)
            # Rotate on any 403 (including forbidden/suspended)
            if status == 403:
                print(f"🔁 Rotating key due to 403 (status forbidden/suspended)", flush=True)
                rotate_key()
                continue
            # Retry on server errors
            if status in (500, 503) and retries < 3:
                retries += 1
                time.sleep(min(BACKOFF_INITIAL * 2**(retries-1), BACKOFF_MAX))
                continue
            # Otherwise bubble up
            raise
        try:
            return youtube.search().list(**kwargs).execute()
        except HttpError as e:
            status = getattr(e.resp, 'status', None)
            # Skip invalid requests (processing failures)
            if status == 400:
                raise CommentsDisabled()
            if status == 403:
                try:
                    err_info = json.loads(e.content.decode())['error']['errors'][0]
                    reason = err_info.get('reason')
                except:
                    reason = None
                if reason == 'quotaExceeded':
                    print("⚠️ Comment quota hit — rotating key", flush=True)
                    rotate_key(); continue
                if reason == 'commentsDisabled':
                    raise CommentsDisabled()
            if status in (500, 503) and retries < 3:
                retries += 1
                time.sleep(min(BACKOFF_INITIAL * 2**(retries-1), BACKOFF_MAX))
                continue
            raise
            retries += 1
            time.sleep(min(BACKOFF_INITIAL * 2**(retries-1), BACKOFF_MAX))
            continue
            raise

def safe_videos_list(**kwargs):
    retries = 0
    while True:
        try:
            return youtube.videos().list(**kwargs).execute()
        except HttpError as e:
            status = getattr(e.resp, 'status', None)
            if status == 403:
                rotate_key(); continue
            if status in (500, 503) and retries < 3:
                retries += 1
                time.sleep(min(BACKOFF_INITIAL * 2**(retries-1), BACKOFF_MAX))
                continue
            raise

# Comments-safe helper: rotate on comment quota, skip on disabled
class CommentsDisabled(Exception):
    pass

def safe_comment_threads(**kwargs):
    """
    Safe wrapper for commentThreads.list: rotates key on quota, skips on disabled or invalid requests.
    """
    retries = 0
    while True:
        try:
            return youtube.commentThreads().list(**kwargs).execute()
        except HttpError as e:
            status = getattr(e.resp, 'status', None)
            # Skip invalid requests (processing failures)
            if status == 400:
                # treat as no comments for this video/window
                return {'items': [], 'nextPageToken': None}
            # Quota exceeded -> rotate key
            if status == 403:
                try:
                    err_info = json.loads(e.content.decode())['error']['errors'][0]
                    reason = err_info.get('reason')
                except:
                    reason = None
                if reason == 'quotaExceeded':
                    print("⚠️ Comment quota hit — rotating key", flush=True)
                    rotate_key()
                    continue
                if reason == 'commentsDisabled':
                    return {'items': [], 'nextPageToken': None}
            # Server errors -> retry
            if status in (500, 503) and retries < 3:
                retries += 1
                time.sleep(min(BACKOFF_INITIAL * 2**(retries-1), BACKOFF_MAX))
                continue
            # otherwise rethrow
            raise
        try:
            return youtube.commentThreads().list(**kwargs).execute()
        except HttpError as e:
            status = getattr(e.resp, 'status', None)
            if status == 403:
                try:
                    err_info = json.loads(e.content.decode())['error']['errors'][0]
                    reason = err_info.get('reason')
                except:
                    reason = None
                if reason == 'quotaExceeded':
                    print("⚠️ Comment quota hit — rotating key", flush=True)
                    rotate_key(); continue
                if reason == 'commentsDisabled':
                    raise CommentsDisabled()
            if status in (500, 503) and retries < 3:
                retries += 1
                time.sleep(min(BACKOFF_INITIAL * 2**(retries-1), BACKOFF_MAX))
                continue
            raise

# Year start/end bounds
def get_published_bounds(year):
    now = datetime.now(timezone.utc)
    start = f"{year}-01-01T00:00:00Z"
    end = f"{year}-12-31T23:59:59Z" if year < now.year else now.strftime('%Y-%m-%dT%H:%M:%SZ')
    return start, end

# Exclude likely music videos
def is_music_video(item):
    title = item['snippet']['title'].lower()
    desc = item['snippet'].get('description', '').lower()
    channel = item['snippet']['channelTitle'].lower()
    kws = ['official music video', 'lyrics', 'remix']
    chs = ['vevo']
    return any(kw in title for kw in kws) or any(kw in desc for kw in kws) or any(ch in channel for ch in chs)

# Adaptive binary-split sampling to reach VIDEO_SAMPLE_SIZE
def sample_videos(year):
    start_iso, end_iso = get_published_bounds(year)
    targets = VIDEO_SAMPLE_SIZE
    candidates = []
    seen = set()

    def fetch_window(s, e):
        ids = []
        # Iterate over all query terms and sort orders, regardless of interim target
        for term in QUERY_TERMS:
            for order in ('viewCount', 'date'):
                token = None
                while True:
                    resp = safe_search(
                        part='id', type='video', q=term, order=order,
                        maxResults=50, publishedAfter=s, publishedBefore=e, pageToken=token
                    )
                    # collect video IDs
                    ids.extend(item['id']['videoId'] for item in resp.get('items', []))
                    token = resp.get('nextPageToken')
                    if not token:
                        break
        # return up to the target sample size
        return ids[:targets]

    windows = [(start_iso, end_iso)]
    from datetime import datetime
    while windows and len(candidates) < targets:
        s, e = windows.pop(0)
        # Perform window search
        raw = fetch_window(s, e)
        valid_new = []
        for vid in raw:
            if vid in seen:
                continue
            info = safe_videos_list(part='snippet', id=vid)
            # Skip videos whose titles aren't detected as English
            title = info['items'][0]['snippet']['title']
            try:
                if detect(title) != 'en':
                    continue
            except:
                pass
            items = info.get('items', [])
            if not items:
                continue
            snip = items[0]['snippet']
            if not snip['publishedAt'].startswith(str(year)) or is_music_video(items[0]):
                continue
            valid_new.append(vid)
        # Report how many videos this window yielded
        print(f"🔍 Window {s}–{e} yielded {len(valid_new)} videos", flush=True)
        before = len(candidates)
        for vid in valid_new:
            if len(candidates) >= targets:
                break
            seen.add(vid)
            candidates.append(vid)
        got = len(candidates) - before
        needed = targets - before
        if len(candidates) < targets:
            # Split window in half and retry
            st = datetime.fromisoformat(s.replace('Z', '+00:00'))
            en = datetime.fromisoformat(e.replace('Z', '+00:00'))
            if (en - st).days > 1 and got < needed:
                mid = st + (en - st) / 2
                mid_iso = mid.strftime('%Y-%m-%dT%H:%M:%SZ')
                windows.insert(0, (mid_iso, e))
                windows.insert(0, (s, mid_iso))
    print(f"🎥 {len(candidates)} videos sampled for {year}", flush=True)
    return candidates

# Fetch comments with per-year and per-video caps
def fetch_comments_for_year(year, vids):
    out = []
    seen = set()
    for vid in vids:
        # Track comments fetched per video
        vid_before = len(out)
        tok = None
        cnt = 0
        while cnt < MAX_COMMENTS_PER_VIDEO:
            try:
                resp = safe_comment_threads(
                    part='snippet', videoId=vid, textFormat='plainText',
                    maxResults=COMMENT_PAGE_SIZE, pageToken=tok
                )
            except CommentsDisabled:
                break
            for th in resp.get('items', []):
                cs = th['snippet']['topLevelComment']['snippet']
                pub = cs.get('publishedAt', '')
                if not pub.startswith(str(year)): continue
                cid = th['snippet']['topLevelComment']['id']
                if cid in seen: continue
                seen.add(cid)
                txt = cs['textDisplay']
                try:
                    if detect(txt) != 'en': continue
                except:
                    pass
                out.append({'comment_id': cid, 'video_id': vid, 'text': txt, 'published_at': pub})
                cnt += 1
                if len(out) >= COMMENTS_PER_YEAR or cnt >= MAX_COMMENTS_PER_VIDEO:
                    break
            tok = resp.get('nextPageToken')
            if not tok:
                break
        # Report comments fetched for this video
        fetched = len(out) - vid_before
        print(f"💬 Video {vid} returned {fetched} comments", flush=True)
    return out

# Main execution
if __name__ == '__main__':
    def main():
        print("🚀 Starting scraping process...", flush=True)
        summary = {}
        for year in YEARS:
            print(f"=== Processing year {year} ===", flush=True)
            print(f"🚀 Sampling videos for {year}...", flush=True)
            vids = sample_videos(year)
            if len(vids) < MIN_VIDEOS_REQUIRED:
                print(f"⚠️ Only {len(vids)} videos for {year}, proceeding anyway.", flush=True)

            # Fetch video metadata
            VIDEO_TITLES.clear()
            VIDEO_PUBDATES.clear()
            for i in range(0, len(vids), 50):
                batch = vids[i:i+50]
                resp = safe_videos_list(part='snippet', id=','.join(batch))
                for it in resp.get('items', []):
                    VIDEO_TITLES[it['id']] = it['snippet']['title']
                    VIDEO_PUBDATES[it['id']] = it['snippet']['publishedAt']

            print(f"🚀 Fetching comments for {year}...", flush=True)
            comms = fetch_comments_for_year(year, vids)
            if len(comms) < MIN_COMMENTS_REQUIRED:
                print(f"⚠️ Only {len(comms)} comments for {year}, proceeding anyway.", flush=True)

            fn = f"comments_{year}.ndjson"
            with open(fn, 'w', encoding='utf-8') as f:
                for c in comms:
                    c['video_title'] = VIDEO_TITLES.get(c['video_id'], '')
                    c['video_published_at'] = VIDEO_PUBDATES.get(c['video_id'], '')
                    f.write(json.dumps(c, ensure_ascii=False) + "\n")
            print(f"✅ Wrote {len(comms)} comments for {year} to {fn}", flush=True)
            summary[year] = {'videos': len(vids), 'comments': len(comms)}

        print("✅ Summary:", summary, flush=True)

    main()
