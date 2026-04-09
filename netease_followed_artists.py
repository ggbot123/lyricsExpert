import argparse
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

import requests

from lyric_reptile import (
    deduplicate_by_lyric_signature,
    fetch_song_lyric,
    load_artist_songs,
    pick_unique_songs_by_name,
)
from netease_artist_songs import DEFAULT_HEADERS, fetch_artist_songs

FOLLOWED_ARTISTS_URL = "https://music.163.com/api/artist/sublist"


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def has_valid_lyrics_data(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False

    lyrics = data.get("lyrics", [])
    if not isinstance(lyrics, list) or not lyrics:
        return False

    for item in lyrics:
        if isinstance(item, dict) and str(item.get("lyric") or "").strip():
            return True
    return False


def fetch_followed_artists(
    cookie: str,
    page_size: int = 100,
    sleep_seconds: float = 0.25,
    timeout: int = 10,
) -> List[Dict[str, Any]]:
    session = requests.Session()
    headers = dict(DEFAULT_HEADERS)
    headers["Cookie"] = cookie

    offset = 0
    artists: List[Dict[str, Any]] = []

    while True:
        params = {
            "limit": page_size,
            "offset": offset,
            "total": "true",
        }
        response = session.get(
            FOLLOWED_ARTISTS_URL,
            headers=headers,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()

        page_items = payload.get("data", [])
        if not isinstance(page_items, list) or not page_items:
            break

        for item in page_items:
            if not isinstance(item, dict):
                continue
            artist_id = item.get("id")
            if not isinstance(artist_id, int):
                continue

            artists.append(
                {
                    "artist_id": artist_id,
                    "artist_name": item.get("name") or "",
                    "album_size": item.get("albumSize"),
                    "mv_size": item.get("mvSize"),
                    "pic_url": item.get("picUrl") or "",
                    "trans": item.get("trans") or "",
                    "alias": item.get("alias") or [],
                }
            )

        has_more = bool(payload.get("more"))
        offset += len(page_items)
        print(f"fetched followed artists: {len(artists)}")

        if not has_more:
            break
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return artists


def build_lyrics_from_catalog(
    artist_json_path: str,
    lyrics_output_path: str,
    song_list_output_path: str,
    timeout: int = 10,
    sleep_seconds: float = 0.1,
    fuzzy_threshold: float = 0.97,
) -> None:
    artist_info = load_artist_songs(artist_json_path)
    artist_id = artist_info.get("artist_id")
    song_items: List[Dict[str, Any]] = artist_info.get("songs", [])

    if not song_items:
        raise ValueError("artist json 中没有可用歌曲")

    selected_songs = pick_unique_songs_by_name(song_items)
    results: List[Dict[str, Any]] = []

    for index, song in enumerate(selected_songs, start=1):
        song_id = int(song["song_id"])
        song_name = str(song.get("song_name") or "")
        normalized_song_name = str(song.get("normalized_song_name") or "")

        try:
            lyric_data = fetch_song_lyric(song_id=song_id, timeout=timeout)
            lyric_data["song_name"] = song_name
            lyric_data["normalized_song_name"] = normalized_song_name
            results.append(lyric_data)
            print(f"lyric ok [{index}/{len(selected_songs)}]: {song_id}")
        except Exception as e:
            results.append(
                {
                    "song_id": song_id,
                    "song_name": song_name,
                    "normalized_song_name": normalized_song_name,
                    "code": None,
                    "lyric": "",
                    "translated_lyric": "",
                    "error": str(e),
                }
            )
            print(f"lyric fail [{index}/{len(selected_songs)}]: {song_id}, error={e}")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    threshold = max(0.0, min(1.0, fuzzy_threshold))
    deduped = deduplicate_by_lyric_signature(results, fuzzy_threshold=threshold)
    final_data = {
        "artist_id": artist_id,
        "source_artist_json": artist_json_path,
        "total_song_count": len(song_items),
        "selected_song_count": len(selected_songs),
        "name_deduped_count": len(results),
        "lyric_deduped_count": len(deduped),
        "fuzzy_threshold": threshold,
        "fetched_count": len(results),
        "lyrics": deduped,
    }

    write_json(lyrics_output_path, final_data)

    with open(song_list_output_path, "w", encoding="utf-8") as f:
        for item in deduped:
            f.write(f"{item.get('song_id')}\t{str(item.get('song_name') or '')}\n")


def save_followed_artists_snapshot(output_dir: str, artists: List[Dict[str, Any]]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    snapshot_path = os.path.join(output_dir, "followed_artists.json")
    data = {
        "fetched_at": int(time.time()),
        "total": len(artists),
        "artists": artists,
    }
    write_json(snapshot_path, data)
    return snapshot_path


def save_followed_artists_text(output_dir: str, artists: List[Dict[str, Any]]) -> str:
    txt_path = os.path.join(output_dir, "followed_artists.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for item in artists:
            artist_id = item.get("artist_id")
            artist_name = str(item.get("artist_name") or "")
            f.write(f"{artist_id}\t{artist_name}\n")
    return txt_path


def save_artist_catalogs(
    output_dir: str,
    artists: List[Dict[str, Any]],
    cookie: str,
    page_size: int,
    sleep_seconds: float,
    timeout: int,
    refresh_existing: bool,
    fetch_lyrics: bool,
    lyric_sleep: float,
    lyric_timeout: int,
    progress_callback: Optional[Callable[[int, int, str, int, str], None]] = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    total = len(artists)

    for index, artist in enumerate(artists, start=1):
        artist_id = int(artist["artist_id"])
        artist_name = str(artist.get("artist_name") or "")
        artist_json_path = os.path.join(output_dir, f"artist_{artist_id}.json")
        lyrics_json_path = os.path.join(output_dir, f"lyrics_artist_{artist_id}.json")
        song_list_path = os.path.join(output_dir, f"song_list_artist_{artist_id}.txt")

        should_fetch_catalog = refresh_existing or (not os.path.exists(artist_json_path))
        if should_fetch_catalog:
            if progress_callback is not None:
                progress_callback(index, total, "catalog_fetch", artist_id, artist_name)
            print(f"[{index}/{len(artists)}] fetch songs for artist: {artist_id} {artist_name}")
            catalog = fetch_artist_songs(
                artist_id=artist_id,
                page_size=page_size,
                sleep_seconds=sleep_seconds,
                timeout=timeout,
                cookie=cookie,
            )
            catalog["resolved_artist_name"] = artist_name
            write_json(artist_json_path, catalog)
        else:
            if progress_callback is not None:
                progress_callback(index, total, "catalog_cache", artist_id, artist_name)
            print(f"[{index}/{len(artists)}] skip songs cache: {artist_id} {artist_name}")

        if not fetch_lyrics:
            if progress_callback is not None:
                progress_callback(index, total, "artist_done", artist_id, artist_name)
            continue

        should_fetch_lyrics = refresh_existing or (not has_valid_lyrics_data(lyrics_json_path))
        if should_fetch_lyrics:
            if progress_callback is not None:
                progress_callback(index, total, "lyrics_fetch", artist_id, artist_name)
            print(f"[{index}/{len(artists)}] fetch lyrics for artist: {artist_id} {artist_name}")
            build_lyrics_from_catalog(
                artist_json_path=artist_json_path,
                lyrics_output_path=lyrics_json_path,
                song_list_output_path=song_list_path,
                timeout=lyric_timeout,
                sleep_seconds=lyric_sleep,
            )
        else:
            if progress_callback is not None:
                progress_callback(index, total, "lyrics_cache", artist_id, artist_name)
            print(f"[{index}/{len(artists)}] skip lyrics cache: {artist_id} {artist_name}")

        if progress_callback is not None:
            progress_callback(index, total, "artist_done", artist_id, artist_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="读取网易云账号关注艺人并落盘到本地 lyrics_output"
    )
    parser.add_argument(
        "--cookie",
        type=str,
        default="",
        help="网易云登录 Cookie；不传则尝试读取环境变量 NETEASE_COOKIE",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="lyrics_output",
        help="输出目录，默认 lyrics_output",
    )
    parser.add_argument(
        "--max-artists",
        type=int,
        default=0,
        help="最多处理多少位关注艺人，0 表示全部",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="拉取关注列表和歌曲列表时的分页大小，默认 100",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="请求间隔秒数，默认 0.25",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="请求超时秒数，默认 10",
    )
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="强制刷新已有 artist/lyrics 缓存",
    )
    parser.add_argument(
        "--fetch-lyrics",
        action="store_true",
        help="同时抓取每位艺人的歌词缓存（耗时更长）",
    )
    parser.add_argument(
        "--lyric-sleep",
        type=float,
        default=0.1,
        help="抓歌词时每首歌间隔秒数，默认 0.1",
    )
    parser.add_argument(
        "--lyric-timeout",
        type=int,
        default=10,
        help="抓歌词请求超时秒数，默认 10",
    )
    args = parser.parse_args()

    cookie = (args.cookie or os.getenv("NETEASE_COOKIE", "")).strip()
    if not cookie:
        raise ValueError("请通过 --cookie 或环境变量 NETEASE_COOKIE 提供登录 Cookie")

    artists = fetch_followed_artists(
        cookie=cookie,
        page_size=args.page_size,
        sleep_seconds=args.sleep,
        timeout=args.timeout,
    )
    if not artists:
        raise ValueError("未读取到关注艺人，请检查 Cookie 是否有效")

    if args.max_artists > 0:
        artists = artists[: args.max_artists]

    snapshot_path = save_followed_artists_snapshot(args.output_dir, artists)
    txt_path = save_followed_artists_text(args.output_dir, artists)
    print(f"saved followed artists json: {snapshot_path}")
    print(f"saved followed artists txt: {txt_path}")

    save_artist_catalogs(
        output_dir=args.output_dir,
        artists=artists,
        cookie=cookie,
        page_size=args.page_size,
        sleep_seconds=args.sleep,
        timeout=args.timeout,
        refresh_existing=args.refresh_existing,
        fetch_lyrics=args.fetch_lyrics,
        lyric_sleep=args.lyric_sleep,
        lyric_timeout=args.lyric_timeout,
    )

    print(f"done, total artists processed: {len(artists)}")


if __name__ == "__main__":
    main()