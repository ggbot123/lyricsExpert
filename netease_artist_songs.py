import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import requests

BASE_URL = "https://music.163.com/api/v1/artist/songs"
SEARCH_URL = "https://music.163.com/api/search/get"
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://music.163.com/",
    "Accept": "application/json, text/plain, */*",
}


def resolve_artist_id_by_name(
    artist_name: str,
    timeout: int = 10,
    cookie: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve artist id by artist name using Netease search API."""
    session = requests.Session()
    headers = dict(DEFAULT_HEADERS)
    if cookie:
        headers["Cookie"] = cookie

    params = {
        "s": artist_name,
        "type": 100,
        "limit": 20,
        "offset": 0,
    }
    response = session.get(SEARCH_URL, headers=headers, params=params, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    artists = (((data or {}).get("result") or {}).get("artists") or [])
    if not artists:
        raise ValueError(f"未找到歌手: {artist_name}")

    target = artist_name.strip().lower()
    selected = None
    for artist in artists:
        name = str(artist.get("name", "")).strip().lower()
        if name == target:
            selected = artist
            break

    if selected is None:
        selected = artists[0]

    return {
        "artist_id": selected.get("id"),
        "artist_name": selected.get("name"),
        "matched_candidates": [
            {"id": item.get("id"), "name": item.get("name")} for item in artists
        ],
    }


def fetch_artist_songs(
    artist_id: int,
    page_size: int = 100,
    sleep_seconds: float = 0.25,
    timeout: int = 10,
    cookie: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch all songs for one artist by paging through Netease web API."""
    session = requests.Session()
    headers = dict(DEFAULT_HEADERS)
    if cookie:
        headers["Cookie"] = cookie

    songs: List[Dict[str, Any]] = []
    offset = 0
    expected_total: Optional[int] = None

    while True:
        params = {"id": artist_id, "limit": page_size, "offset": offset}
        response = session.get(BASE_URL, headers=headers, params=params, timeout=timeout)
        response.raise_for_status()

        data = response.json()
        page_songs = data.get("songs", [])

        if expected_total is None:
            expected_total = data.get("total")

        if not page_songs:
            break

        songs.extend(page_songs)
        offset += page_size

        if expected_total is not None and offset >= expected_total:
            break

        time.sleep(sleep_seconds)

    result_songs: List[Dict[str, Any]] = []
    song_ids: List[int] = []

    for song in songs:
        song_id = song.get("id")
        if song_id is not None:
            song_ids.append(song_id)

        result_songs.append(
            {
                "id": song_id,
                "name": song.get("name"),
                "alias": song.get("alia", []),
                "album": (song.get("al") or {}).get("name"),
            }
        )

    return {
        "artist_id": artist_id,
        "total": len(result_songs),
        "song_ids": song_ids,
        "songs": result_songs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Get all songs of a Netease artist by artist id or artist name."
    )
    parser.add_argument(
        "artist_id",
        nargs="?",
        type=int,
        help="Netease artist ID (optional if --artist-name is used)",
    )
    parser.add_argument(
        "--artist-name",
        type=str,
        default="",
        help="artist name used to auto-resolve artist_id",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=100,
        help="page size for each request (default: 100)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="sleep seconds between requests (default: 0.25)",
    )
    parser.add_argument(
        "--timeout", type=int, default=10, help="request timeout seconds (default: 10)"
    )
    parser.add_argument(
        "--cookie",
        type=str,
        default="",
        help="optional request cookie if anti-scrape is triggered",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="output json file path; defaults to lyrics_output/artist_<id>.json",
    )

    args = parser.parse_args()

    artist_id = args.artist_id
    resolved_artist_name = ""
    matched_candidates: List[Dict[str, Any]] = []

    if args.artist_name:
        resolved = resolve_artist_id_by_name(
            artist_name=args.artist_name,
            timeout=args.timeout,
            cookie=args.cookie or None,
        )
        artist_id = resolved.get("artist_id")
        resolved_artist_name = str(resolved.get("artist_name") or "")
        matched_candidates = resolved.get("matched_candidates", [])

    if artist_id is None:
        raise ValueError("请提供 artist_id，或使用 --artist-name 自动查询")

    data = fetch_artist_songs(
        artist_id=artist_id,
        page_size=args.page_size,
        sleep_seconds=args.sleep,
        timeout=args.timeout,
        cookie=args.cookie or None,
    )

    if resolved_artist_name:
        data["resolved_artist_name"] = resolved_artist_name
        data["matched_candidates"] = matched_candidates

    print(f"artist_id={data['artist_id']}, total={data['total']}")
    if resolved_artist_name:
        print(f"resolved_artist_name={resolved_artist_name}")
    print(f"song_ids(sample first 20)={data['song_ids'][:20]}")

    output_path = args.output.strip()
    if not output_path:
        os.makedirs("lyrics_output", exist_ok=True)
        output_path = os.path.join("lyrics_output", f"artist_{data['artist_id']}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"saved to: {output_path}")


if __name__ == "__main__":
    main()
