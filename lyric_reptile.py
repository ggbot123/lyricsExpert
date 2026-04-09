import argparse
import difflib
import json
import os
import re
import time
import unicodedata
from typing import Any, Dict, List

import requests

LYRIC_URL = "https://music.163.com/api/song/lyric"
BRACKET_CONTENT_PATTERN = re.compile(r"[\(（][^\)）]*[\)）]")


def normalize_song_name(song_name: str) -> str:
	# Remove (...) and （...） fragments before comparing duplicate titles.
	normalized = BRACKET_CONTENT_PATTERN.sub("", song_name)
	normalized = " ".join(normalized.split())
	return normalized.strip().lower()


def load_artist_songs(json_path: str) -> Dict[str, Any]:
	with open(json_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	artist_id = data.get("artist_id")
	songs_raw = data.get("songs", [])
	items: List[Dict[str, Any]] = []

	if isinstance(songs_raw, list) and songs_raw:
		for song in songs_raw:
			song_id = song.get("id")
			song_name = str(song.get("name") or "").strip()
			if not isinstance(song_id, int):
				continue
			if not song_name:
				song_name = f"song_{song_id}"
			items.append(
				{
					"song_id": song_id,
					"song_name": song_name,
					"normalized_song_name": normalize_song_name(song_name),
				}
			)
	else:
		song_ids = data.get("song_ids", [])
		if not isinstance(song_ids, list):
			raise ValueError("artist json 中的 songs/song_ids 格式不正确")
		for song_id in song_ids:
			if not isinstance(song_id, int):
				continue
			song_name = f"song_{song_id}"
			items.append(
				{
					"song_id": song_id,
					"song_name": song_name,
					"normalized_song_name": normalize_song_name(song_name),
				}
			)

	return {
		"artist_id": artist_id,
		"songs": items,
		"source": data,
	}


def pick_unique_songs_by_name(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	seen = set()
	selected: List[Dict[str, Any]] = []

	for item in items:
		key = str(item.get("normalized_song_name") or "").strip()
		if not key:
			key = str(item.get("song_name") or "").strip().lower()
		if key in seen:
			continue
		seen.add(key)
		selected.append(item)

	return selected


def fetch_song_lyric(song_id: int, timeout: int = 10) -> Dict[str, Any]:
	params = {
		"os": "pc",
		"id": song_id,
		"lv": -1,
		"kv": -1,
		"tv": -1,
	}
	response = requests.get(LYRIC_URL, params=params, timeout=timeout)
	response.raise_for_status()
	payload = response.json()

	return {
		"song_id": song_id,
		"code": payload.get("code"),
		"lyric": ((payload.get("lrc") or {}).get("lyric") or ""),
		"translated_lyric": ((payload.get("tlyric") or {}).get("lyric") or ""),
	}


def normalize_lyric_line(line: str) -> str:
	line = re.sub(r"\[[0-9]{1,2}:[0-9]{1,2}(\.[0-9]{1,3})?\]", "", line)
	line = line.strip().lower()

	chars: List[str] = []
	for ch in line:
		cat = unicodedata.category(ch)
		if cat.startswith("P"):
			continue
		if ch.isspace():
			continue
		chars.append(ch)
	return "".join(chars)


def is_non_lyric_line(normalized_line: str) -> bool:
	meta_keywords = ["作词", "作曲", "编曲", "制作人", "制作"]
	for keyword in meta_keywords:
		if keyword in normalized_line:
			return True
	return False


def build_first_five_signature(lyric: str) -> str:
	lines = lyric.splitlines()
	normalized: List[str] = []
	for line in lines:
		clean = normalize_lyric_line(line)
		if clean and not is_non_lyric_line(clean):
			normalized.append(clean)
		if len(normalized) >= 5:
			break
	return "|".join(normalized)


def build_normalized_prefix_signature(lyric: str, prefix_chars: int = 120) -> str:
	lines = lyric.splitlines()
	parts: List[str] = []
	for line in lines:
		clean = normalize_lyric_line(line)
		if not clean or is_non_lyric_line(clean):
			continue
		parts.append(clean)

	full_text = "".join(parts)
	if not full_text:
		return ""
	return full_text[:prefix_chars]


def similar_enough(a: str, b: str, threshold: float) -> bool:
	if a == b:
		return True
	if not a or not b:
		return False
	return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def deduplicate_by_lyric_signature(
	items: List[Dict[str, Any]],
	fuzzy_threshold: float = 0.97,
) -> List[Dict[str, Any]]:
	seen_signatures: List[str] = []
	result: List[Dict[str, Any]] = []

	for item in items:
		lyric = str(item.get("lyric") or "")
		line_signature = build_first_five_signature(lyric)
		prefix_signature = build_normalized_prefix_signature(lyric, prefix_chars=120)
		signature = prefix_signature or line_signature
		if not signature:
			signature = f"__EMPTY__:{item.get('song_id')}"

		duplicate = False
		for existing in seen_signatures:
			if similar_enough(signature, existing, fuzzy_threshold):
				duplicate = True
				break

		if duplicate:
			continue
		seen_signatures.append(signature)
		result.append(item)

	return result


def main() -> None:
	parser = argparse.ArgumentParser(description="批量抓取 artist_xxx.json 中所有歌曲的歌词")
	parser.add_argument(
		"artist_json",
		type=str,
		help="artist json 文件路径，例如 lyrics_output/artist_12345.json",
	)
	parser.add_argument(
		"--output",
		type=str,
		default="",
		help="输出文件路径，默认: lyrics_output/lyrics_artist_<artist_id>.json",
	)
	parser.add_argument(
		"--song-list-output",
		type=str,
		default="",
		help="歌名列表输出路径，默认: lyrics_output/song_list_artist_<artist_id>.txt",
	)
	parser.add_argument(
		"--fuzzy-threshold",
		type=float,
		default=0.97,
		help="歌词近似去重阈值(0-1)，越高越严格，默认: 0.97",
	)
	parser.add_argument("--timeout", type=int, default=10, help="请求超时秒数")
	parser.add_argument("--sleep", type=float, default=0.1, help="每次请求间隔秒数")
	args = parser.parse_args()

	artist_info = load_artist_songs(args.artist_json)
	artist_id = artist_info.get("artist_id")
	song_items: List[Dict[str, Any]] = artist_info.get("songs", [])

	if not song_items:
		raise ValueError("未在 artist json 中找到可用歌曲")

	selected_songs = pick_unique_songs_by_name(song_items)

	results: List[Dict[str, Any]] = []
	for index, song in enumerate(selected_songs, start=1):
		song_id = int(song["song_id"])
		song_name = str(song.get("song_name") or "")
		normalized_song_name = str(song.get("normalized_song_name") or "")
		try:
			lyric_data = fetch_song_lyric(song_id=song_id, timeout=args.timeout)
			lyric_data["song_name"] = song_name
			lyric_data["normalized_song_name"] = normalized_song_name
			results.append(lyric_data)
			print(f"[{index}/{len(selected_songs)}] ok: {song_id}")
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
			print(f"[{index}/{len(selected_songs)}] fail: {song_id}, error={e}")

		if args.sleep > 0:
			time.sleep(args.sleep)

	after_name_dedup = len(results)
	threshold = max(0.0, min(1.0, args.fuzzy_threshold))
	results = deduplicate_by_lyric_signature(results, fuzzy_threshold=threshold)

	os.makedirs("lyrics_output", exist_ok=True)

	output_path = args.output.strip()
	if not output_path:
		output_path = os.path.join("lyrics_output", f"lyrics_artist_{artist_id}.json")

	song_list_output_path = args.song_list_output.strip()
	if not song_list_output_path:
		song_list_output_path = os.path.join(
			"lyrics_output", f"song_list_artist_{artist_id}.txt"
		)

	final_data = {
		"artist_id": artist_id,
		"source_artist_json": args.artist_json,
		"total_song_count": len(song_items),
		"selected_song_count": len(selected_songs),
		"name_deduped_count": after_name_dedup,
		"lyric_deduped_count": len(results),
		"fuzzy_threshold": threshold,
		"fetched_count": after_name_dedup,
		"lyrics": results,
	}

	with open(output_path, "w", encoding="utf-8") as f:
		json.dump(final_data, f, ensure_ascii=False, indent=2)

	with open(song_list_output_path, "w", encoding="utf-8") as f:
		for item in results:
			song_id = item.get("song_id")
			song_name = str(item.get("song_name") or "")
			f.write(f"{song_id}\t{song_name}\n")

	print(f"saved to: {output_path}")
	print(f"saved song list to: {song_list_output_path}")


if __name__ == "__main__":
	main()