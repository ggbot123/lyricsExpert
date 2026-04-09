import argparse
import json
import re
from typing import Dict, List

from lyric_wordcloud import clean_line


TIMESTAMP_PATTERN = re.compile(r"\[\d{1,2}:\d{2}(?:\.\d{1,3})?\]")


def load_lyrics(json_path: str) -> List[Dict[str, str]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("lyrics", [])
    if not isinstance(items, list):
        raise ValueError("输入 JSON 格式不正确：缺少 lyrics 列表")

    results: List[Dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        song_name = str(item.get("song_name") or "").strip()
        lyric = str(item.get("lyric") or "")
        if not song_name:
            song_name = f"song_{item.get('song_id', 'unknown')}"
        if lyric.strip():
            results.append({"song_name": song_name, "lyric": lyric})

    return results


def strip_timestamps(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = TIMESTAMP_PATTERN.sub("", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def search_keyword(
    songs: List[Dict[str, str]],
    keyword: str,
    clean_timestamp: bool,
    show_lines: bool,
) -> List[Dict[str, object]]:
    matched: List[Dict[str, object]] = []

    for song in songs:
        song_name = song["song_name"]
        lyric = song["lyric"]
        if clean_timestamp:
            lines = []
            for raw_line in lyric.splitlines():
                no_ts = TIMESTAMP_PATTERN.sub("", raw_line).strip()
                cleaned = clean_line(no_ts)
                if cleaned:
                    lines.append(cleaned)
            text = "\n".join(lines)
        else:
            text = lyric

        count = text.count(keyword)
        if count <= 0:
            continue

        hit_lines: List[str] = []
        if show_lines:
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if keyword in line:
                    hit_lines.append(line)

        matched.append(
            {
                "song_name": song_name,
                "count": count,
                "hit_lines": hit_lines,
            }
        )

    matched.sort(key=lambda x: (-int(x["count"]), str(x["song_name"])))
    return matched


def main() -> None:
    parser = argparse.ArgumentParser(description="搜索歌词中包含指定词语的歌曲")
    parser.add_argument("target_json", type=str, help="目标歌词 JSON 路径")
    parser.add_argument("keyword", type=str, help="要搜索的词语")
    parser.add_argument(
        "--keep-timestamp",
        action="store_true",
        help="保留时间戳参与搜索（默认会移除时间戳）",
    )
    parser.add_argument(
        "--show-lines",
        action="store_true",
        help="显示命中的歌词行",
    )

    args = parser.parse_args()
    keyword = args.keyword.strip()
    if not keyword:
        raise ValueError("keyword 不能为空")

    songs = load_lyrics(args.target_json)
    matched = search_keyword(
        songs=songs,
        keyword=keyword,
        clean_timestamp=not args.keep_timestamp,
        show_lines=args.show_lines,
    )

    if not matched:
        print(f"未找到包含“{keyword}”的歌曲")
        return

    total_count = sum(int(item["count"]) for item in matched)
    print(f"关键词: {keyword}")
    print(f"命中歌曲数: {len(matched)}")
    print(f"总出现次数: {total_count}")
    print("-" * 40)

    for idx, item in enumerate(matched, start=1):
        song_name = str(item["song_name"])
        count = int(item["count"])
        print(f"{idx}. {song_name} (出现 {count} 次)")

        hit_lines = item.get("hit_lines", [])
        if isinstance(hit_lines, list) and hit_lines:
            for line in hit_lines:
                print(f"   - {line}")


if __name__ == "__main__":
    main()
