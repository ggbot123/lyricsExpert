import json
import os
import time
import base64
import io
import threading
import uuid
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

from flask import Flask, jsonify, redirect, render_template, request, url_for
import requests

import lyric_wordcloud as lw
import lyrics_search as ls
from lyric_reptile import (
    deduplicate_by_lyric_signature,
    fetch_song_lyric,
    load_artist_songs,
    pick_unique_songs_by_name,
)
from netease_artist_songs import fetch_artist_songs, resolve_artist_id_by_name
from netease_followed_artists import (
    fetch_followed_artists,
    save_artist_catalogs,
    save_followed_artists_snapshot,
    save_followed_artists_text,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "lyrics_output")
STATIC_GENERATED_DIR = os.path.join(BASE_DIR, "static", "generated")

# In-script defaults for AI provider settings.
# Replace OPENROUTER_API_KEY with your real key for local usage.
os.environ.setdefault("OPENROUTER_API_KEY", "")
os.environ.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
os.environ.setdefault("OPENAI_MODEL", "deepseek/deepseek-v3.2")
os.environ.setdefault("OPENROUTER_SITE_URL", "http://localhost:5000")
os.environ.setdefault("OPENROUTER_APP_NAME", "lyricExpert")

app = Flask(__name__, static_folder="static", template_folder="templates")
PREPARE_TASKS: Dict[str, Dict[str, Any]] = {}
PREPARE_TASKS_LOCK = threading.Lock()
FOLLOWED_IMPORT_TASKS: Dict[str, Dict[str, Any]] = {}
FOLLOWED_IMPORT_TASKS_LOCK = threading.Lock()
WORD_RATIO_TASKS: Dict[str, Dict[str, Any]] = {}
WORD_RATIO_TASKS_LOCK = threading.Lock()


def ensure_directories() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(STATIC_GENERATED_DIR, exist_ok=True)


def artist_file_paths(
    artist_id: int,
    keep_single_char: bool = True,
    count_by_song: bool = False,
    filter_english_words: bool = True,
) -> Dict[str, str]:
    mode_suffix = "keep1" if keep_single_char else "keep0"
    count_suffix = "song1" if count_by_song else "song0"
    english_suffix = "eng1" if filter_english_words else "eng0"
    return {
        "artist": os.path.join(OUTPUT_DIR, f"artist_{artist_id}.json"),
        "lyrics": os.path.join(OUTPUT_DIR, f"lyrics_artist_{artist_id}.json"),
        "song_list": os.path.join(OUTPUT_DIR, f"song_list_artist_{artist_id}.txt"),
        "wordcloud_image": os.path.join(
            STATIC_GENERATED_DIR,
            f"wordcloud_artist_{artist_id}_{mode_suffix}_{count_suffix}_{english_suffix}.png",
        ),
        "top_words": os.path.join(
            STATIC_GENERATED_DIR,
            f"top_words_artist_{artist_id}_{mode_suffix}_{count_suffix}_{english_suffix}.txt",
        ),
        "cleaned": os.path.join(
            STATIC_GENERATED_DIR,
            f"cleaned_artist_{artist_id}_{mode_suffix}_{count_suffix}_{english_suffix}.txt",
        ),
    }


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def has_valid_lyrics_data(lyrics_json_path: str) -> bool:
    if not os.path.exists(lyrics_json_path):
        return False

    try:
        data = read_json(lyrics_json_path)
    except Exception:
        return False

    lyrics = data.get("lyrics", [])
    if not isinstance(lyrics, list) or not lyrics:
        return False

    for item in lyrics:
        if isinstance(item, dict) and str(item.get("lyric") or "").strip():
            return True

    return False


def resolve_artist(artist_name: str) -> Tuple[int, str]:
    artist_name = (artist_name or "").strip()
    if not artist_name:
        raise ValueError("artist_name 不能为空")

    resolved = resolve_artist_id_by_name(artist_name=artist_name)
    artist_id = int(resolved.get("artist_id"))
    resolved_name = str(resolved.get("artist_name") or artist_name)
    return artist_id, resolved_name


def ensure_artist_catalog(artist_id: int, artist_name: str = "") -> str:
    paths = artist_file_paths(artist_id)
    artist_json_path = paths["artist"]

    if os.path.exists(artist_json_path):
        return artist_json_path

    catalog = fetch_artist_songs(artist_id=artist_id)
    if artist_name:
        catalog["resolved_artist_name"] = artist_name
    write_json(artist_json_path, catalog)
    return artist_json_path


def build_lyrics_from_catalog(
    artist_json_path: str,
    lyrics_output_path: str,
    song_list_output_path: str,
    timeout: int = 10,
    sleep_seconds: float = 0.1,
    fuzzy_threshold: float = 0.97,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    artist_info = load_artist_songs(artist_json_path)
    artist_id = artist_info.get("artist_id")
    song_items: List[Dict[str, Any]] = artist_info.get("songs", [])

    if not song_items:
        raise ValueError("artist json 中没有可用歌曲")

    selected_songs = pick_unique_songs_by_name(song_items)
    results: List[Dict[str, Any]] = []

    total = len(selected_songs)
    for index, song in enumerate(selected_songs, start=1):
        song_id = int(song["song_id"])
        song_name = str(song.get("song_name") or "")
        normalized_song_name = str(song.get("normalized_song_name") or "")

        try:
            lyric_data = fetch_song_lyric(song_id=song_id, timeout=timeout)
            lyric_data["song_name"] = song_name
            lyric_data["normalized_song_name"] = normalized_song_name
            results.append(lyric_data)
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

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

        if progress_callback is not None:
            progress_callback(index, total)

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


def ensure_artist_data(artist_id: int, artist_name: str = "") -> Dict[str, Any]:
    ensure_directories()
    paths = artist_file_paths(artist_id)

    artist_json_path = ensure_artist_catalog(artist_id=artist_id, artist_name=artist_name)

    if not has_valid_lyrics_data(paths["lyrics"]):
        build_lyrics_from_catalog(
            artist_json_path=artist_json_path,
            lyrics_output_path=paths["lyrics"],
            song_list_output_path=paths["song_list"],
        )

    return paths


def update_prepare_task(task_id: str, **updates: Any) -> None:
    with PREPARE_TASKS_LOCK:
        task = PREPARE_TASKS.get(task_id)
        if not task:
            return
        task.update(updates)
        task["updated_at"] = time.time()


def prepare_artist_data_worker(task_id: str, artist_id: int, artist_name: str) -> None:
    try:
        update_prepare_task(task_id, status="running", progress=5, message="开始准备歌手数据")
        ensure_directories()
        paths = artist_file_paths(artist_id)

        update_prepare_task(task_id, progress=10, message="检查本地歌单缓存")
        catalog_exists = os.path.exists(paths["artist"])
        if not catalog_exists:
            update_prepare_task(task_id, progress=20, message="正在获取歌手歌曲库")
            ensure_artist_catalog(artist_id=artist_id, artist_name=artist_name)
        else:
            update_prepare_task(task_id, progress=30, message="已命中本地歌曲库缓存")

        update_prepare_task(task_id, progress=40, message="检查本地歌词缓存")
        lyrics_exists = has_valid_lyrics_data(paths["lyrics"])
        if not lyrics_exists:
            update_prepare_task(task_id, progress=45, message="正在抓取歌词数据")

            def lyric_progress(current: int, total: int) -> None:
                if total <= 0:
                    return
                percent = 45 + int((current / total) * 40)
                update_prepare_task(
                    task_id,
                    progress=min(85, percent),
                    message=f"正在抓取歌词 {current}/{total}",
                )

            build_lyrics_from_catalog(
                artist_json_path=paths["artist"],
                lyrics_output_path=paths["lyrics"],
                song_list_output_path=paths["song_list"],
                progress_callback=lyric_progress,
            )
        else:
            update_prepare_task(task_id, progress=85, message="已命中本地歌词缓存")

        update_prepare_task(task_id, progress=90, message="准备词云资源")
        try:
            ensure_wordcloud_assets(artist_id=artist_id, lyrics_json_path=paths["lyrics"])
        except OSError as e:
            if getattr(e, "errno", None) != 28:
                raise
            update_prepare_task(task_id, progress=95, message="磁盘空间不足，词云将使用临时模式")

        update_prepare_task(task_id, status="completed", progress=100, message="歌手数据准备完成")
    except Exception as e:
        update_prepare_task(task_id, status="failed", error=str(e), message=f"准备失败: {e}")


def create_prepare_task(artist_id: int, artist_name: str) -> Dict[str, Any]:
    task_id = uuid.uuid4().hex
    now = time.time()
    task = {
        "task_id": task_id,
        "artist_id": artist_id,
        "artist_name": artist_name,
        "status": "pending",
        "progress": 0,
        "message": "任务已创建",
        "error": "",
        "created_at": now,
        "updated_at": now,
    }

    with PREPARE_TASKS_LOCK:
        PREPARE_TASKS[task_id] = task

    thread = threading.Thread(
        target=prepare_artist_data_worker,
        args=(task_id, artist_id, artist_name),
        daemon=True,
    )
    thread.start()
    return task


def update_followed_import_task(task_id: str, **updates: Any) -> None:
    with FOLLOWED_IMPORT_TASKS_LOCK:
        task = FOLLOWED_IMPORT_TASKS.get(task_id)
        if not task:
            return
        task.update(updates)
        task["updated_at"] = time.time()


def followed_import_worker(task_id: str, params: Dict[str, Any]) -> None:
    try:
        cookie = str(params.get("cookie") or "").strip()
        output_dir = str(params.get("output_dir") or OUTPUT_DIR).strip() or OUTPUT_DIR
        max_artists = int(params.get("max_artists", 0) or 0)
        page_size = int(params.get("page_size", 100) or 100)
        sleep_seconds = float(params.get("sleep_seconds", 0.25) or 0.25)
        timeout = int(params.get("timeout", 10) or 10)
        refresh_existing = bool(params.get("refresh_existing", False))
        fetch_lyrics = bool(params.get("fetch_lyrics", False))
        lyric_sleep = float(params.get("lyric_sleep", 0.1) or 0.1)
        lyric_timeout = int(params.get("lyric_timeout", 10) or 10)
        selected_artist_ids = set(params.get("selected_artist_ids") or [])

        update_followed_import_task(task_id, status="running", progress=5, message="开始读取关注艺人列表")
        artists = fetch_followed_artists(
            cookie=cookie,
            page_size=max(1, page_size),
            sleep_seconds=max(0.0, sleep_seconds),
            timeout=max(1, timeout),
        )
        if not artists:
            raise ValueError("未读取到关注艺人，请检查 Cookie 是否有效")

        total_followed = len(artists)
        update_followed_import_task(task_id, total_followed=total_followed)

        if max_artists > 0:
            artists = artists[:max_artists]

        if selected_artist_ids:
            artists = [item for item in artists if int(item.get("artist_id", 0)) in selected_artist_ids]
            if not artists:
                raise ValueError("选中的艺人不在当前关注列表中")

        processed_artists = len(artists)
        update_followed_import_task(
            task_id,
            processed_artists=processed_artists,
            progress=20,
            message=f"已确认导入范围，共 {processed_artists} 位艺人",
        )

        snapshot_path = save_followed_artists_snapshot(output_dir, artists)
        txt_path = save_followed_artists_text(output_dir, artists)
        update_followed_import_task(task_id, progress=30, message="已保存关注艺人快照")

        def import_progress_callback(
            current: int,
            total: int,
            stage: str,
            artist_id: int,
            artist_name: str,
        ) -> None:
            stage_ratio_map = {
                "catalog_fetch": 0.20,
                "catalog_cache": 0.45,
                "lyrics_fetch": 0.70,
                "lyrics_cache": 0.90,
                "artist_done": 1.00,
            }
            stage_ratio = stage_ratio_map.get(stage, 0.50)
            total_safe = max(1, int(total))
            progress_ratio = ((max(1, int(current)) - 1) + stage_ratio) / total_safe
            progress = 30 + int(progress_ratio * 65)
            progress = max(30, min(95, progress))

            stage_label_map = {
                "catalog_fetch": "抓取歌曲目录",
                "catalog_cache": "命中歌曲缓存",
                "lyrics_fetch": "抓取歌词",
                "lyrics_cache": "命中歌词缓存",
                "artist_done": "处理完成",
            }
            stage_label = stage_label_map.get(stage, "处理中")
            display_name = artist_name or f"歌手{artist_id}"
            update_followed_import_task(
                task_id,
                progress=progress,
                message=f"[{current}/{total}] {display_name}: {stage_label}",
            )

        save_artist_catalogs(
            output_dir=output_dir,
            artists=artists,
            cookie=cookie,
            page_size=max(1, page_size),
            sleep_seconds=max(0.0, sleep_seconds),
            timeout=max(1, timeout),
            refresh_existing=refresh_existing,
            fetch_lyrics=fetch_lyrics,
            lyric_sleep=max(0.0, lyric_sleep),
            lyric_timeout=max(1, lyric_timeout),
            progress_callback=import_progress_callback,
        )

        update_followed_import_task(
            task_id,
            status="completed",
            progress=100,
            message="导入完成",
            output_dir=output_dir,
            snapshot_json=snapshot_path,
            snapshot_txt=txt_path,
            fetch_lyrics=fetch_lyrics,
            refresh_existing=refresh_existing,
        )
    except Exception as e:
        update_followed_import_task(
            task_id,
            status="failed",
            progress=100,
            error=str(e),
            message=f"导入失败: {e}",
        )


def create_followed_import_task(params: Dict[str, Any]) -> Dict[str, Any]:
    task_id = uuid.uuid4().hex
    now = time.time()
    task = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0,
        "message": "任务已创建",
        "error": "",
        "created_at": now,
        "updated_at": now,
        "total_followed": 0,
        "processed_artists": 0,
    }

    with FOLLOWED_IMPORT_TASKS_LOCK:
        FOLLOWED_IMPORT_TASKS[task_id] = task

    thread = threading.Thread(
        target=followed_import_worker,
        args=(task_id, params),
        daemon=True,
    )
    thread.start()
    return task


def extract_clean_song_texts(lyrics_json_path: str) -> List[str]:
    data = read_json(lyrics_json_path)
    items = data.get("lyrics", [])
    if not isinstance(items, list):
        return []

    song_texts: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        lyric = str(item.get("lyric") or "")
        if not lyric.strip():
            continue

        cleaned_lines: List[str] = []
        for raw_line in lyric.splitlines():
            clean = lw.clean_line(raw_line)
            if clean:
                cleaned_lines.append(clean)

        if cleaned_lines:
            song_texts.append("\n".join(cleaned_lines))

    return song_texts


def build_word_frequency_and_text(
    lyrics_json_path: str,
    keep_single_char: bool = True,
    count_by_song: bool = False,
    filter_english_words: bool = True,
) -> Tuple[Counter, str]:
    song_texts = extract_clean_song_texts(lyrics_json_path)
    if not song_texts:
        return Counter(), ""

    text = "\n".join(song_texts)
    stopwords = lw.load_stopwords("")
    single_char_min_freq = 10 if keep_single_char else 0

    if not count_by_song:
        freq = lw.tokenize_and_count(
            text,
            stopwords,
            single_char_min_freq=single_char_min_freq,
            filter_english_words=filter_english_words,
        )
        return freq, text

    song_counter: Counter = Counter()
    # 按歌曲计数时，每首歌内的单字阈值应放宽，否则单字几乎不会被纳入。
    song_level_single_char_min_freq = 1 if keep_single_char else 0
    for song_text in song_texts:
        song_freq = lw.tokenize_and_count(
            song_text,
            stopwords,
            single_char_min_freq=song_level_single_char_min_freq,
            filter_english_words=filter_english_words,
        )
        for word in song_freq.keys():
            song_counter[word] += 1

    return song_counter, text


def count_wordcloud_songs(lyrics_json_path: str) -> int:
    return len(extract_clean_song_texts(lyrics_json_path))


def ensure_wordcloud_assets(
    artist_id: int,
    lyrics_json_path: str,
    keep_single_char: bool = True,
    count_by_song: bool = False,
    filter_english_words: bool = True,
    force_refresh: bool = False,
) -> Dict[str, str]:
    paths = artist_file_paths(
        artist_id=artist_id,
        keep_single_char=keep_single_char,
        count_by_song=count_by_song,
        filter_english_words=filter_english_words,
    )

    if (
        (not force_refresh)
        and os.path.exists(paths["wordcloud_image"])
        and os.path.exists(paths["top_words"])
    ):
        return {
            "image": paths["wordcloud_image"],
            "top_words": paths["top_words"],
            "cleaned": paths["cleaned"],
        }

    freq, text = build_word_frequency_and_text(
        lyrics_json_path=lyrics_json_path,
        keep_single_char=keep_single_char,
        count_by_song=count_by_song,
        filter_english_words=filter_english_words,
    )

    with open(paths["cleaned"], "w", encoding="utf-8") as f:
        f.write(text)

    filtered_freq = Counter({k: v for k, v in freq.items() if v >= 2})

    if not filtered_freq:
        raise ValueError("清洗后没有可用于词云的词，请尝试其他歌手")

    lw.save_top_words(filtered_freq, paths["top_words"], topn=200)
    font_path = lw.pick_font_path("")
    lw.generate_wordcloud(filtered_freq, paths["wordcloud_image"], font_path)

    return {
        "image": paths["wordcloud_image"],
        "top_words": paths["top_words"],
        "cleaned": paths["cleaned"],
    }


def build_wordcloud_in_memory(
    lyrics_json_path: str,
    keep_single_char: bool = True,
    count_by_song: bool = False,
    filter_english_words: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    freq, _ = build_word_frequency_and_text(
        lyrics_json_path=lyrics_json_path,
        keep_single_char=keep_single_char,
        count_by_song=count_by_song,
        filter_english_words=filter_english_words,
    )
    filtered_freq = Counter({k: v for k, v in freq.items() if v >= 2})

    if not filtered_freq:
        raise ValueError("清洗后没有可用于词云的词，请尝试其他歌手")

    try:
        from wordcloud import WordCloud
    except ImportError as e:
        raise ImportError("缺少依赖 wordcloud，请先安装：pip install wordcloud") from e

    font_path = lw.pick_font_path("")
    wc = WordCloud(
        width=1200,
        height=700,
        background_color="white",
        font_path=font_path,
        max_words=300,
        collocations=False,
    )
    wc.generate_from_frequencies(filtered_freq)

    image = wc.to_image()
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    image_data_url = f"data:image/png;base64,{encoded}"

    top_words = []
    for idx, (word, count) in enumerate(filtered_freq.most_common(200), start=1):
        top_words.append({"rank": idx, "word": word, "count": int(count)})

    return image_data_url, top_words


def parse_top_words(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return items

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            parts = line.strip().split("\t")
            if len(parts) != 2:
                continue
            word, count = parts
            try:
                num = int(count)
            except ValueError:
                continue
            items.append({"rank": idx, "word": word, "count": num})

    return items


def list_local_artist_ids() -> List[int]:
    artist_ids = set()
    if not os.path.exists(OUTPUT_DIR):
        return []

    prefix = "lyrics_artist_"
    suffix = ".json"
    for filename in os.listdir(OUTPUT_DIR):
        if not (filename.startswith(prefix) and filename.endswith(suffix)):
            continue
        raw_id = filename[len(prefix):-len(suffix)]
        if raw_id.isdigit():
            artist_ids.add(int(raw_id))

    return sorted(artist_ids)


def get_local_artist_name(artist_id: int) -> str:
    artist_path = artist_file_paths(artist_id)["artist"]
    default_name = f"歌手 {artist_id}"
    if not os.path.exists(artist_path):
        return default_name

    try:
        data = read_json(artist_path)
    except Exception:
        return default_name

    # 尽量从缓存文件中提取可读歌手名，缺失时回退为 ID 名称。
    for key in ("resolved_artist_name", "artist_name", "name"):
        value = str(data.get(key) or "").strip()
        if value:
            return value

    songs = data.get("songs", [])
    if isinstance(songs, list) and songs:
        first = songs[0]
        if isinstance(first, dict):
            for key in ("artist_name", "singer", "artist"):
                value = str(first.get(key) or "").strip()
                if value:
                    return value

    return default_name


def collect_artist_related_files(artist_id: int) -> List[str]:
    files: List[str] = []
    fixed_paths = [
        artist_file_paths(artist_id)["artist"],
        artist_file_paths(artist_id)["lyrics"],
        artist_file_paths(artist_id)["song_list"],
    ]
    for path in fixed_paths:
        if os.path.exists(path):
            files.append(path)

    artist_token = f"artist_{artist_id}"
    if os.path.exists(STATIC_GENERATED_DIR):
        for filename in os.listdir(STATIC_GENERATED_DIR):
            if artist_token in filename:
                full_path = os.path.join(STATIC_GENERATED_DIR, filename)
                if os.path.isfile(full_path):
                    files.append(full_path)

    # 去重并保持稳定顺序。
    unique_files = []
    seen = set()
    for path in files:
        if path in seen:
            continue
        seen.add(path)
        unique_files.append(path)
    return unique_files


def delete_local_artist_data(artist_id: int) -> int:
    target_files = collect_artist_related_files(artist_id)
    removed_count = 0
    for path in target_files:
        try:
            os.remove(path)
            removed_count += 1
        except FileNotFoundError:
            continue
    return removed_count


def build_local_artist_list() -> List[Dict[str, Any]]:
    artists: List[Dict[str, Any]] = []
    for artist_id in list_local_artist_ids():
        file_count = len(collect_artist_related_files(artist_id))
        artists.append(
            {
                "artist_id": artist_id,
                "artist_name": get_local_artist_name(artist_id),
                "file_count": file_count,
            }
        )
    return artists


def build_artist_word_ratio(
    artist_id: int,
    word: str,
    keep_single_char: bool,
    count_by_song: bool,
    filter_english_words: bool,
) -> Optional[Dict[str, Any]]:
    paths = artist_file_paths(
        artist_id=artist_id,
        keep_single_char=keep_single_char,
        count_by_song=count_by_song,
        filter_english_words=filter_english_words,
    )
    lyrics_path = paths["lyrics"]
    if not has_valid_lyrics_data(lyrics_path):
        return None

    freq, _ = build_word_frequency_and_text(
        lyrics_json_path=lyrics_path,
        keep_single_char=keep_single_char,
        count_by_song=count_by_song,
        filter_english_words=filter_english_words,
    )

    word_count = int(freq.get(word, 0))
    if count_by_song:
        total_base = count_wordcloud_songs(lyrics_path)
    else:
        total_base = sum(int(v) for v in freq.values())

    ratio = (word_count / total_base) if total_base > 0 else 0.0
    return {
        "artist_id": artist_id,
        "artist_name": get_local_artist_name(artist_id),
        "word": word,
        "word_count": word_count,
        "total_base": total_base,
        "ratio": ratio,
    }


def build_artist_word_ratio_from_top_words(
    artist_id: int,
    word: str,
    keep_single_char: bool,
    count_by_song: bool,
    filter_english_words: bool,
) -> Optional[Dict[str, Any]]:
    paths = artist_file_paths(
        artist_id=artist_id,
        keep_single_char=keep_single_char,
        count_by_song=count_by_song,
        filter_english_words=filter_english_words,
    )

    top_words_path = paths["top_words"]
    if not os.path.exists(top_words_path):
        return None

    items = parse_top_words(top_words_path)
    if not items:
        return None

    word_count = 0
    total_base = 0
    for item in items:
        count = int(item.get("count", 0))
        total_base += count
        if str(item.get("word") or "") == word:
            word_count = count

    # 快速模式下按歌曲覆盖数统计：分母优先使用歌词去重后的歌曲数。
    if count_by_song:
        lyrics_path = paths["lyrics"]
        total_base = 0
        if os.path.exists(lyrics_path):
            try:
                lyrics_data = read_json(lyrics_path)
                total_base = int(lyrics_data.get("lyric_deduped_count") or 0)
            except Exception:
                total_base = 0

    ratio = (word_count / total_base) if total_base > 0 else 0.0
    return {
        "artist_id": artist_id,
        "artist_name": get_local_artist_name(artist_id),
        "word": word,
        "word_count": int(word_count),
        "total_base": int(total_base),
        "ratio": ratio,
        "ratio_basis": "top_words",
    }


def build_word_ratio_rank_result(
    word: str,
    keep_single_char: bool,
    count_by_song: bool,
    filter_english_words: bool,
    quick_mode: bool,
    current_artist_id_raw: Any,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    local_artist_ids = list_local_artist_ids()
    if not local_artist_ids:
        raise ValueError("本地暂无可用歌手词云数据")

    total_artists = len(local_artist_ids)
    rows: List[Dict[str, Any]] = []

    if progress_callback is not None:
        progress_callback(0, total_artists, "已获取本地歌手列表")

    for index, artist_id in enumerate(local_artist_ids, start=1):
        if quick_mode:
            item = build_artist_word_ratio_from_top_words(
                artist_id=artist_id,
                word=word,
                keep_single_char=keep_single_char,
                count_by_song=count_by_song,
                filter_english_words=filter_english_words,
            )
        else:
            item = build_artist_word_ratio(
                artist_id=artist_id,
                word=word,
                keep_single_char=keep_single_char,
                count_by_song=count_by_song,
                filter_english_words=filter_english_words,
            )
        if item is not None:
            rows.append(item)

        if progress_callback is not None:
            progress_callback(index, total_artists, f"正在计算 {index}/{total_artists}")

    if not rows:
        raise ValueError("本地暂无有效歌词数据")

    # 仅保留出现过该词的歌手，避免在榜单中显示 0 次命中项。
    rows = [row for row in rows if int(row.get("word_count", 0)) > 0]
    matched_count = len(rows)

    current_artist_id = None
    if current_artist_id_raw is not None:
        try:
            current_artist_id = int(current_artist_id_raw)
        except (TypeError, ValueError):
            current_artist_id = None

    if matched_count == 0:
        return {
            "word": word,
            "keep_single_char": keep_single_char,
            "count_by_song": count_by_song,
            "filter_english_words": filter_english_words,
            "quick_mode": quick_mode,
            "total_artists": total_artists,
            "matched_artists": 0,
            "current_artist_id": current_artist_id,
            "current_artist_rank": None,
            "items": [],
        }

    rows.sort(
        key=lambda x: (
            float(x.get("ratio", 0.0)),
            int(x.get("word_count", 0)),
            -int(x.get("artist_id", 0)),
        ),
        reverse=True,
    )

    current_rank = None
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
        if current_artist_id is not None and int(row["artist_id"]) == current_artist_id:
            current_rank = idx

    return {
        "word": word,
        "keep_single_char": keep_single_char,
        "count_by_song": count_by_song,
        "filter_english_words": filter_english_words,
        "quick_mode": quick_mode,
        "total_artists": total_artists,
        "matched_artists": matched_count,
        "current_artist_id": current_artist_id,
        "current_artist_rank": current_rank,
        "items": rows,
    }


def update_word_ratio_task(task_id: str, **updates: Any) -> None:
    with WORD_RATIO_TASKS_LOCK:
        task = WORD_RATIO_TASKS.get(task_id)
        if not task:
            return
        task.update(updates)
        task["updated_at"] = time.time()


def word_ratio_rank_worker(
    task_id: str,
    word: str,
    keep_single_char: bool,
    count_by_song: bool,
    filter_english_words: bool,
    quick_mode: bool,
    current_artist_id_raw: Any,
) -> None:
    try:
        update_word_ratio_task(task_id, status="running", progress=1, message="任务开始")

        def on_progress(done: int, total: int, message: str) -> None:
            if total <= 0:
                progress = 1
            else:
                progress = min(99, int((done / total) * 99))
            update_word_ratio_task(task_id, progress=progress, message=message)

        result = build_word_ratio_rank_result(
            word=word,
            keep_single_char=keep_single_char,
            count_by_song=count_by_song,
            filter_english_words=filter_english_words,
            quick_mode=quick_mode,
            current_artist_id_raw=current_artist_id_raw,
            progress_callback=on_progress,
        )

        update_word_ratio_task(
            task_id,
            status="completed",
            progress=100,
            message="计算完成",
            result=result,
        )
    except Exception as e:
        update_word_ratio_task(
            task_id,
            status="failed",
            progress=100,
            message=f"计算失败: {e}",
            error=str(e),
        )


def create_word_ratio_task(
    word: str,
    keep_single_char: bool,
    count_by_song: bool,
    filter_english_words: bool,
    quick_mode: bool,
    current_artist_id_raw: Any,
) -> Dict[str, Any]:
    task_id = uuid.uuid4().hex
    now = time.time()
    task = {
        "task_id": task_id,
        "status": "pending",
        "progress": 0,
        "message": "任务已创建",
        "error": "",
        "result": None,
        "created_at": now,
        "updated_at": now,
    }

    with WORD_RATIO_TASKS_LOCK:
        WORD_RATIO_TASKS[task_id] = task

    thread = threading.Thread(
        target=word_ratio_rank_worker,
        args=(
            task_id,
            word,
            keep_single_char,
            count_by_song,
            filter_english_words,
            quick_mode,
            current_artist_id_raw,
        ),
        daemon=True,
    )
    thread.start()
    return task


def paginate(items: List[Any], page: int, page_size: int) -> Dict[str, Any]:
    total = len(items)
    if page_size <= 0:
        page_size = 20
    if page <= 0:
        page = 1

    start = (page - 1) * page_size
    end = start + page_size
    sliced = items[start:end]

    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": (total + page_size - 1) // page_size,
        "items": sliced,
    }


def build_top_words_for_mode(
    lyrics_json_path: str,
    keep_single_char: bool,
    count_by_song: bool,
    filter_english_words: bool,
    min_freq: int = 2,
    topn: int = 80,
) -> List[Dict[str, Any]]:
    freq, _ = build_word_frequency_and_text(
        lyrics_json_path=lyrics_json_path,
        keep_single_char=keep_single_char,
        count_by_song=count_by_song,
        filter_english_words=filter_english_words,
    )
    filtered_freq = Counter({k: v for k, v in freq.items() if v >= min_freq})

    items: List[Dict[str, Any]] = []
    for idx, (word, count) in enumerate(filtered_freq.most_common(topn), start=1):
        items.append({"rank": idx, "word": word, "count": int(count)})
    return items


def analyze_sentiment_with_ai(
    artist_name: str,
    top_words: List[Dict[str, Any]],
    count_by_song: bool,
) -> Tuple[str, str]:
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    api_key = openrouter_api_key or openai_api_key
    if not api_key:
        raise ValueError(
            "未配置 API Key，请先设置 OPENROUTER_API_KEY 或 OPENAI_API_KEY"
        )

    default_base_url = "https://openrouter.ai/api/v1" if openrouter_api_key else "https://api.openai.com/v1"
    default_model = "deepseek/deepseek-v3.2" if openrouter_api_key else "gpt-4o-mini"
    base_url = os.getenv("OPENAI_BASE_URL", default_base_url).strip().rstrip("/")
    model = os.getenv("OPENAI_MODEL", default_model).strip()
    endpoint = f"{base_url}/chat/completions"

    words_text = "\n".join(
        [f"{item['rank']}. {item['word']} ({item['count']})" for item in top_words]
    )
    count_mode_text = "按歌曲覆盖数统计" if count_by_song else "按词频统计"

    system_prompt = (
        "你是专业中文歌词情感分析师。"
        "请基于给定高频词判断整体情感倾向，不要虚构未提供的数据。"
        "输出要简洁、结构化。"
    )
    user_prompt = (
        f"歌手: {artist_name or '未知歌手'}\n"
        f"统计口径: {count_mode_text}\n"
        "以下是高频词:\n"
        f"{words_text}\n\n"
        "请输出:\n"
        "1) 情感主倾向(如偏积极/偏消极/复杂混合)\n"
        "2) 3条核心依据（引用词）\n"
        "3) 风格标签（3-5个）\n"
        "4) 一句总结（20字内）"
    )

    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # OpenRouter 推荐头，便于请求来源识别；未配置则跳过。
    referer = os.getenv("OPENROUTER_SITE_URL", "").strip()
    title = os.getenv("OPENROUTER_APP_NAME", "lyricExpert").strip()
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title

    resp = requests.post(endpoint, headers=headers, json=payload, timeout=45)
    resp.raise_for_status()
    data = resp.json()
    content = (
        (((data.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
    ).strip()
    if not content:
        raise ValueError("AI 返回为空，请稍后重试")
    return content, model


@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/go-artist", methods=["POST"])
def go_artist() -> Any:
    artist_name = (request.form.get("artist_name") or "").strip()
    if not artist_name:
        return redirect(url_for("index"))

    artist_id, resolved_name = resolve_artist(artist_name)
    return redirect(url_for("artist_page", artist_id=artist_id, artist_name=resolved_name))


@app.route("/artist/<int:artist_id>")
def artist_page(artist_id: int) -> str:
    artist_name = (request.args.get("artist_name") or "").strip()
    return render_template("artist.html", artist_id=artist_id, artist_name=artist_name)


@app.route("/api/artist/resolve", methods=["POST"])
def api_artist_resolve() -> Any:
    payload = request.get_json(silent=True) or {}
    artist_name = (payload.get("artist_name") or "").strip()
    if not artist_name:
        return jsonify({"ok": False, "error": "artist_name 不能为空"}), 400

    try:
        artist_id, resolved_name = resolve_artist(artist_name)
        return jsonify({"ok": True, "artist_id": artist_id, "artist_name": resolved_name})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/local-artists", methods=["GET"])
def api_local_artists() -> Any:
    try:
        artists = build_local_artist_list()
        return jsonify(
            {
                "ok": True,
                "total": len(artists),
                "artists": artists,
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/local-artists/delete", methods=["POST"])
def api_local_artists_delete() -> Any:
    payload = request.get_json(silent=True) or {}
    artist_ids_raw = payload.get("artist_ids")
    if not isinstance(artist_ids_raw, list) or not artist_ids_raw:
        return jsonify({"ok": False, "error": "artist_ids 不能为空"}), 400

    artist_ids: List[int] = []
    for item in artist_ids_raw:
        try:
            artist_id = int(item)
        except (TypeError, ValueError):
            continue
        if artist_id > 0:
            artist_ids.append(artist_id)

    if not artist_ids:
        return jsonify({"ok": False, "error": "artist_ids 无有效值"}), 400

    removed_files = 0
    removed_artists = 0
    for artist_id in sorted(set(artist_ids)):
        count = delete_local_artist_data(artist_id)
        if count > 0:
            removed_artists += 1
            removed_files += count

    artists = build_local_artist_list()
    return jsonify(
        {
            "ok": True,
            "removed_artists": removed_artists,
            "removed_files": removed_files,
            "total": len(artists),
            "artists": artists,
        }
    )


@app.route("/api/artist/<int:artist_id>/wordcloud", methods=["GET"])
def api_wordcloud(artist_id: int) -> Any:
    page = int(request.args.get("page", 1))
    page_size = int(request.args.get("page_size", 20))
    artist_name = (request.args.get("artist_name") or "").strip()
    skip_prepare = request.args.get("skip_prepare", "0") == "1"
    keep_single_char = request.args.get("keep_single_char", "0") == "1"
    count_by_song = request.args.get("count_by_song", "0") == "1"
    filter_english_words = request.args.get("filter_english_words", "1") == "1"
    force_refresh = request.args.get("refresh", "0") == "1"

    try:
        if skip_prepare:
            paths = artist_file_paths(
                artist_id=artist_id,
                keep_single_char=keep_single_char,
                count_by_song=count_by_song,
                filter_english_words=filter_english_words,
            )
            if not has_valid_lyrics_data(paths["lyrics"]):
                return jsonify({"ok": False, "error": "数据尚未准备完成，请先执行准备任务"}), 409
        else:
            paths = ensure_artist_data(artist_id=artist_id, artist_name=artist_name)
        use_memory_mode = False
        image_url = ""

        try:
            assets = ensure_wordcloud_assets(
                artist_id=artist_id,
                lyrics_json_path=paths["lyrics"],
                keep_single_char=keep_single_char,
                count_by_song=count_by_song,
                filter_english_words=filter_english_words,
                force_refresh=force_refresh,
            )
            top_words = parse_top_words(assets["top_words"])
            image_name = os.path.basename(assets["image"])
            image_url = url_for("static", filename=f"generated/{image_name}")
        except OSError as e:
            if getattr(e, "errno", None) != 28:
                raise
            image_url, top_words = build_wordcloud_in_memory(
                paths["lyrics"],
                keep_single_char=keep_single_char,
                count_by_song=count_by_song,
                filter_english_words=filter_english_words,
            )
            use_memory_mode = True

        paged = paginate(top_words, page=page, page_size=page_size)

        song_total = 0
        if count_by_song:
            song_total = count_wordcloud_songs(paths["lyrics"])
            for item in paged["items"]:
                count = int(item.get("count", 0))
                ratio = (count / song_total) if song_total > 0 else 0.0
                item["song_ratio"] = ratio
        else:
            for item in paged["items"]:
                item["song_ratio"] = None

        return jsonify(
            {
                "ok": True,
                "artist_id": artist_id,
                "artist_name": artist_name,
                "keep_single_char": keep_single_char,
                "count_by_song": count_by_song,
                "filter_english_words": filter_english_words,
                "song_total": song_total,
                "force_refresh": force_refresh,
                "image_url": image_url,
                "memory_mode": use_memory_mode,
                "top_words": paged,
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/artist/<int:artist_id>/prepare", methods=["POST"])
def api_prepare_artist(artist_id: int) -> Any:
    payload = request.get_json(silent=True) or {}
    artist_name = str(payload.get("artist_name") or "").strip()

    task = create_prepare_task(artist_id=artist_id, artist_name=artist_name)
    return jsonify(
        {
            "ok": True,
            "task_id": task["task_id"],
            "artist_id": artist_id,
            "artist_name": artist_name,
        }
    )


@app.route("/api/progress/<task_id>", methods=["GET"])
def api_task_progress(task_id: str) -> Any:
    with PREPARE_TASKS_LOCK:
        task = PREPARE_TASKS.get(task_id)

    if not task:
        return jsonify({"ok": False, "error": "任务不存在"}), 404

    return jsonify(
        {
            "ok": True,
            "task_id": task_id,
            "artist_id": task.get("artist_id"),
            "artist_name": task.get("artist_name"),
            "status": task.get("status"),
            "progress": int(task.get("progress", 0)),
            "message": task.get("message") or "",
            "error": task.get("error") or "",
        }
    )


@app.route("/api/search", methods=["POST"])
def api_search() -> Any:
    payload = request.get_json(silent=True) or {}
    keyword = (payload.get("keyword") or "").strip()
    artist_name = (payload.get("artist_name") or "").strip()
    artist_id_raw = payload.get("artist_id")

    page = int(payload.get("page", 1))
    page_size = int(payload.get("page_size", 10))

    if not keyword:
        return jsonify({"ok": False, "error": "keyword 不能为空"}), 400

    try:
        if artist_name:
            artist_id, resolved_name = resolve_artist(artist_name)
        elif artist_id_raw is not None:
            artist_id = int(artist_id_raw)
            resolved_name = ""
        else:
            return jsonify({"ok": False, "error": "artist_name 或 artist_id 必填其一"}), 400

        paths = ensure_artist_data(artist_id=artist_id, artist_name=resolved_name)
        songs = ls.load_lyrics(paths["lyrics"])
        matched = ls.search_keyword(
            songs=songs,
            keyword=keyword,
            clean_timestamp=True,
            show_lines=True,
        )
        paged = paginate(matched, page=page, page_size=page_size)
        total_count = sum(int(item.get("count", 0)) for item in matched)

        return jsonify(
            {
                "ok": True,
                "artist_id": artist_id,
                "artist_name": resolved_name or artist_name,
                "keyword": keyword,
                "hit_song_count": len(matched),
                "total_count": total_count,
                "results": paged,
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/word-ratio-rank", methods=["POST"])
def api_word_ratio_rank() -> Any:
    payload = request.get_json(silent=True) or {}
    word = str(payload.get("word") or "").strip()
    keep_single_char = bool(payload.get("keep_single_char", False))
    count_by_song = bool(payload.get("count_by_song", False))
    filter_english_words = bool(payload.get("filter_english_words", True))
    quick_mode = bool(payload.get("quick_mode", False))
    current_artist_id_raw = payload.get("current_artist_id")

    if not word:
        return jsonify({"ok": False, "error": "word 不能为空"}), 400

    try:
        result = build_word_ratio_rank_result(
            word=word,
            keep_single_char=keep_single_char,
            count_by_song=count_by_song,
            filter_english_words=filter_english_words,
            quick_mode=quick_mode,
            current_artist_id_raw=current_artist_id_raw,
        )
        return jsonify({"ok": True, **result})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/word-ratio-rank/start", methods=["POST"])
def api_word_ratio_rank_start() -> Any:
    payload = request.get_json(silent=True) or {}
    word = str(payload.get("word") or "").strip()
    keep_single_char = bool(payload.get("keep_single_char", False))
    count_by_song = bool(payload.get("count_by_song", False))
    filter_english_words = bool(payload.get("filter_english_words", True))
    quick_mode = bool(payload.get("quick_mode", False))
    current_artist_id_raw = payload.get("current_artist_id")

    if not word:
        return jsonify({"ok": False, "error": "word 不能为空"}), 400

    task = create_word_ratio_task(
        word=word,
        keep_single_char=keep_single_char,
        count_by_song=count_by_song,
        filter_english_words=filter_english_words,
        quick_mode=quick_mode,
        current_artist_id_raw=current_artist_id_raw,
    )
    return jsonify({"ok": True, "task_id": task["task_id"]})


@app.route("/api/word-ratio-rank/progress/<task_id>", methods=["GET"])
def api_word_ratio_rank_progress(task_id: str) -> Any:
    with WORD_RATIO_TASKS_LOCK:
        task = WORD_RATIO_TASKS.get(task_id)

    if not task:
        return jsonify({"ok": False, "error": "任务不存在"}), 404

    response = {
        "ok": True,
        "task_id": task_id,
        "status": task.get("status"),
        "progress": int(task.get("progress", 0)),
        "message": task.get("message") or "",
        "error": task.get("error") or "",
    }

    if task.get("status") == "completed" and task.get("result") is not None:
        response["result"] = task.get("result")

    return jsonify(response)


@app.route("/api/artist/<int:artist_id>/ai-sentiment", methods=["POST"])
def api_ai_sentiment(artist_id: int) -> Any:
    payload = request.get_json(silent=True) or {}
    artist_name = (payload.get("artist_name") or "").strip()
    keep_single_char = bool(payload.get("keep_single_char", False))
    count_by_song = bool(payload.get("count_by_song", False))
    filter_english_words = bool(payload.get("filter_english_words", True))
    topn = int(payload.get("topn", 80))

    try:
        paths = ensure_artist_data(artist_id=artist_id, artist_name=artist_name)
        top_words = build_top_words_for_mode(
            lyrics_json_path=paths["lyrics"],
            keep_single_char=keep_single_char,
            count_by_song=count_by_song,
            filter_english_words=filter_english_words,
            min_freq=2,
            topn=max(20, min(200, topn)),
        )
        if not top_words:
            return jsonify({"ok": False, "error": "当前配置下无可分析词语"}), 400

        analysis, model = analyze_sentiment_with_ai(
            artist_name=artist_name,
            top_words=top_words,
            count_by_song=count_by_song,
        )

        return jsonify(
            {
                "ok": True,
                "artist_id": artist_id,
                "artist_name": artist_name,
                "keep_single_char": keep_single_char,
                "count_by_song": count_by_song,
                "filter_english_words": filter_english_words,
                "model": model,
                "analysis": analysis,
            }
        )
    except requests.HTTPError as e:
        detail = ""
        try:
            detail = e.response.text
        except Exception:
            detail = str(e)
        return jsonify({"ok": False, "error": f"AI 接口调用失败: {detail}"}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/followed-artists/list", methods=["POST"])
def api_followed_artists_list() -> Any:
    payload = request.get_json(silent=True) or {}

    cookie = str(payload.get("cookie") or os.getenv("NETEASE_COOKIE", "")).strip()
    if not cookie:
        return jsonify({"ok": False, "error": "cookie 不能为空，可通过 body.cookie 或环境变量 NETEASE_COOKIE 提供"}), 400

    max_artists = int(payload.get("max_artists", 0) or 0)
    page_size = int(payload.get("page_size", 100) or 100)
    sleep_seconds = float(payload.get("sleep", 0.25) or 0.25)
    timeout = int(payload.get("timeout", 10) or 10)

    if page_size <= 0:
        page_size = 100
    if max_artists < 0:
        max_artists = 0

    try:
        artists = fetch_followed_artists(
            cookie=cookie,
            page_size=page_size,
            sleep_seconds=max(0.0, sleep_seconds),
            timeout=max(1, timeout),
        )
        if not artists:
            return jsonify({"ok": False, "error": "未读取到关注艺人，请检查 Cookie 是否有效"}), 400

        total_followed = len(artists)
        if max_artists > 0:
            artists = artists[:max_artists]

        return jsonify(
            {
                "ok": True,
                "total_followed": total_followed,
                "visible_artists": len(artists),
                "artists": artists,
            }
        )
    except requests.HTTPError as e:
        detail = ""
        try:
            detail = e.response.text
        except Exception:
            detail = str(e)
        return jsonify({"ok": False, "error": f"网易云接口调用失败: {detail}"}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/followed-artists/import", methods=["POST"])
def api_import_followed_artists() -> Any:
    payload = request.get_json(silent=True) or {}

    cookie = str(payload.get("cookie") or os.getenv("NETEASE_COOKIE", "")).strip()
    if not cookie:
        return jsonify({"ok": False, "error": "cookie 不能为空，可通过 body.cookie 或环境变量 NETEASE_COOKIE 提供"}), 400

    output_dir = str(payload.get("output_dir") or OUTPUT_DIR).strip() or OUTPUT_DIR
    max_artists = int(payload.get("max_artists", 0) or 0)
    page_size = int(payload.get("page_size", 100) or 100)
    sleep_seconds = float(payload.get("sleep", 0.25) or 0.25)
    timeout = int(payload.get("timeout", 10) or 10)
    refresh_existing = bool(payload.get("refresh_existing", False))
    fetch_lyrics = bool(payload.get("fetch_lyrics", False))
    lyric_sleep = float(payload.get("lyric_sleep", 0.1) or 0.1)
    lyric_timeout = int(payload.get("lyric_timeout", 10) or 10)
    selected_artist_ids_raw = payload.get("selected_artist_ids")

    if page_size <= 0:
        page_size = 100
    if max_artists < 0:
        max_artists = 0

    try:
        selected_artist_ids = set()
        if isinstance(selected_artist_ids_raw, list):
            for item in selected_artist_ids_raw:
                try:
                    selected_artist_ids.add(int(item))
                except (TypeError, ValueError):
                    continue
        task = create_followed_import_task(
            {
                "cookie": cookie,
                "output_dir": output_dir,
                "max_artists": max_artists,
                "page_size": page_size,
                "sleep_seconds": max(0.0, sleep_seconds),
                "timeout": max(1, timeout),
                "refresh_existing": refresh_existing,
                "fetch_lyrics": fetch_lyrics,
                "lyric_sleep": max(0.0, lyric_sleep),
                "lyric_timeout": max(1, lyric_timeout),
                "selected_artist_ids": sorted(selected_artist_ids),
            }
        )

        return jsonify(
            {
                "ok": True,
                "task_id": task["task_id"],
                "status": task["status"],
                "progress": task["progress"],
                "message": task["message"],
            }
        )
    except requests.HTTPError as e:
        detail = ""
        try:
            detail = e.response.text
        except Exception:
            detail = str(e)
        return jsonify({"ok": False, "error": f"网易云接口调用失败: {detail}"}), 502
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/followed-artists/import-progress/<task_id>", methods=["GET"])
def api_followed_artists_import_progress(task_id: str) -> Any:
    with FOLLOWED_IMPORT_TASKS_LOCK:
        task = FOLLOWED_IMPORT_TASKS.get(task_id)

    if not task:
        return jsonify({"ok": False, "error": "任务不存在"}), 404

    return jsonify(
        {
            "ok": True,
            "task_id": task_id,
            "status": task.get("status"),
            "progress": int(task.get("progress", 0)),
            "message": task.get("message") or "",
            "error": task.get("error") or "",
            "total_followed": int(task.get("total_followed", 0) or 0),
            "processed_artists": int(task.get("processed_artists", 0) or 0),
            "output_dir": task.get("output_dir") or "",
            "snapshot_json": task.get("snapshot_json") or "",
            "snapshot_txt": task.get("snapshot_txt") or "",
            "fetch_lyrics": bool(task.get("fetch_lyrics", False)),
            "refresh_existing": bool(task.get("refresh_existing", False)),
        }
    )


if __name__ == "__main__":
    ensure_directories()
    app.run(host="127.0.0.1", port=5000, debug=True)
