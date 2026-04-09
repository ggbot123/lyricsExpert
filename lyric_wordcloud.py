import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List


TIMESTAMP_PATTERN = re.compile(r"\[\d{1,2}:\d{2}(?:\.\d{1,3})?\]")
SPEAKER_PREFIX_PATTERN = re.compile(r"^(?:男|女|合|旁白|RAP|Rap|rap)[：:]\s*")
CREDIT_LINE_PATTERN = re.compile(
    r"(?:^|\s)(?:"
    r"作词|作曲|编曲|制作人|监制|录音|混音|母带|和声|配唱|录音室|混音室|"
    r"制作助理|执行制作|发行|出品|专辑制作人|配唱制作人|"
    r"Producer|Recording|Mix|Mastering|Arrang|Vocal|Studio|Assistant|"
    r"Executive Producer|Publisher|ISRC|OP|General Manager|原唱|翻唱"
    r")(?:\s*[：:：\-]|\s*$)",
    flags=re.IGNORECASE,
)
PURE_NOISE_PATTERN = re.compile(r"^[\W_]+$")
BY_WORD_PATTERN = re.compile(r"\bby\b", flags=re.IGNORECASE)
ALLOWED_SINGLE_CHAR_POS_PREFIXES = ("n", "v", "a")

DEFAULT_STOPWORDS = {
    "我们",
    "你们",
    "他们",
    "一个",
    "没有",
    "不是",
    "可以",
    "不会",
    "还是",
    "什么",
    "自己",
    "真的",
    "这样",
    "那个",
    "这个",
    "一点",
    "时候",
    "已经",
    "因为",
    "如果",
    "怎么",
    "我要",
    "你要",
    "我的",
    "你的",
    "他的",
    "她的",
    "然后",
    "只是",
    "还有",
    "就是",
    "可是",
    "但是",
    "自己",
    "不是",
    "也",
    "很",
    "都",
    "在",
    "了",
    "着",
    "是",
    "我",
    "你",
    "他",
    "她",
    "它",
    "啊",
    "呀",
    "吧",
    "呢",
    "吗",
    "哦",
    "在",
    "从",
    "对",
    "向",
    "往",
    "朝",
    "于",
    "与",
    "和",
    "跟",
    "同",
    "给",
    "替",
    "帮",
    "为",
    "为了",
    "把",
    "被",
    "由",
    "按",
    "依",
    "照",
    "按着",
    "沿",
    "沿着",
    "根据",
    "通过",
    "对于",
    "关于",
    "除了",
    "比",
    "随着",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "from",
    "by",
    "with",
    "as",
    "into",
    "onto",
    "Production",
    "production",
    "assistant",
    "Assistant",
}


def load_lyrics(json_path: str) -> List[str]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("lyrics", [])
    if not isinstance(items, list):
        raise ValueError("输入 JSON 格式不正确：缺少 lyrics 列表")

    lyrics: List[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("lyric") or "").strip()
        if text:
            lyrics.append(text)

    if not lyrics:
        raise ValueError("没有读取到有效歌词内容")

    return lyrics


def clean_line(line: str) -> str:
    line = TIMESTAMP_PATTERN.sub("", line)
    line = SPEAKER_PREFIX_PATTERN.sub("", line).strip()
    if not line:
        return ""

    line = line.replace("\u3000", " ").strip()

    # 用户规则：带 by 或带冒号的行统一视为非歌词信息。
    if BY_WORD_PATTERN.search(line):
        return ""
    if ":" in line or "：" in line:
        return ""
    if "-" in line:
        return ""
    if "|" in line:
        return ""

    if CREDIT_LINE_PATTERN.search(line):
        return ""

    lower_line = line.lower()
    # 清洗常见英文制作署名行，如 "Production Assistant xxx"。
    if "production" in lower_line or "assistant" in lower_line:
        return ""

    if "翻唱" in line:
        return ""

    # 一些常见的句尾说明行，通常是制作信息或版权信息。
    if any(key in line for key in ("TW-", "@", "录音室", "Mastering", "Studio", "OP:", "ISRC")):
        if len(line) <= 60:
            return ""

    if PURE_NOISE_PATTERN.match(line):
        return ""

    return line


def preprocess_lyrics(lyrics: Iterable[str]) -> str:
    cleaned: List[str] = []
    for lyric in lyrics:
        for raw_line in lyric.splitlines():
            line = clean_line(raw_line)
            if line:
                cleaned.append(line)

    return "\n".join(cleaned)


def load_stopwords(extra_stopwords_path: str) -> set:
    stopwords = set(DEFAULT_STOPWORDS)
    if not extra_stopwords_path:
        return stopwords

    with open(extra_stopwords_path, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word and not word.startswith("#"):
                stopwords.add(word)

    return stopwords


def pick_font_path(user_font_path: str = "") -> str:
    if user_font_path:
        if not os.path.exists(user_font_path):
            raise FileNotFoundError(f"指定字体不存在: {user_font_path}")
        return user_font_path

    candidates = [
        r"C:\\Windows\\Fonts\\msyh.ttc",
        r"C:\\Windows\\Fonts\\simhei.ttf",
        r"C:\\Windows\\Fonts\\simsun.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError("未找到中文字体，请使用 --font-path 指定字体文件路径")


def tokenize_and_count(
    text: str,
    stopwords: set,
    single_char_min_freq: int,
    filter_english_words: bool = False,
) -> Counter:
    try:
        import jieba
        import jieba.posseg as pseg
    except ImportError as e:
        raise ImportError("缺少依赖 jieba，请先安装：pip install jieba") from e

    words = jieba.lcut(text)
    freq: Counter = Counter()
    for word in words:
        token = word.strip()
        if not token:
            continue
        if token in stopwords:
            continue
        if token.isdigit():
            continue

        if len(token) == 1:
            continue

        if not re.match(r"[A-Za-z0-9\u4e00-\u9fff]+$", token):
            continue
        if filter_english_words and re.fullmatch(r"[A-Za-z]+", token):
            continue
        freq[token] += 1

    # 更智能地保留少量单字：要求词性是名词/动词/形容词且频次达到阈值。
    if single_char_min_freq > 0:
        single_char_counter: Counter = Counter()
        for pair in pseg.cut(text):
            token = pair.word.strip()
            pos = (pair.flag or "").strip().lower()
            if len(token) != 1:
                continue
            if token in stopwords:
                continue
            if token.isdigit():
                continue
            if not re.match(r"[A-Za-z0-9\u4e00-\u9fff]$", token):
                continue
            if filter_english_words and re.fullmatch(r"[A-Za-z]", token):
                continue
            if not pos.startswith(ALLOWED_SINGLE_CHAR_POS_PREFIXES):
                continue
            single_char_counter[token] += 1

        for token, count in single_char_counter.items():
            if count >= single_char_min_freq:
                freq[token] += count

    return freq


def save_top_words(counter: Counter, output_path: str, topn: int) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for word, count in counter.most_common(topn):
            f.write(f"{word}\t{count}\n")


def generate_wordcloud(counter: Counter, output_image_path: str, font_path: str) -> None:
    try:
        from wordcloud import WordCloud
    except ImportError as e:
        raise ImportError("缺少依赖 wordcloud，请先安装：pip install wordcloud") from e

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("缺少依赖 matplotlib，请先安装：pip install matplotlib") from e

    wc = WordCloud(
        width=1600,
        height=900,
        background_color="white",
        font_path=font_path,
        max_words=300,
        collocations=False,
    )
    wc.generate_from_frequencies(counter)

    os.makedirs(os.path.dirname(output_image_path) or ".", exist_ok=True)
    wc.to_file(output_image_path)

    plt.figure(figsize=(12, 7))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.close()


def build_default_output_paths(input_json: str) -> Dict[str, str]:
    base = os.path.splitext(os.path.basename(input_json))[0]
    out_dir = "lyrics_output"
    return {
        "image": os.path.join(out_dir, f"wordcloud_{base}.png"),
        "top_words": os.path.join(out_dir, f"top_words_{base}.txt"),
        "cleaned": os.path.join(out_dir, f"cleaned_{base}.txt"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="对歌词 JSON 进行预处理并生成中文词云")
    parser.add_argument("target_json", type=str, help="目标歌词 JSON 路径")
    parser.add_argument("--image", type=str, default="", help="词云图片输出路径")
    parser.add_argument("--top-words", type=str, default="", help="高频词输出路径")
    parser.add_argument("--cleaned-text", type=str, default="", help="清洗后文本输出路径")
    parser.add_argument("--topn", type=int, default=200, help="输出高频词数量")
    parser.add_argument("--min-freq", type=int, default=2, help="最小词频阈值")
    parser.add_argument(
        "--single-char-min-freq",
        type=int,
        default=10,
        help="单字自动保留的最小词频，默认 10；设为 0 表示不保留任何单字",
    )
    parser.add_argument(
        "--filter-english-words",
        action="store_true",
        help="过滤纯英文词（默认不过滤）",
    )
    parser.add_argument("--stopwords", type=str, default="", help="额外停用词文件路径")
    parser.add_argument("--font-path", type=str, default="", help="中文字体路径")
    args = parser.parse_args()

    lyrics = load_lyrics(args.target_json)
    text = preprocess_lyrics(lyrics)

    defaults = build_default_output_paths(args.target_json)
    cleaned_text_path = args.cleaned_text.strip() or defaults["cleaned"]
    os.makedirs(os.path.dirname(cleaned_text_path) or ".", exist_ok=True)
    with open(cleaned_text_path, "w", encoding="utf-8") as f:
        f.write(text)

    stopwords = load_stopwords(args.stopwords.strip())
    freq = tokenize_and_count(
        text,
        stopwords,
        args.single_char_min_freq,
        filter_english_words=args.filter_english_words,
    )
    filtered_freq = Counter({k: v for k, v in freq.items() if v >= args.min_freq})

    if not filtered_freq:
        raise ValueError("清洗后没有可用于词云的词，请调整停用词或 min-freq")

    top_words_path = args.top_words.strip() or defaults["top_words"]
    os.makedirs(os.path.dirname(top_words_path) or ".", exist_ok=True)
    save_top_words(filtered_freq, top_words_path, args.topn)

    image_path = args.image.strip() or defaults["image"]
    font_path = pick_font_path(args.font_path.strip())
    generate_wordcloud(filtered_freq, image_path, font_path)

    print(f"歌词条目数: {len(lyrics)}")
    print(f"清洗文本输出: {cleaned_text_path}")
    print(f"高频词输出: {top_words_path}")
    print(f"词云图片输出: {image_path}")


if __name__ == "__main__":
    main()
