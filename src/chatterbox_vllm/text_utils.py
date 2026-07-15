
import re


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",","、","，","。","？","！"}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text

# Supported languages for the multilingual model
SUPPORTED_LANGUAGES = {
  "ar": "Arabic",
  "da": "Danish",
  "de": "German",
  "el": "Greek",
  "en": "English",
  "es": "Spanish",
  "fi": "Finnish",
  "fr": "French",
  "he": "Hebrew",
  "hi": "Hindi",
  "it": "Italian",
  "ja": "Japanese",
  "ko": "Korean",
  "ms": "Malay",
  "nl": "Dutch",
  "no": "Norwegian",
  "pl": "Polish",
  "pt": "Portuguese",
  "ru": "Russian",
  "sv": "Swedish",
  "sw": "Swahili",
  "tr": "Turkish",
  "zh": "Chinese",
}


# --- Inline per-segment language tags ---------------------------------------
# Lets a mostly-one-language utterance pronounce specific spans in another
# language while keeping the SAME voice, e.g.:
#     "Ladda vid en <en>Supercharger</en> nu."
# The model conditions on ONE language per generation, so the server splits the
# text into runs, synthesizes each run as its own single-language call (same
# voice clone), and concatenates the PCM. Tags are stripped before synthesis so
# the tokenizer only ever sees a leading language tag, never a mid-text one.
_SEGMENT_TAG_RE = re.compile(r"<([a-z]{2,3})>(.*?)</\1>", re.DOTALL | re.IGNORECASE)
_STRAY_TAG_RE = re.compile(r"</?[a-z]{2,3}>", re.IGNORECASE)


def split_language_segments(text: str, base_language_id: str) -> list[tuple[str, str]]:
    """Split text into ``[(language_id, segment_text)]`` runs.

    Text wrapped in ``<xx>...</xx>`` (xx = a supported language code, see
    ``SUPPORTED_LANGUAGES``) becomes a segment in language xx; everything else
    uses ``base_language_id``. Unknown codes are left as literal text. Backward
    compatible: text with no valid tags returns a single
    ``(base_language_id, text)`` segment.
    """
    segments: list[tuple[str, str]] = []
    pos = 0
    for m in _SEGMENT_TAG_RE.finditer(text):
        lang = m.group(1).lower()
        if lang not in SUPPORTED_LANGUAGES:
            continue  # not a real language tag — leave as literal text
        if m.start() > pos:
            segments.append((base_language_id, text[pos:m.start()]))
        segments.append((lang, m.group(2)))
        pos = m.end()
    if pos < len(text):
        segments.append((base_language_id, text[pos:]))

    # Strip any stray/unmatched tags and drop empty segments.
    cleaned = [(lang, _STRAY_TAG_RE.sub("", seg)) for lang, seg in segments]
    cleaned = [(lang, seg) for lang, seg in cleaned if seg.strip()]
    return cleaned or [(base_language_id, _STRAY_TAG_RE.sub("", text))]
