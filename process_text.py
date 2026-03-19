# pip install sentencex
from sentencex import segment
import re
import uuid
import os
LANGUAGE_CODE = {
    'Akan': 'aka', 'Albanian': 'sq', 'Amharic': 'am', 'Arabic': 'ar', 'Armenian': 'hy',
    'Assamese': 'as', 'Azerbaijani': 'az', 'Basque': 'eu', 'Bashkir': 'ba', 'Bengali': 'bn',
    'Bosnian': 'bs', 'Bulgarian': 'bg', 'Burmese': 'my', 'Catalan': 'ca', 'Chinese': 'zh',
    'Croatian': 'hr', 'Czech': 'cs', 'Danish': 'da', 'Dutch': 'nl', 'English': 'en',
    'Estonian': 'et', 'Faroese': 'fo', 'Finnish': 'fi', 'French': 'fr', 'Galician': 'gl',
    'Georgian': 'ka', 'German': 'de', 'Greek': 'el', 'Gujarati': 'gu', 'Haitian Creole': 'ht',
    'Hausa': 'ha', 'Hebrew': 'he', 'Hindi': 'hi', 'Hungarian': 'hu', 'Icelandic': 'is',
    'Indonesian': 'id', 'Italian': 'it', 'Japanese': 'ja', 'Kannada': 'kn', 'Kazakh': 'kk',
    'Korean': 'ko', 'Kurdish': 'ckb', 'Kyrgyz': 'ky', 'Lao': 'lo', 'Lithuanian': 'lt',
    'Luxembourgish': 'lb', 'Macedonian': 'mk', 'Malay': 'ms', 'Malayalam': 'ml', 'Maltese': 'mt',
    'Maori': 'mi', 'Marathi': 'mr', 'Mongolian': 'mn', 'Nepali': 'ne', 'Norwegian': 'no',
    'Norwegian Nynorsk': 'nn', 'Pashto': 'ps', 'Persian': 'fa', 'Polish': 'pl', 'Portuguese': 'pt',
    'Punjabi': 'pa', 'Romanian': 'ro', 'Russian': 'ru', 'Serbian': 'sr', 'Sinhala': 'si',
    'Slovak': 'sk', 'Slovenian': 'sl', 'Somali': 'so', 'Spanish': 'es', 'Sundanese': 'su',
    'Swahili': 'sw', 'Swedish': 'sv', 'Tamil': 'ta', 'Telugu': 'te', 'Thai': 'th',
    'Turkish': 'tr', 'Ukrainian': 'uk', 'Urdu': 'ur', 'Uzbek': 'uz', 'Vietnamese': 'vi',
    'Welsh': 'cy', 'Yiddish': 'yi', 'Yoruba': 'yo', 'Zulu': 'zu'
}

# ==================================================
# CONSTANTS
# ==================================================

QUOTE_SPACE = "\uFFFF"  # invisible placeholder for protected quotes
PUNCT_RE = re.compile(r'[.,;:!?]')


# ==================================================
# CLEAN TEXT (KEEP PUNCTUATION)
# ==================================================

def clean_text(text):
    replacements = {
        "**": "",
        "*": "",
        "#": "",
        "—": "",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ==================================================
# PROTECT SHORT QUOTES (ATOMIC QUOTE RULE)
# ==================================================

def protect_short_quotes(text, max_chars):
    """
    If a quoted span fits entirely within max_chars,
    protect it so it behaves like a single token.
    """
    def repl(match):
        quote = match.group(0)
        if len(quote) <= max_chars:
            return quote.replace(" ", QUOTE_SPACE)
        return quote

    return re.sub(r'"[^"]+"', repl, text)


def restore_quotes(text):
    return text.replace(QUOTE_SPACE, " ")


# ==================================================
# SMART SPLIT FOR LONG SENTENCES (QUOTE AWARE)
# ==================================================

def smart_split_long_sentence(sentence, max_chars=300, lookback=60):
    words = re.findall(r'\S+\s*', sentence)
    chunks = []
    buffer = ""
    in_quote = False

    for w in words:
        tentative = buffer + w
        quote_count = w.count('"')

        # 1️⃣ SAFE ADD
        if len(tentative) <= max_chars:
            buffer = tentative
            if quote_count % 2 != 0:
                in_quote = not in_quote
            continue

        # 2️⃣ OVERFLOW INSIDE QUOTE → MOVE WHOLE QUOTE
        if in_quote:
            if buffer.strip():
                chunks.append(buffer.strip())
            buffer = w
            if quote_count % 2 != 0:
                in_quote = not in_quote
            continue

        # 3️⃣ NORMAL PUNCTUATION-AWARE REBALANCE
        split_at = None
        search_region = buffer[-lookback:]

        matches = list(PUNCT_RE.finditer(search_region))
        if matches:
            last = matches[-1]
            split_at = len(buffer) - lookback + last.end()

        if split_at:
            chunks.append(buffer[:split_at].strip())
            buffer = buffer[split_at:].lstrip() + w
        else:
            chunks.append(buffer.strip())
            buffer = w

        if quote_count % 2 != 0:
            in_quote = not in_quote

    if buffer.strip():
        chunks.append(buffer.strip())

    return chunks


# ==================================================
# SENTENCE-FIRST CHUNKER
# ==================================================

def split_into_chunks(text, lang_code="en", max_chars=300):
    if len(text) <= max_chars:
        return [text]

    sentences = list(segment(lang_code, text))
    chunks = []
    current = ""

    for sen in sentences:
        sen = sen.strip()

        if len(sen) > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(smart_split_long_sentence(sen, max_chars))
            continue

        tentative = f"{current} {sen}".strip() if current else sen

        if len(tentative) <= max_chars:
            current = tentative
        else:
            chunks.append(current.strip())
            current = sen

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ==================================================
# FIX DANGLING QUOTES BETWEEN CHUNKS
# ==================================================

def repair_dangling_quotes(chunks):
    fixed = []

    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()

        if i > 0:
            prev = fixed[-1]
            if prev.endswith('"') and chunk.startswith('"'):
                chunk = chunk[1:].lstrip()

        fixed.append(chunk)

    return fixed


# ==================================================
# TTS FILE NAME
# ==================================================

def get_tts_file_name(text, language="en"):
    temp_audio_dir = "./ai_tts_voice/"
    os.makedirs(temp_audio_dir, exist_ok=True)

    clean = re.sub(r'[^a-zA-Z\s]', '', text or "")
    clean = clean.lower().strip().replace(" ", "_")[:20] or "audio"

    uid = uuid.uuid4().hex[:8].upper()
    language = language.lower().strip()

    return os.path.join(
        temp_audio_dir,
        f"{clean}_{language}_{uid}.wav"
    )


# ==================================================
# main funtion
# ==================================================

def text_chunk(text, language="English", char_limit=280):
    lang_code = LANGUAGE_CODE.get(language, LANGUAGE_CODE.get("English", "en"))

    # text = clean_text(text) #because Qwen3-TTS can handle that

    # 🔒 Atomic quote protection
    text = protect_short_quotes(text, char_limit)

    if len(text) > char_limit:
        print("⚠️ The text is too long. Breaking it into smaller pieces for TTS.")

    chunks = split_into_chunks(text, lang_code, char_limit)
    chunks = repair_dangling_quotes(chunks)

    # 🔓 Restore spaces inside quotes
    chunks = [restore_quotes(c) for c in chunks]

    tts_file_name = get_tts_file_name(text, lang_code)
    return chunks, tts_file_name


# ==================================================
# TEST
# ==================================================

# from process_text import text_chunk
# text="Hi, this is a test"
# chunks, tts_filename =text_chunk(text,  language="English", char_limit=280)

if __name__ == "__main__":
    text = "He said \"You are a looser\""  # @param {type: "string"}

    language="English"  # @param {type: "string"}
    char_limit = 20  # @param {type: "number"}

    chunks, filename = text_chunk(text, language, char_limit)

    print(filename)
    print(len(chunks))
    for c in chunks:
        print(len(c), c)
