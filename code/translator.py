# This script detects non-english text and tranlates it to English via DeepL API. Text has to be in Pandas Dataframe 
# with the name "df" in the column "response_text"

import os, time, hashlib, json, sqlite3
import pandas as pd
from pathlib import Path
import deepl
from langdetect import detect


RESP_COL = "response_text"   # <-- your column
OUT_COL  = "response_text_en"
LANG_COL = "response_lang"

DEEPL_KEY = os.environ.get("DEEPL_API_KEY", None) 
deepl_translator = deepl.Translator(DEEPL_KEY) if DEEPL_KEY else None

# Simple SQLite cache to avoid re-translating
CACHE_DB = "translate_cache.sqlite"
Path(CACHE_DB).touch(exist_ok=True)
conn = sqlite3.connect(CACHE_DB)
conn.execute("""CREATE TABLE IF NOT EXISTS cache (
  provider TEXT,
  src_lang TEXT,
  text_hash TEXT PRIMARY KEY,
  text_original TEXT,
  text_translated TEXT
)""")
conn.commit()

def _hash(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def cache_get(provider, src_lang, text):
    h = _hash(text)
    row = conn.execute("SELECT text_translated FROM cache WHERE text_hash=?", (h,)).fetchone()
    return row[0] if row else None

def cache_put(provider, src_lang, text, translated):
    h = _hash(text)
    conn.execute("REPLACE INTO cache(provider, src_lang, text_hash, text_original, text_translated) VALUES (?,?,?,?,?)",
                 (provider, src_lang or "", h, text, translated))
    conn.commit()

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def _chunk_long_text(text, max_chars=3500):
    if not text:
        return [""]
    # naive split on sentence-ish boundaries
    parts, cur = [], []
    cur_len = 0
    for seg in re.split(r'(?<=[\\.\\!\\?])\\s+', text):
        if cur_len + len(seg) + 1 > max_chars and cur:
            parts.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(seg)
        cur_len += len(seg) + 1
    if cur:
        parts.append(" ".join(cur))
    return parts or [text]

def translate_deepl(texts, source_lang=None, target_lang="EN_GB", tag_handling=None, log_errors=True):
    assert deepl_translator, "DEEPL_API_KEY not set"
    out = []
    for batch in chunk(texts, 10):  # keep batches modest
        # Pre-split long items
        split_batches = []
        reconstruct = []
        for t in batch:
            chunks = _chunk_long_text(t, max_chars=3500)
            split_batches.append(chunks)
            reconstruct.append(len(chunks))

        flat = [c for chunks in split_batches for c in chunks]
        translated_flat = []
        if not flat:
            out.extend(batch)
            continue

        # Try a few retries with backoff
        attempts = 0
        while attempts < 4:
            try:
                res = deepl_translator.translate_text(
                    flat,
                    source_lang=source_lang,     # None = autodetect
                    target_lang=target_lang,     # try "EN" first
                    tag_handling=tag_handling    # e.g., "html" if needed
                )
                translated_flat = [r.text for r in res]
                break
            except deepl.exceptions.QuotaExceededException as e:
                raise  # don’t hide quota problems
            except Exception as e:
                attempts += 1
                if log_errors:
                    print(f"[DeepL] Error on batch (attempt {attempts}): {e.__class__.__name__}: {e}")
                time.sleep(1.5 * attempts)
        if not translated_flat:
            # failed completely: mark as failed so you can see it later
            translated_flat = flat  # fallback to originals, but we’ll flag later

        # Reconstruct per original item
        idx = 0
        for n in reconstruct:
            segs = translated_flat[idx:idx+n]
            idx += n
            out.append(" ".join(segs))
    return out

def detect_lang_safe(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return "unknown"
    try:
        return detect(s[:1000])
    except Exception:
        return "unknown"

# 1) Detect language of response_text
df["response_lang"] = df[RESP_COL].astype(str).apply(detect_lang_safe)

# 2) Translate only non-English responses
non_en_mask = df["response_lang"].ne("en")

# initialize output col as original (so EN rows and blanks are kept)
df[OUT_COL] = df[RESP_COL].astype(str)

# translate only the non-EN slice
to_translate = df.loc[non_en_mask, RESP_COL].astype(str).tolist()
translated = translate_deepl(to_translate, source_lang=None, target_lang="EN-GB")  # or "EN-GB"
df.loc[non_en_mask, OUT_COL] = translated

# quick sanity
print("Non-EN rows:", int(non_en_mask.sum()))
print("Unchanged after translation:", int((df[OUT_COL] == df[RESP_COL]).sum()))