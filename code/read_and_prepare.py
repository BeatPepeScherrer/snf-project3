'''
pip install pandas numpy scikit-learn umap-learn hdbscan sentence-transformers nltk tqdm
# Optional (for OpenAI-based embeddings or labels)
pip install openai

Expected input: a CSV or JSON with at least these columns:
- `Companies`, `Company Sectors`, `Company Headquarters`, `Countries`, `Backdate`
- `story_text` and `response_text`
'''


import unicodedata
import pandas as pd
import numpy as np
import os, re, math
from datetime import datetime
import json
from pathlib import Path
import csv


# === Config ===
ROOT_DIR = "C:/Users/bscherrer/Documents/snf-project3"
INPUT_PATH = "data/20250905_1712_bhrrc_scraper_output.json"
INPUT_FORMAT = "json"           # "csv" or "json"
TEXT_COLUMNS = ["response_text"]  # adapt if needed
ID_COL = None  # if you have a unique ID column, put its name here
DATE_COL = "Backdate"  # optional; we'll try to parse
SAVE_PREFIX = "unsup_themes"

OUTPUT_PATH = Path(os.path.join(ROOT_DIR, 'data', 'prepared_df.csv'))

def load_data(path, fmt="json"):
    if fmt == "csv":
        df = pd.read_csv(path)
    elif fmt == "json":
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            df = [json.loads(line) for line in lines]   
        with open('20250905_1712_bhrrc_scraper_output.json', 'w', encoding='utf-8') as f:
            json.dump(df, f, indent=2, ensure_ascii=False) 
    else:
        raise ValueError("Unsupported format")
    return df

# set working directory
os.chdir(ROOT_DIR)

df = load_data(INPUT_PATH, INPUT_FORMAT)
df = pd.DataFrame(df)
print("Loaded rows:", len(df))
'''
## 2) Clean & prepare the corpus

- Merge story & response into one `text` field per case (or keep both; we’ll default to a combined text).
- Light normalization (whitespace, unicode). Keep punctuation and case (can help for names).

We also add a short `doc_id` for traceability in later exports.
'''

# check for empty content strings and remove them
# mask = df.response_text.str.strip() == ''
# df = df[~mask].copy().reset_index(drop=True)
# print(f'Removed {np.sum(mask)} entries with empty content strings. Remaining entries: {len(df)}')

# check for duplicates and remove them
mask = df[['Companies', 'Company Sectors', 'Company Headquarters', 'Countries', 'Response Sectors', 'Backdate']].duplicated(keep = 'first')
df = df[~mask].copy().reset_index(drop=True)
print(f'Found and removed {mask.sum()} duplicates.')


def normalize_text(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Normalize text and assign to new column
df["text"] = df["response_text"].apply(normalize_text)

# Create a doc_id for traceability
if ID_COL and ID_COL in df.columns:
    df["doc_id"] = df[ID_COL].astype(str)
else:
    df["doc_id"] = [f"doc_{i:04d}" for i in range(len(df))]

# Parse Backdate if present (day.month.year common format)
def parse_backdate(x):
    if pd.isna(x):
        return pd.NaT
    x = str(x).strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(x, fmt).date()
        except:
            pass
    return pd.NaT

if "Backdate" in df.columns:
    df["date"] = df["Backdate"].apply(parse_backdate)
else:
    df["date"] = pd.NaT

# Filter empty texts
df = df[df["text"].str.len() > 0].reset_index(drop=True)

# Check if data cleaning worked
print("After cleaning, rows:", len(df))
df[["doc_id", "Companies", "Company Sectors", "Countries", "text", "date"]].head(3)

# write cleaned df to .csv for later analysis
df.to_csv(
    OUTPUT_PATH,
    sep=',',
    index=False,
    encoding='utf-8',
    quoting=csv.QUOTE_MINIMAL,
    errors='replace' if pd.__version__ >= '1.5' else None
)