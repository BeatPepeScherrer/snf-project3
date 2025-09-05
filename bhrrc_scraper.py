# save as bhrrc_chrome_bs.py
# usage:
#   python bhrrc_chrome_bs.py --url https://www.business-humanrights.org/en/latest-news/shell-response-re-corrib-gas-protest/ --csv bhrrc_texts.csv
#   add --headless for headless mode

import os, csv, re, argparse, time, json, sys
from datetime import datetime
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup, NavigableString
from io import BytesIO
from urllib.parse import urljoin
import requests
from pdfminer.high_level import extract_text as pdf_extract_text
from io import BytesIO
from pdf2image import convert_from_bytes
import pytesseract

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (BHRRC scraper)"})

STOP_RESP = re.compile(r"(This is a response to|Timeline|Latest news|Company Responses)", re.I)
STOP_STORY = re.compile(r"(Company Responses|Timeline|Latest news)", re.I)

timestamp = datetime.now().strftime("%Y%m%d_%H%M")
output_json = f"data/{timestamp}_bhrrc_scraper_output.json"

# pytesseract / tessdata config
cand = [
    os.path.join(sys.prefix, "Library", "share", "tessdata"),  # Win + conda-forge
    os.path.join(sys.prefix, "share", "tessdata"),              # *nix + conda-forge
    r"C:\Program Files\Tesseract-OCR\tessdata",                 # Win official installer
]
tessdata = next((p for p in cand if os.path.exists(os.path.join(p, "eng.traineddata"))), None)
os.environ["TESSDATA_PREFIX"] = tessdata  # set to the folder containing *.traineddata

def find_first_pdf_url(response_page_html, base_url):
    """
    Prefer explicit attachment buttons, but accept any anchor ending in .pdf.
    Returns an absolute URL or None.
    """
    soup = BeautifulSoup(response_page_html, "lxml")

    # 1) Prefer "Download attachment" buttons
    for a in soup.select("a.button[href], a[href]"):
        href = (a.get("href") or "").strip()
        if href.lower().endswith(".pdf"):
            return urljoin(base_url, href)

    # 2) Fallback: any link into /documents/ that includes .pdf
    a = soup.select_one("a[href*='/documents/'][href*='.pdf']")
    if a and a.get("href"):
        return urljoin(base_url, a["href"])

    return None

def find_first_pdf_url(html, base):
    s = BeautifulSoup(html, "lxml"); j = lambda u: urljoin(base, u)
    for h in (t.get("href","").strip() for t in s.select("a[href],link[href]")):
        if ".pdf" in h.lower(): return j(h)
    try: d = json.loads(s.find("script", id="pageAsDataJSON", type="application/json").string or "{}")
    except: d = {}
    for k in ("source","downloadUrl","download_url","pdf","url","file"):
        v = d.get(k,"")
        if ".pdf" in v.lower(): return j(v)
    m = re.search(r'"(?:source|url|downloadUrl|download_url|pdf)"\s*:\s*"([^"]+?\.pdf[^"]*)"', html, re.I)
    return j(m.group(1)) if m else None

def download_pdf_text(pdf_url, timeout=60):
    r = SESSION.get(pdf_url, timeout=timeout); r.raise_for_status()
    buf = BytesIO(r.content)

    # Try pdfminer
    try:
        t = (pdf_extract_text(buf) or "").strip()
        if t: return t
    except Exception: pass

    # Fallback: pypdf
    try:
        from pypdf import PdfReader
        buf.seek(0); t = "\n".join((p.extract_text() or "") for p in PdfReader(buf).pages).strip()
        if t: return t
    except Exception: pass

    # OCR fallback
    buf.seek(0)
    pages = convert_from_bytes(buf.getvalue())
    return "\n\n".join(pytesseract.image_to_string(p) for p in pages).strip()
    
def init_browser(headless=True):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")   # no window
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    d = webdriver.Chrome(options=opts)
    d.set_page_load_timeout(30)
    return d
    
def get_html_with_chrome(driver, url):
    driver.get(url)
    return driver.page_source

def collect_after_h1_until_marker(soup, stop_regex):
    """Grab paragraphs/list items/quotes AFTER the H1 until a block matches stop_regex."""
    h1 = soup.find("h1")
    if not h1:
        # fallback: all <p> text
        return "\n\n".join(p.get_text(" ", strip=True) for p in soup.select("p"))
    out = []
    node = h1.find_next_sibling()
    while node:
        if isinstance(node, NavigableString):
            node = node.next_sibling
            continue
        text = node.get_text(" ", strip=True)
        if text and stop_regex.search(text):
            break
        for t in node.find_all(["p", "li", "blockquote"], recursive=True):
            s = t.get_text(" ", strip=True)
            if s: out.append(s)
        node = node.next_sibling
    return "\n\n".join(out).strip()

def find_parent_story_url_from_html(response_url, html):
    soup = BeautifulSoup(html, "lxml")
    anchor = soup.find(string=re.compile(r"This\s+is\s+a\s+response\s+to", re.I))
    if not anchor:
        return None
    link = anchor.find_next("a")
    if not link or not link.get("href"):
        return None
    return urljoin(response_url, link["href"])


def extract_response_text_preferring_pdf(response_page_url, response_page_html):
    """
    If a PDF attachment exists: download it and use its text.
    Otherwise: fall back to your HTML extraction logic.
    """
    # 1) Prefer PDF if present
    pdf_url = find_first_pdf_url(response_page_html, response_page_url)
    if pdf_url:
        pdf_text = download_pdf_text(pdf_url)
        if pdf_text:            # use PDF text if we got any
            return pdf_text
        # (optional) if you want to ALWAYS use PDF even if empty, just `return pdf_text`
        # return pdf_text

    # 2) Fallback: native HTML response (your existing logic)
    soup = BeautifulSoup(response_page_html, "lxml")

    # First try BHRRC's html-block container
    blocks = soup.select("div.block.html-block")
    texts = [b.get_text("\n", strip=True) for b in blocks if b.get_text(strip=True)]
    if texts:
        return "\n\n".join(texts).strip()

    # Then paragraphs after H1 until common markers
    STOP = re.compile(r"(This is a response to|Timeline|Latest news|Company Responses)", re.I)
    h1 = soup.find("h1")
    if not h1:
        return "\n\n".join(p.get_text(" ", strip=True) for p in soup.select("p"))
    out, node = [], h1.find_next_sibling()
    while node:
        if isinstance(node, NavigableString):
            node = node.next_sibling; continue
        txt = node.get_text(" ", strip=True)
        if txt and STOP.search(txt): break
        for t in node.find_all(["p","li","blockquote"], recursive=True):
            s = t.get_text(" ", strip=True)
            if s: out.append(s)
        node = node.next_sibling
    return "\n\n".join(out).strip()

def extract_story_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("title")
    if title_tag:
        return title_tag.get_text(strip=True)
    return ""
    

def append_jsonl(out_path, obj):
    """Append one JSON object per line (JSONL)."""
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def enrich_row_from_url(row, url_col="URL", headless=False):
    """Return {all input cols ... + story_text + response_text} for this row."""
    url = (row.get(url_col) or "").strip()
    if not url:
        return None
    
    driver = init_browser(headless=True)

    # Reuse existing scraping helpers:
    resp_html = get_html_with_chrome(driver, url)
    response_text = extract_response_text_preferring_pdf(url, resp_html)
    story_url = find_parent_story_url_from_html(url, resp_html)

    story_text = ""
    if story_url:
        story_html = get_html_with_chrome(driver, story_url)
        story_text = extract_story_text(story_html)

    enriched = dict(row)  # keep ALL original CSV fields
    enriched["story_text"] = (story_text or "").strip()
    enriched["response_text"] = (response_text or "").strip()
    return enriched

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="BHRRC -> JSONL")
    ap.add_argument("--in-csv", default="data/bhrrc_filtered.csv", help="Input CSV with a URL column")
    ap.add_argument("--url-col", default="URL", help="Name of the URL column")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    # robust CSV read
    with open(args.in_csv, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096); f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
            reader = csv.DictReader(f, dialect=dialect)
           # rows = list(reader)
           # row = rows[6] 
        except csv.Error:
            reader = csv.DictReader(f)


        for row in reader:
            try:
                obj = enrich_row_from_url(row, url_col=args.url_col, headless=args.headless)
                if obj:
                    append_jsonl(output_json, obj)
                    print(f"[ok] {row.get(args.url_col)}")
                time.sleep(0.2)  # polite pause
            except Exception as e:
                print(f"[error] {row.get(args.url_col)} -> {e}")
