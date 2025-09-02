import os, json, re, hashlib, collections, argparse
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import JsonOutputParser


parser = argparse.ArgumentParser(description="Run Langflow API on a set of PDF files and output results to a CSV.")
parser.add_argument("--input_file", type=str, help="Relative path from the current working directory to the directory containing PDFs", default="data/test.json")
parser.add_argument("--json_path", type=str, help="Relative path from the current working directory to the directory containing jsons", default="data")
parser.add_argument("--output_path", type=str, help="Relative path from the current working directory to the directory containing result files", default="results")
args = parser.parse_args()

current_working_directory = Path.cwd()
relative_json_path = Path(args.json_path)
relative_output_path = Path(args.output_path)
relative_json_file = Path(args.input_file)
json_directory = current_working_directory / relative_json_path
output_directory = current_working_directory / relative_output_path
json_file = current_working_directory / relative_json_file

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
load_dotenv(find_dotenv())

def parse_date(s: str) -> Optional[str]:
    for fmt in ("%d.%m.%Y","%Y-%m-%d","%d/%m/%Y","%m/%d/%Y"):
        try:
            return datetime.datetime.strptime(s.strip(), fmt).date().isoformat()
        except Exception:
            pass
    return None

def load_rows(path: str) -> List[Dict[str, Any]]:
    p = path
    rows = []
    with p.open("r", encoding="utf-8") as f:
        try:
            # Try JSON Lines
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        except Exception:
            f.seek(0)
            rows = json.load(f)

    out = []
    for idx, r in enumerate(rows):
        story = (r.get("story_text") or "").strip()
        resp  = (r.get("response_text") or "").strip()
        if not story and not resp:
            continue
        merged = " ".join([str(r.get("Title") or ""), str(r.get("Responded To") or ""), story, resp]).strip()
        out.append({
            "row_id": idx,
            "company": r.get("Companies") or "",
            "sector": r.get("Company Sectors") or "",
            "title": r.get("Title") or "",
            "url": r.get("URL") or "",
            "backdate": parse_date(r.get("Backdate") or "") if r.get("Backdate") else None,
            "countries": [c.strip() for c in (r.get("Countries") or "").split("|") if c.strip()],
            "tags": [t.strip() for t in (r.get("Tags") or "").split("|") if t.strip()],
            "story_text": story,
            "response_text": resp,
            "retrieval_text": merged
        })
    return out

# ---------- Embedding & Vector store (for optional RAG) ----------
def build_vectorstore(docs: List[Dict[str,Any]], persist_dir="outputs/chroma"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
    texts, metas = [], []
    for d in docs:
        for chunk in splitter.split_text(d["retrieval_text"]):
            texts.append(chunk)
            metas.append({k:d[k] for k in ["row_id","company","title","url","backdate","countries","tags"]})
    vs = Chroma.from_texts(texts=texts, embedding=OpenAIEmbeddings(), metadatas=metas, persist_directory=persist_dir)
    vs.persist()
    return vs

# ---------- LLM setup ----------
llm = ChatOpenAI(model=os.getenv("MODEL_NAME"), temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

class Extraction(BaseModel):
    row_id: int
    company: str
    sector: str
    backdate: Optional[str]
    evidence: List[Dict[str,str]] = Field(default_factory=list)
    relational_contracts: List[Dict[str,str]] = Field(default_factory=list)
    uncertainty_types: List[str] = Field(default_factory=list)
    mitigation_strategies: List[str] = Field(default_factory=list)
    communication_negotiation: List[str] = Field(default_factory=list)
    misunderstanding_handling: List[str] = Field(default_factory=list)
    assurances: List[str] = Field(default_factory=list)
    non_compliance_issues: List[str] = Field(default_factory=list)
    sanctions: List[str] = Field(default_factory=list)

extractor_template = PromptTemplate.from_template(
    """You are an analyst extracting specific, factual items from corporate “story_text” and “response_text”.
Use ONLY the provided text. If not present, output empty arrays. No speculation.
Return JSON only, matching this schema:

{schema}

Metadata:
ROW_ID: {row_id}
COMPANY: {company}
SECTOR: {sector}
TITLE: {title}
URL: {url}
BACKDATE: {backdate}
COUNTRIES: {countries}
TAGS: {tags}

STORY_TEXT:
{story_text}

RESPONSE_TEXT:
{response_text}
"""
)
parser = JsonOutputParser(pydantic_object=Extraction)
extractor_chain = LLMChain(llm=llm, prompt=extractor_template)

def run_extraction(docs: List[Dict[str,Any]]):
    out_path = json_directory / "extractions.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for d in docs:
            resp = extractor_chain.run({
                "schema": parser.get_format_instructions(),
                "row_id": d["row_id"],
                "company": d["company"],
                "sector": d["sector"],
                "title": d["title"],
                "url": d["url"],
                "backdate": d["backdate"],
                "countries": d["countries"],
                "tags": d["tags"],
                "story_text": d["story_text"][:12000],
                "response_text": d["response_text"][:12000],
            })
            # Parse & validate
            try:
                obj = json.loads(resp)
            except Exception:
                obj = parser.parse(resp)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return str(out_path)

def aggregate(extractions_file: str):
    rows = []
    with open(extractions_file, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))

    def count_list(field):
        c = collections.Counter()
        for r in rows:
            for x in r.get(field) or []:
                if isinstance(x, dict):
                    c[x.get("type","").strip().lower()] += 1
                else:
                    c[str(x).strip().lower()] += 1
        return c

    agg = {
        "n_docs": len(rows),
        "relational_contracts": count_list("relational_contracts"),
        "uncertainty_types": count_list("uncertainty_types"),
        "mitigation_strategies": count_list("mitigation_strategies"),
        "communication_negotiation": count_list("communication_negotiation"),
        "misunderstanding_handling": count_list("misunderstanding_handling"),
        "assurances": count_list("assurances"),
        "non_compliance_issues": count_list("non_compliance_issues"),
        "sanctions": count_list("sanctions"),
        "examples": []
    }

    def first_example_with(label, field):
        for r in rows:
            labels = []
            if field == "relational_contracts":
                labels = [rc.get("type","").strip().lower() for rc in (r.get(field) or [])]
            else:
                labels = [str(x).strip().lower() for x in (r.get(field) or [])]
            if label in labels:
                # You can carry more metadata if you keep it in extractions
                return {
                    "row_id": r.get("row_id"),
                    "company": r.get("company"),
                    "label": label,
                    "quote": (r.get("evidence") or [{}])[0].get("span","")
                }
        return None

    for field in ["relational_contracts","uncertainty_types","assurances","sanctions"]:
        for label, _ in agg[field].most_common(5):
            ex = first_example_with(label, field)
            if ex: agg["examples"].append({"field": field, **ex})

    with open(output_directory / "summary.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)

    return rows, agg

def write_report(agg: Dict[str,Any], questions: List[str]):
    def top(counter, k=8):
        return ", ".join([f"{lbl} ({cnt})" for lbl, cnt in counter.most_common(k)])

    lines = []
    lines.append("# Supply Chain Sustainability – Corpus Synthesis\n")
    lines.append(f"**Documents analyzed:** {agg['n_docs']}\n")
    lines.append("## 1) Importance of relational contracts\n")
    lines.append(f"Top themes: {top(agg['relational_contracts'])}\n")
    lines.append("## 2) Most important uncertainty types\n")
    lines.append(f"{top(agg['uncertainty_types'])}\n")
    lines.append("## 3) How companies/managers deal with uncertainty\n")
    lines.append(f"{top(agg['mitigation_strategies'])}\n")
    lines.append("## 4) Communicating/negotiating new expectations\n")
    lines.append(f"{top(agg['communication_negotiation'])}\n")
    lines.append("## 5) Avoiding/dealing with misunderstandings\n")
    lines.append(f"{top(agg['misunderstanding_handling'])}\n")
    lines.append("## 6) Assurances given for cooperation/compliance\n")
    lines.append(f"{top(agg['assurances'])}\n")
    lines.append("## 7) Typical non‑compliance issues & sanctions\n")
    lines.append(f"Issues: {top(agg['non_compliance_issues'])}\n")
    lines.append(f"Sanctions: {top(agg['sanctions'])}\n")
    lines.append("\n### Illustrative examples\n")
    for ex in agg["examples"]:
        lines.append(f"- **{ex['field']} – {ex['label']}** → row {ex['row_id']} ({ex.get('company','')}): “{ex.get('quote','')[:220]}”")
    (output_directory / "final_report.md").write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    docs = load_rows(path = json_file)
    # Optional: build a vector store for interactive querying
    # vs = build_vectorstore(docs)

    extractions_file = run_extraction(docs)
    rows, agg = aggregate(extractions_file)

    QUESTIONS = [
        "importance of relational contracts",
        "uncertainty types from sustainability transitions",
        "managerial responses to uncertainty",
        "communication/negotiation of new expectations",
        "handling misunderstandings",
        "assurances for cooperation/compliance",
        "typical non-compliance issues and sanctions",
    ]
    write_report(agg, QUESTIONS)
    print("Done. Files in ./outputs")
