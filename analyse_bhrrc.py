#!/usr/bin/env python3
"""
LLM IE pipeline (function-calling) for BHRRC rows

Input: JSON or JSONL with fields incl. "story_text" and "response_text"
Model: any GPT-class chat model supporting tool/function-calling
Output:
  1) ie_output.csv (row-level extractions + story_type)
  2) rq_bullets.txt (concise bullets answering the 6 research questions)
  3) charts/ (pngs): sector_year_counts.png, story_type_by_sector.png, uncertainties_bars.png

Usage:
  export OPENAI_API_KEY=sk-...
  python ie_llm_pipeline.py --in /path/to/input.jsonl --out ie_output.csv --bullets rq_bullets.txt --model gpt-4o
"""

import os, json, argparse, time, re
from typing import List, Dict, Any, Optional
from collections import Counter
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_MODEL = os.environ.get("MODEL_NAME", "gpt-4o")
RATE_LIMIT_SLEEP = float(os.environ.get("RATE_LIMIT_SLEEP", "1.0"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="in_path", default="data/test.json", help="Input JSON/JSONL file")
ap.add_argument("--out", dest="out_csv", default="ie_output.csv", help="Output CSV path")
ap.add_argument("--bullets", dest="bullets_path", default="rq_bullets.txt", help="Bullets TXT path")
ap.add_argument("--model", dest="model", default=DEFAULT_MODEL, help="Model name")
ap.add_argument("--limit", type=int, default=0, help="Optional cap on processed rows (0 = all)")
args = ap.parse_args()

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

def get_client():
    if not _HAS_OPENAI:
        raise RuntimeError("openai package missing. Install with: pip install openai")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set; pass --api-key or set the env var.")
    return OpenAI(api_key=key)

def tool_spec() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "extract_ie_schema",
            "description": "Extract structured IE fields from story_text/response_text for sustainability & human-rights analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "story_type": {
                        "type": "string",
                        "enum": ["human rights", "environmental", "ethical"]
                    },
                    "relational_contract_importance": {"type": "string"},
                    "uncertainty_types": {
                        "type": "array",
                        "items": {"type": "string",
                                  "enum": [
                                      "regulatory/policy",
                                      "technological/process",
                                      "market/demand",
                                      "social-license/community",
                                      "operational/logistics",
                                      "supplier/traceability",
                                      "geopolitical/legal",
                                      "data/measurement"
                                  ]}
                    },
                    "coping_strategies": {"type": "array", "items": {"type": "string"}},
                    "communication_negotiation": {"type": "array", "items": {"type": "string"}},
                    "assurances": {"type": "array", "items": {"type": "string"}},
                    "non_compliance_issues": {"type": "array", "items": {"type": "string"}},
                    "sanctions": {"type": "array", "items": {"type": "string"}},
                    "evidence_spans": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                },
                "required": ["story_type","relational_contract_importance","uncertainty_types",
                             "coping_strategies","communication_negotiation","assurances",
                             "non_compliance_issues","sanctions","confidence"]
            }
        }
    }

SYSTEM_PROMPT = """
You are an information-extraction assistant for research on RELATIONAL CONTRACTS in supply-chain sustainability transitions.

INPUTS
- Two text fields from a single JSON record:
  • STORY TEXT: third-party allegation/summary (may be terse, multilingual).
  • RESPONSE TEXT: company response (often PR/legal tone, may deny or deflect).
- Do NOT use outside knowledge. Extract ONLY what is explicitly stated or can be plainly inferred from these two inputs.

OUTPUT
- You MUST call the provided tool/function `extract_ie_schema` and populate its fields.
- Each field maps directly to one of six research questions; fill them with concise, evidence-grounded content:
  1) relational_contract_importance  → RQ1 (why & how relational contracts matter here)
  2) uncertainty_types               → RQ2 (main uncertainties created by the transition)
  3) coping_strategies               → RQ3 (how managers/companies deal with those uncertainties)
  4) communication_negotiation       → RQ4 (how expectations are communicated/negotiated; how misunderstandings are avoided/handled)
  5) assurances                      → RQ5 (assurances given to partners to ensure cooperation & compliance)
  6) non_compliance_issues + sanctions → RQ6 (typical issues of non-compliance and associated sanctions/remedies)

GENERAL RULES
- Be strictly factual: if a field is unsupported, return an EMPTY ARRAY (or “not evident” for the one summary field).
- Prefer SHORT, SPECIFIC noun phrases taken from or paraphrasing the texts (≤ 12 words per item).
- Deduplicate within each array; keep 1-8 best items (most explicit/specific).
- Quote snippets in `evidence_spans` for every non-empty field when possible (≤ 30 words each). Prefix each snippet with the source: “story: …” or “response: …”.
- Multilingual: if the input is not in English, you may paraphrase in English but keep quoted snippets in the original language.
- If the company denies an allegation, still classify `story_type` from the STORY TEXT content; set `non_compliance_issues` empty if no issue is actually described.

FIELD-BY-FIELD INSTRUCTIONS

story_type  (human rights | environmental | ethical)
- Classify based on the ALLEGATION (STORY TEXT), not the company response.
- human rights: labour/FOA/CB, discrimination/harassment, child/forced labour, security abuses/violence, land/FPIC, access to water/health/education, OHS/safety.
- environmental: pollution/contamination/spill, biodiversity/ecosystems/tailings/deforestation, water/air/soil impacts, climate/GHG.
- ethical: corruption/bribery/kickbacks, conflicts of interest, lobbying/misleading reporting/greenwashing, governance/CSR image.

relational_contract_importance  (short synthesis; 1-2 sentences)
- Explain the role of relational mechanisms IF present: repeated dealings, trust-building, norms, informal expectations, joint problem-solving, adaptation/flexibility clauses, works-council/union structures, grievance/mediation forums.
- If absent/ambiguous, return “not evident”.

uncertainty_types  (choose from the controlled list ONLY)
- Options: ["regulatory/policy","technological/process","market/demand","social-license/community","operational/logistics","supplier/traceability","geopolitical/legal","data/measurement"].
- Select all that clearly apply, grounded in the texts (e.g., permits/opposition → regulatory/policy or social-license/community; multi-tier opacity → supplier/traceability; OHS/process risks → technological/process or operational/logistics).

coping_strategies
- How managers/companies deal with uncertainties: e.g., “supplier engagement & capacity-building”, “on-site assessments”, “risk hotspot mapping”, “remediation before withdrawal”, “alliances with OEMs/industry”, “third-party studies”, “dialogue with authorities”, “traceability initiatives”, “audits & follow-ups”.

communication_negotiation
- Concrete communication/negotiation modes and misunderstanding-avoidance: e.g., “public statement/press response”, “works-council/union meetings”, “stakeholder/community consultations”, “grievance channel”, “mediation/facilitated dialogue”, “MoUs/agreements”, “disclosure of deeper supply chain”.

assurances
- Promises/assurances to partners for cooperation & compliance: e.g., “code/standard (RSS, CoC)”, “third-party audit/verification”, “traceability/disclosure commitments”, “training & corrective action plans”, “timelines & KPIs”, “remediation first; withdrawal last”, “collaborative initiatives/alliances”, “contractual clauses”, “operational rules (e.g., ‘No PPE, no work’)”.

non_compliance_issues
- Typical issues (ONLY if described): e.g., “wage/working hours”, “OHS breach”, “child/forced labour risk”, “FOA interference”, “harassment/abuse”, “pollution/tailings/biodiversity harm”, “water depletion/contamination”.
- If the text only contains a denial with no specific issue, leave EMPTY.

sanctions
- Associated sanctions/remedies IF mentioned: e.g., “probation”, “order suspension”, “contract termination/withdrawal”, “financial restitution”, “public disclosure”, “operational stop (‘No PPE, no work’)”, “corrective action with re-audit”.
- If absent, leave EMPTY.

evidence_spans
- Add 1-3 decisive short quotes per populated field when available, each prefixed by “story:” or “response:”.
- Keep each quote ≤ 30 words; choose the strongest, most specific phrasing.

confidence  (0-1)
- 0.8-1.0: explicit statements with clear quotes for most fields.
- 0.5-0.7: partial/implicit evidence.
- 0.0-0.4: minimal or conflicting signals.

QUALITY CHECKS BEFORE SUBMITTING
- Arrays contain unique, specific items (≤ 8). No generic restatements.
- Only controlled labels are used for `uncertainty_types`.
- No content invented beyond the provided texts.
- If nothing is supported for a field, return an empty array (or “not evident” for the summary field).
"""

USER_INSTRUCTIONS = """Extract the schema by calling the provided function. Do not answer in prose.
Return empty arrays when information is absent or ambiguous.
"""

def call_llm(client, model: str, story: str, response: str) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_INSTRUCTIONS},
        {"role": "user", "content": f"STORY TEXT:\n{story or ''}\n\nRESPONSE TEXT:\n{response or ''}"}
    ]
    tools = [tool_spec()]
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "extract_ie_schema"}},
                timeout=120
            )
            choice = resp.choices[0]
            tcalls = getattr(choice.message, "tool_calls", None)
            if tcalls:
                args = tcalls[0].function.arguments
                return json.loads(args)
            # Fallback: attempt JSON
            content = choice.message.content or "{}"
            try:
                return json.loads(content)
            except Exception:
                return {"story_type":"ethical","relational_contract_importance":"not evident",
                        "uncertainty_types":[],"coping_strategies":[],"communication_negotiation":[],
                        "assurances":[],"non_compliance_issues":[],"sanctions":[],"evidence_spans":[],"confidence":0.2}
        except Exception:
            if attempt == MAX_RETRIES-1:
                return {"story_type":"ethical","relational_contract_importance":"not evident (LLM error)",
                        "uncertainty_types":[],"coping_strategies":[],"communication_negotiation":[],
                        "assurances":[],"non_compliance_issues":[],"sanctions":[],"evidence_spans":[],"confidence":0.0}
            time.sleep(RATE_LIMIT_SLEEP*(attempt+1))

def load_rows(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
        if "\n{" in raw:  # JSONL
            for line in raw.splitlines():
                line=line.strip()
                if line:
                    data.append(json.loads(line))
        else:
            obj = json.loads(raw)
            data = obj if isinstance(obj, list) else [obj]
    return data

def parse_year(backdate: Optional[str]) -> Optional[int]:
    if not backdate:
        return None
    for fmt in ("%d.%m.%Y","%Y-%m-%d","%d/%m/%Y","%Y/%m/%d","%d.%m.%y","%Y.%m.%d"):
        try:
            return datetime.strptime(backdate, fmt).year
        except Exception:
            continue
    m = re.search(r"(20\d{2}|19\d{2})", backdate)
    return int(m.group(1)) if m else None

def aggregate_and_write(df: pd.DataFrame, bullets_path: str, charts_dir: str):
    os.makedirs(charts_dir, exist_ok=True)

    by_year = df.groupby("year").size().sort_index()
    plt.figure(); by_year.plot(kind="bar"); plt.title("Items per Year"); plt.xlabel("Year"); plt.ylabel("Count"); plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "sector_year_counts.png")); plt.close()

    top8 = df["Company Sectors"].value_counts().head(8).index.tolist() if "Company Sectors" in df.columns else []
    if top8:
        df_top = df[df["Company Sectors"].isin(top8)]
        pivot = pd.crosstab(df_top["Company Sectors"], df_top["story_type"])
        plt.figure(); pivot.plot(kind="bar", stacked=True); plt.title("Story Type by Sector (Top 8)")
        plt.xlabel("Company Sectors"); plt.ylabel("Count"); plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "story_type_by_sector.png")); plt.close()

    # Flatten lists and count uncertainties
    def flatten(col):
        bag = []
        for lst in df[col].dropna():
            if isinstance(lst, list): bag += lst
            elif isinstance(lst, str) and lst: bag += [x.strip() for x in lst.split(";") if x.strip()]
        return Counter(bag)

    unc_counts = flatten("uncertainty_types")
    if unc_counts:
        plt.figure(); pd.Series(unc_counts).sort_values(ascending=False).head(10).plot(kind="bar")
        plt.title("Top Uncertainties"); plt.xlabel("Type"); plt.ylabel("Count"); plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "uncertainties_bars.png")); plt.close()

    # Bullets (concise answers to RQs)
    def top_items(c: Counter, n=6): return [f"{k} ({v})" for k,v in c.most_common(n)]
    strat_counts = flatten("coping_strategies")
    assu_counts  = flatten("assurances")
    nc_counts    = flatten("non_compliance_issues")
    san_counts   = flatten("sanctions")

    bullets = []
    rci_like = df["relational_contract_importance"].dropna().astype(str).head(10).tolist()
    bullets.append("RQ1 – Importance of relational contracts:\n- " + ("\n- ".join(rci_like) if rci_like else "Not evident in most items."))

    bullets.append("RQ2 – Most important types of uncertainty:\n- " + ("\n- ".join(top_items(unc_counts)) if unc_counts else "No clear pattern."))

    bullets.append("RQ3 – How companies/managers cope:\n- " + ("\n- ".join(top_items(strat_counts)) if strat_counts else "Not clearly stated."))

    comm_counts = flatten("communication_negotiation")
    bullets.append("RQ4 – Communication/negotiation & avoiding misunderstandings:\n- " + ("\n- ".join(top_items(comm_counts)) if comm_counts else "Limited detail; few explicit channels mentioned."))

    bullets.append("RQ5 – Assurances to partners:\n- " + ("\n- ".join(top_items(assu_counts, n=8)) if assu_counts else "Few explicit assurances."))

    bullets.append("RQ6 – Typical non-compliance issues & associated sanctions:\n- Non-compliance: " +
                   (", ".join(top_items(nc_counts, n=8)) if nc_counts else "n/a") +
                   "\n- Sanctions: " + (", ".join(top_items(san_counts, n=8)) if san_counts else "n/a"))

    with open(bullets_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(bullets))

def main():
    rows = load_rows(args.in_path)
    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    client = get_client()

    out_records = []
    for r in rows:
        story = r.get("story_text", "") or r.get("Story", "") or ""
        response = r.get("response_text", "") or r.get("Response", "") or ""
        data = call_llm(client, args.model, story, response)
        record = {
            **r,
            "story_type": data.get("story_type"),
            "relational_contract_importance": data.get("relational_contract_importance"),
            "uncertainty_types": data.get("uncertainty_types", []),
            "coping_strategies": data.get("coping_strategies", []),
            "communication_negotiation": data.get("communication_negotiation", []),
            "assurances": data.get("assurances", []),
            "non_compliance_issues": data.get("non_compliance_issues", []),
            "sanctions": data.get("sanctions", []),
            "evidence_spans": data.get("evidence_spans", []),
            "confidence": data.get("confidence", None),
            "year": parse_year(r.get("Backdate") or r.get("backdate") or "")
        }
        out_records.append(record)
        time.sleep(RATE_LIMIT_SLEEP)

    df = pd.DataFrame(out_records)
    # Join lists as semicolon strings for CSV; split again later for charts if needed
    list_cols = ["uncertainty_types","coping_strategies","communication_negotiation",
                 "assurances","non_compliance_issues","sanctions","evidence_spans"]
    for c in list_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: "; ".join(x) if isinstance(x, list) else (x or ""))

    df.to_csv(args.out_csv, index=False, encoding="utf-8")

    # For charts/bullets, convert back to lists
    for c in list_cols:
        if c in df.columns:
            df[c] = df[c].apply(lambda s: [t.strip() for t in s.split(";") if t.strip()] if isinstance(s, str) else ([] if s is None else s))

    aggregate_and_write(df, args.bullets_path, charts_dir="charts")

if __name__ == "__main__":
    main()
