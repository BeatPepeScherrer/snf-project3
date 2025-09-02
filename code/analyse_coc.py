import os
import argparse
from datetime import datetime
from pathlib import Path
import csv
import json
import requests
import fitz

# CLI args
parser = argparse.ArgumentParser(description="Run Langflow API on a set of PDF files and output results to a CSV.")
parser.add_argument("--pdf_folder", type=str, help="Relative path from the current working directory to the directory containing PDFs", default="pdfs")
parser.add_argument("--token", type=str, help="Langflow Astra Application Token")
args = parser.parse_args()

# Paths
current_working_directory = Path.cwd()
pdf_directory = current_working_directory / args.pdf_folder

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_csv = f"{timestamp}_coc_analysis_results.csv"

# load token from environment
token = os.environ.get("LANGFLOW_TOKEN")
if not token:
    raise ValueError("No Langflow token found. Please set LANGFLOW_TOKEN environment variable.")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}"
}

# Langflow API endpoint (your flow URL)
url = "https://api.langflow.astra.datastax.com/lf/c9a6c61c-ad3d-4cd2-9b66-3367fc5529c2/api/v1/run/bf21b0d2-a106-4080-8440-4761704f2077"

pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]

# Instructions without using a {context} variable
INSTRUCTIONS = """
You are provided with a company's supplier code of conduct or a similar document, that outlines a company's expectations for its suppliers regarding ethical, social, and environmental standards.

Your task is to analyze context and specifically search for explicit or implicit mentions of clarity and issues and challenges that arise because of a lack of clarity. Your task includes outputting your analysis in the form of a json file with a series of variables.

In relational contracting the credibility problem is well-known: should  one party believe another's promise? Here, we focus on the clarity problem in relational contracting: can one party (supplier) understand another's (company) promise? Both of these problems arise  naturally if formal contracts are infeasible or imperfect because the parties are unable to  articulate ex ante or to verify ex post their roles in and rewards from cooperating together. We focus on the case where roles or rewards cannot be fully articulated ex ante.

Clarity in company-supplier relationships in the context of sustainability means the supplier needs to know (a) what behaviors constitute cooperation by her (adhering or not adhering to expectations laid out in the supplier code of conduct), (b) what behaviors are then available to the company as cooperation or defection by him (Keep or end the business relationship), (c) what payoffs the company would receive from those available behaviors, and (d) what payoffs the supplier would receive if everyone cooperates versus not.

Instructions: Create a proper json output, that can be converted to .csv with the following columns: "Name" which is the name of the company. "Sector" which is the primary sector the company is operating in. "Ethical Behavior" which is the behavior that would constitute cooperation by the supplier with the ethical standards and expectations laid out in the code of conduct. State the standards and policies the company expects the supplier to adhere to if there are any. "Social Behavior" which is the behavior that would constitute cooperation by the supplier with the social and human rights standards and expectations laid out in the code of conduct. State the standards and policies the company expects the supplier to adhere to if there are any. "Environmental behavior" which is the behavior that would constitute cooperation by the supplier with the environmental standards and expectations laid out in the code of conduct. State the standards and policies the company expects the supplier to adhere to if there are any. "Punishment" which are the behaviors available to the company as cooperation or defection by him that are explicitly stated in the document. "Assessment Ethical" which is your assessment regarding how well the company is communicating their expectations about the supplier's ethical behavior. The assessment can be "excellent", "good", "medium", "bad". "Assessment Social" which is your assessment regarding how well the company is communicating their expectations about the supplier's socilal behavior. The assessment can be "excellent", "good", "medium", "bad".  "Assessment Environmental" which is your assessment regarding how well the company is communicating their expectations about the supplier's environmental behavior. The assessment can be "excellent", "good", "medium", "bad". "Other Strategies" which should contain any other strategies that the company uses to resolve potential clarity problems with the supplier.

If context provided does not contain explicit details about a company's supplier code of conduct or specific expectations regarding ethical, social, or environmental standards, just fill the entries of the table with "NA" instead of generating something inaccurate.
"""

# Target columns for CSV
TARGET_COLUMNS = [
    "file",
    "Name",
    "Sector",
    "Ethical Behavior",
    "Social Behavior",
    "Environmental behavior",
    "Punishment",
    "Assessment Ethical",
    "Assessment Social",
    "Assessment Environmental",
    "Other Strategies"
]

results = []

# Loop PDFs
for file in pdf_files:
    pdf_path = pdf_directory / file
    with fitz.open(pdf_path) as pdf_document:
        full_text = "".join([pdf_document[page_number].get_text() for page_number in range(len(pdf_document))])

    print(f"Processing {file}...")

    # Build a single prompt string: PDF text + instructions (no {context} variable)
    prompt_text = f"{full_text}\n\n-----\n\n{INSTRUCTIONS}"

    # If your flow uses a Prompt node, override its template directly (no context variable)
    PROMPT_NODE_ID = "Prompt-RYZzP"  # replace with your actual Prompt node ID
    payload = {
        "input_value": "",
        "output_type": "text",
        "input_type": "text",
        "tweaks": {
            PROMPT_NODE_ID: {
                "template": prompt_text
            }
        }
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    result_json = resp.json()

    def _get_inner_text(result_json, raw_text=""):
        # Try common Langflow shapes
        if isinstance(result_json, dict) and "outputs" in result_json:
            paths = [
                ["outputs", 0, "outputs", 0, "results", "text", "data", "text"],
                ["outputs", 0, "outputs", 0, "results", "message", "text"],
                ["outputs", 0, "results", "text", "data", "text"],
            ]
            for path in paths:
                try:
                    v = result_json
                    for k in path:
                        v = v[k]
                    if isinstance(v, str) and v.strip():
                        return v
                except Exception:
                    pass
        # Fallback: pull fenced JSON from the raw string
        m = re.search(r"```json\s*(\{.*?\})\s*```", raw_text or "", re.S)
        if m:
            return m.group(1)
        return raw_text or ""

    def _json_from_model_text(s: str) -> dict:
        s = (s or "").strip()
        # Strip outer code fences if present
        if s.startswith("```"):
            nl = s.find("\n")
            if nl != -1:
                s = s[nl+1:]
            end = s.rfind("```")
            if end != -1:
                s = s[:end]
            s = s.strip()
        # Try direct JSON
        try:
            return json.loads(s)
        except Exception:
            # Extract first balanced JSON object/array
            n = len(s); i = 0
            while i < n and s[i] not in "{[":
                i += 1
            if i == n:
                return {}
            start = s[i]; endc = "}" if start == "{" else "]"
            depth = 0; in_str = False; esc = False
            for j in range(i, n):
                ch = s[j]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                else:
                    if ch == '"':
                        in_str = True
                    elif ch == start:
                        depth += 1
                    elif ch == endc:
                        depth -= 1
                        if depth == 0:
                            return json.loads(s[i:j+1])
            return {}

    # ---- use the helpers ----
    result_json = resp.json()
    model_text = _get_inner_text(result_json, resp.text)  # <- this finds the ```json block
    parsed = _json_from_model_text(model_text)            # <- this loads it into a dict

    # If the model returned a list, pick the first dict-like item
    if isinstance(parsed, list):
        parsed = next((item for item in parsed if isinstance(item, dict)), {})

    # If it's still not a dict, fall back to empty dict
    if not isinstance(parsed, dict):
        parsed = {}

    # Normalize key casing if needed
    if "Environmental behavior" not in parsed and "Environmental Behavior" in parsed:
        parsed["Environmental behavior"] = parsed.pop("Environmental Behavior")

    # Build the CSV row
    row = {"file": file}
    for key in TARGET_COLUMNS[1:]:  # skip "file"
        val = parsed.get(key)
        if val is None or (isinstance(val, str) and val.strip() == ""):
            row[key] = "na"
        elif isinstance(val, (dict, list)):
            row[key] = json.dumps(val, ensure_ascii=False)
        else:
            row[key] = val if isinstance(val, str) else str(val)
    results.append(row)


# Write to CSV
if results:
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TARGET_COLUMNS)
        writer.writeheader()
        writer.writerows(results)

print(f"✅ Results saved to {output_csv}")
