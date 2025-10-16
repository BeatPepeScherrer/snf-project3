from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
import json
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import time
import os

from prompts import narrative_extraction_prompt

# read in openai key from .env
load_dotenv()

ROOT_DIR = "C:/Users/bscherrer/Documents/snf-project3"
INPUT_PATH = Path(os.path.join(ROOT_DIR, "data", "prepared_df.csv"))
OUTPUT_PATH = Path(os.path.join(ROOT_DIR, 'data', 'narratives.json'))


LLM = ChatOpenAI(model_name="gpt-5-mini", temperature=0)
PROMPT = ChatPromptTemplate.from_template(narrative_extraction_prompt)

if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)
    doc_ids = df["doc_id"].astype(str).tolist()
    response_texts = df["text"].astype(str).tolist()

    results = []

    print("Extracting narratives with LLM...")
    for doc_id, response in tqdm(zip(doc_ids, response_texts), total=len(doc_ids)):
        prompt = narrative_extraction_prompt + f'\n\nDocument ID: {doc_id}\n\nResponse:\n{response}'
        messages = [HumanMessage(content=prompt)]
        llm_response = LLM.invoke(messages)

        try:
            response_text = llm_response.content.strip()
            json_resp = json.loads(response_text)
# Ensure document_id matches your doc_id (as string)
            json_resp["document_id"] = str(doc_id)
            results.append(json_resp)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON for doc_id: {doc_id}. Saving raw output.")
            results.append({
                "document_id": str(doc_id),
                "has_relevant": False,
                "questions_addressed": [],
                "items": [],
                "annotations": [],
                "overall_explanation": "JSON decode error. Raw output saved.",
                "raw_output": llm_response.content
            })
        except Exception as e:
            print(f"An error occurred for doc_id {doc_id}: {e}")
            results.append({
                "document_id": str(doc_id),
                "has_relevant": False,
                 "questions_addressed": [],
                "items": [],
                "annotations": [],
                "overall_explanation": f"Exception: {e}"
            })
            if "RateLimitError" in str(e):
                print("Rate limit exceeded. Sleeping for 60 seconds...")
                time.sleep(60)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)