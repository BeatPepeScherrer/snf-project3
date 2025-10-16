narrative_extraction_prompt = """
You are an expert on ethical, human rights and environmental violations of large corporations.
You are given company responses to allegations of misconduct on issues mainly related to human rights but also to the environment and business ethics. 
The responses may be in German, French, or Spanish or Portuguese. Your reply should be in English.

Your task is to identify and extract relational contracting narratives from the responses.
In particular, your task is to extract narratives that can provide an answer to at least one of the following resarch questions:
1.	What is the importance of relational contracts in supply chain sustainability transitions?
2.	What are the most important types of uncertainty in the supply chain created by sustainability transitions?
3.	How do companies and managers deal with different types of uncertainty in the supply chain created by sustainability transitions?
4.	How do companies communicate or negotiate new expectations? How do parties avoid or deal with misunderstandings?
5.	What are the assurances given to partners for ensuring cooperation and fostering compliance with sustainability standards?
6.	What are typical issues of non-compliance and what are the associated sanctions?

A relational contract refers to an implicit agreement between parties that goes beyond formal contracts, emphasizing trust, cooperation, and long-term relationships.
Sustainability transitions refer to the complex processes involved in shifting established socio-technical systems towards enhanced sustainability, requiring profound transformations rather than just incremental changes, and emphasizing the importance of governance and collaboration among diverse stakeholders.


Your job:

Decide whether the text contains any sentences or short narratives that help answer ≥1 RQ.

Output those items verbatim in a single list (this is the only content I will cluster).

Explain why the items (or the absence of items) are relevant; include which RQs are addressed.

Return only a JSON object that validates the schema below. No extra commentary.

***Exact schema***:
{
  "type": "object",
  "required": ["document_id", "has_relevant", "questions_addressed", "items", "annotations", "overall_explanation"],
  "properties": {
    "document_id": { "type": "string" },
    "has_relevant": { "type": "boolean" },
    "questions_addressed": {
      "type": "array",
      "items": { "type": "integer", "minimum": 1, "maximum": 6 }
    },
    "items": {
      "description": "List of verbatim sentences or short narratives to be clustered later.",
      "type": "array",
      "items": { "type": "string" }
    },
    "annotations": {
      "description": "Same length as items; each entry describes the corresponding item.",
      "type": "array",
      "items": {
        "type": "object",
        "required": ["unit", "questions", "why_relevant"],
        "properties": {
          "unit": { "type": "string", "enum": ["sentence", "narrative"] },
          "sentence_indices": {
            "type": "array",
            "items": { "type": "integer", "minimum": 0 }
          },
          "questions": {
            "type": "array",
            "items": { "type": "integer", "minimum": 1, "maximum": 6 }
          },
          "why_relevant": { "type": "string" }
        }
      }
    },
    "overall_explanation": { "type": "string" }
  }
}

Notes: 
If nothing is relevant: set "has_relevant": false, "questions_addressed": [], "items": [], "annotations": [], and explain in "overall_explanation".
Keep each items[i] short: a single sentence or a tight 1-2 sentence narrative (≤ 60 words).
"""


labelling_prompt = """
You are an analytical assistant. You will read sentences of company responses to to allegations of misconduct on issues mainly related to human rights but also to the environment and business ethics

Your task is to identify the overarching narrative(s) that connect all sentences in the cluster.

***Instructions***:
1. Carefully read all sentences in the cluster.
2. Determine whether a single coherent overarching narrative connects the sentences.
3. If a single narrative exists:
   - Provide a concise label (max 3 words) summarizing the narrative under "overarching_label".
   - Provide a 2-3 sentence description (max 150 words) under "overarching_narrative".
   - Set "alternative_narratives" as an empty list.
4. If no single coherent narrative exists:
   - Leave "overarching_label" and "overarching_narrative" empty.
   - Identify up to 3 distinct narratives. For each, provide a concise label (max 3 words) and a 2-3 sentence description (max 150 words).
   - Store these under "alternative_narratives" as a list of objects with "label" and "description", ordered from most to least represented.
5. If the summaries are too diverse or unrelated to form any coherent narratives:
   - overarching_label: "misc"
   - overarching_narrative: "misc"
   - alternative_narratives: []
6. Do not infer facts or details beyond what is present in the summaries.
7. Return strictly valid JSON only using the key "overarching_narratives". Do not include explanations or text outside the JSON.

***Output format***:
{{
  "cluster": "cluster_{{cluster_id}}",
  "overarching_label": "up to 3 words",
  "overarching_narrative": "up to 150 words",
  "alternative_narratives": [
    {{"label": "up to 3 words", "description": "up to 150 words"}},
    {{"label": "up to 3 words", "description": "up to 150 words"}},
    {{"label": "up to 3 words", "description": "up to 150 words"}}
  ]
}}

"""


labelling_prompt2 = '''
You are an analytical assistant. You will read sentences of company responses to to allegations of misconduct on issues mainly related to human rights but also to the environment and business ethics.
Your task is to identify the overarching narrative(s) that connect all sentences in the cluster.
Return concise JSON only. For each cluster:\n"
    "- a short label (≤6 words),\n"
    "- a one-sentence rationale,\n"
    "- relevant RQs ⟨1..6⟩ from:\n"
    "  1) role of relational contracts; 2) uncertainty types; 3) how managers deal with uncertainty;\n"
    "  4) how expectations are communicated/negotiated; 5) assurances; 6) non-compliance & sanctions.\n"
Be precise and avoid overclaiming.
'''
