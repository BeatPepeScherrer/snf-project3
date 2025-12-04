narrative_extraction_prompt = """
You are an expert on ethical, human rights and environmental violations within supply chains.
You are given company responses to allegations of misconduct on issues mainly related to human rights but also to the environment and business ethics. 
You want to further investigate, why these violations happened in the first place.
The responses may be in German, French, or Spanish or Portuguese. Your reply should be in English.

Your task is to identify and extract sentences relevant to explaining and understanding why the company allegedly was involved or partly responsible for the violation of human rights, environmental or ethical standards.


Your job:

Decide whether the text contains any sentences that contain relevant information for explaining why the accused company was involved in the alleged violation and consequently had to write a response text.

In particular, this means that you Include sentences that:

- Describe root causes or contributing factors (e.g. lack of oversight, supplier behavior, economic incentives).
- Mention company policies or practices that failed or were missing.
- Explain structural problems (e.g. complex supply chain, subcontracting, weak audits).
- Discuss knowledge, negligence, or complicity of the company.

and 

Exclude sentences that are only:

- Generic PR statements (“We take human rights very seriously”).
- Pure factual reporting with no link to why the violation occurred (e.g. “The incident took place in 2021”).

Output those sentences (or items) verbatim in a single list (this is the only content I will cluster).
Explain why the items are relevant.
Return only a JSON object that validates the schema below. No extra commentary.

***Exact schema***:
{
  "type": "object",
  "required": ["document_id", "has_relevant", "items", "annotations", "overall_explanation"],
  "properties": {
    "document_id": { "type": "string" },
    "has_relevant": { "type": "boolean" },
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
        "required": ["unit", "why_relevant"],
        "properties": {
          "unit": { "type": "string", "enum": ["sentence", "narrative"] },
          "sentence_indices": {
            "type": "array",
            "items": { "type": "integer", "minimum": 0 }
          },
          "why_relevant": { "type": "string" }
        }
      }
    },
    "overall_explanation": { "type": "string" }
  }
}

Notes: 
If nothing is relevant: set "has_relevant": false, "items": [], "annotations": [], and explain in "overall_explanation".
Keep each items[i] short: a single sentence or a tight 1-2 sentence narrative (≤ 60 words).
"""


relational_contracting_narrative_extraction = """
You are an expert on ethical, human rights and environmental violations of large corporations.
You are given company responses to allegations of misconduct on issues mainly related to human rights but also to the environment and business ethics. 
The responses may be in German, French, or Spanish or Portuguese. Your reply should be in English.
It is possible that the allegation was caused by a violation of the supplier code of conduct somewhere down the supply chain of the company.

Your task is to identify whether the root cause of the violation might be related to relational contracting issues in the supply chain.
Relational contracting issues could be insufficient communication between company and suppliers (e.g. communication of associated sanctions in case of non-compliance), lack of trust, inadequate assurances, unclear expectations, or challanges in managing uncertainty.
The responses may be in German, French, or Spanish or Portuguese. Your reply should be in English.

Your job:

Decide from the company response whether the text is relevant in that it points to relational contracting issues in the supply chain that might have caused or contributed to the violation.

Output those items verbatim in a single list (this is the only content I will cluster).

Explain why the items (or the absence of items) are relevant; include which RQs are addressed.

Here is an example of an answer that would be relevant:
"Regarding your concern, we started a fact-finding process and communication with Cal-Comp
Electronics (Thailand) to verify the situation."

Here is an example of an answer that would not be relevant:
"We appreciate your engagement and remain committed to continuous improvement in our human
rights due diligence efforts."

The examples are for illustration only. Do not include them in your output.

Return only a JSON object that validates the schema below. No extra commentary.

***Exact schema***:
{
  "type": "object",
  "required": ["document_id", "has_relevant", "questions_addressed", "items", "annotations", "overall_explanation"],
  "properties": {
    "document_id": { "type": "string" },
    "has_relevant": { "type": "boolean" },
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
You are an analytical assistant. You will read sentences of company responses to allegations of misconduct on issues mainly related to human rights but also to the environment and business ethics.
Your task is to identify the overarching narrative(s) that connect all sentences in the cluster.
Return concise JSON only. For each cluster:\n"
    "- a short label (≤6 words),\n"
    "- a one-sentence rationale,\n"
    "- relevant RQs ⟨1..6⟩ from:\n"
    "  1) role of relational contracts; 2) uncertainty types; 3) how managers deal with uncertainty;\n"
    "  4) how expectations are communicated/negotiated; 5) assurances; 6) non-compliance & sanctions.\n"
Be precise and avoid overclaiming.
'''

labelling_prompt3 = '''
You are an analytical assistant. You will read sentences of company responses to allegations of misconduct on issues mainly related to human rights but also to the environment and business ethics.
The sentences refer to relational contracting issues in the supply chain that might have caused or contributed to the violation.
Your task is to identify the overarching narrative(s) that connect all sentences in the cluster.
Return concise JSON only. For each cluster:\n"
    "- a short label (≤6 words),\n"
    "- a one-sentence rationale,\n"
    "- degree of similarity (0-100) among the sentences,\n"
Be precise and avoid overclaiming.
Do not infer facts or details beyond what is present in the sentences.
'''