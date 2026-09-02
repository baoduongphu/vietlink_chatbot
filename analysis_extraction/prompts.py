"""Prompts used by the analysis extraction workflow."""

ANALYSIS_PROMPT = """\
You are an expert conversation analyst specializing in detecting \
AMBIGUITY, UNCLEAR COMMUNICATION, and MISSING CONTEXT.

Analyze the conversation below and identify every ambiguity that \
genuinely occurred.

RULES:
1. Ground every finding strictly in the conversation.
2. Do not invent missing context, intentions, or facts.
3. Infer speaker roles only from the conversation.
4. An ambiguity must have caused, or plausibly could have caused, \
   misunderstanding, misdirected advice, rework, or clarification.
5. Do NOT flag statements merely because they are brief or incomplete \
   if their meaning is clear from context.
6. Keep separate cases separate when their root causes differ.
7. If multiple cases have the same root cause, analyze them separately \
   but mark them for merging.
8. Prefer concise reasoning over repetition.

For EACH case, return:

### Case {{number}}

**Evidence:** \
Quote only the minimum relevant excerpts needed to prove the ambiguity.

**What happened:** \
1-3 sentences describing the misunderstanding or missing information.

**Root cause:** \
Explain WHY the ambiguity occurred, not merely what information was missing. \
Focus on the communication mechanism or assumption that caused it.

**Category:** \
Choose the most specific category. Examples:
Missing Context, Undefined Scope, Ambiguous Terminology, Missing Constraint,
Undefined Success Criteria, Conflicting Requirement, Missing Format/Data Spec,
Unclear Process/Flow, Ambiguous Reference, Undefined Audience/Recipient,
Framework/Tool Side Effect, Other.

**Context type:** \
Select one or more:
Task/Action, Information/Advice, Document/Communication,
Planning/Decision, Technical/Problem-Solving, Emotional/Personal, General.

**Severity:** High / Medium / Low
Briefly explain the consequence if the ambiguity were not clarified.

**Confidence:** High / Medium / Low
Briefly explain how directly the conversation supports the finding.

**Generalizable principle:** \
State ONE concise principle that can apply to similar future conversations.
If no reusable principle exists, write exactly:
NOT generalizable — exclude from KB.

If two or more cases share the same root cause, add:
**Merge note:** Case X and Case Y share the same root cause and should \
be merged into one knowledge entry.

If no sufficiently grounded ambiguity exists, respond exactly:
"{no_case_marker}."

<conversation>
{conversation}
</conversation>
"""


EXTRACTION_PROMPT = """\
You receive an analysis of ambiguous cases found in a conversation (root \
cause, category, priority, confidence, and generalizability have already \
been analyzed). Your task is NOT to re-analyze — it is to CONVERT this \
analysis into knowledge base entries following the exact JSON schema \
below.

Rules:

1. Skip every case marked "NOT generalizable — exclude from KB".

2. If the analysis notes that 2 cases should be merged (same root cause), \
create a SINGLE entry, merging both cases' evidence into the "source" \
array.

3. For each remaining entry, fill in:
   - knowledge_id: "KB-{{n}}" where n is the sequence number within this \
batch (a real, globally-unique ID is assigned later — this is only a \
placeholder).
   - title: a short name based on "What happened"/Category.
   - category: use the category suggested in the analysis, as-is \
(free-text, not required to match a fixed enum).
   - description: rewrite "Root cause analysis" into a general PRINCIPLE \
(drop case-specific details, keep the underlying issue).
   - intent: why applying this knowledge is useful — inferred from \
"Severity reasoning".
   - priority: use the value proposed in "Severity reasoning" as-is.
   - confidence: use the value proposed in "Confidence reasoning" as-is.
   - status: always "Draft".
   - applicable_to: from "Applicable context type".
   - trigger: a short condition to recognize when a NEW request should \
activate this entry (based on "What happened" + category).
   - detection_rules: more specific than trigger — the pattern/signal \
that confirms the issue genuinely exists in a request.
   - clarification_questions: specific questions, reusable verbatim when \
asking about the same kind of issue again.
   - recommendation: how to rewrite the request to avoid this ambiguity \
from the start.
   - expected_outcome: the desired state after applying the \
recommendation.
   - source: an array with 1 object (or more if cases were merged):
     - conversation_id: "{conversation_id}"
     - root_cause: taken as-is from "Root cause analysis"
     - evidence: taken verbatim from "Evidence" (do NOT paraphrase)
     - case_summary: from "What happened", shortened to 1-2 sentences

4. Do NOT add any information beyond what's already in the analysis. If a \
field lacks enough data to fill accurately, leave it as an empty string \
"" rather than making something up.

Output ONLY a JSON array following the schema below, no text outside the \
JSON.

Schema:
[
  {{
    "knowledge_id": "string",
    "title": "string",
    "category": "string",
    "description": "string",
    "intent": "string",
    "priority": "High | Medium | Low",
    "confidence": "High | Medium | Low",
    "status": "Draft",
    "applicable_to": ["Task/Action" | "Information/Advice" | "Document/Communication" | "Planning/Decision" | "Technical/Problem-Solving" | "Emotional/Personal" | "General"],
    "trigger": "string",
    "detection_rules": "string",
    "clarification_questions": ["string"],
    "recommendation": "string",
    "expected_outcome": "string",
    "source": [
      {{
        "conversation_id": "string",
        "root_cause": "string",
        "evidence": "string",
        "case_summary": "string"
      }}
    ]
  }}
]

<analysis_result>
{analysis_result}
</analysis_result>
"""
