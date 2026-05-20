"""Prompt builders for PaperBot graph LLM calls."""

import re


def classification_prompt(question: str) -> str:
    return f"""You are an intent classifier for PaperBot,
a chatbot over 5 academic research papers:
BERT, RoBERTa, BCI/EEG, and two Sentiment Analysis papers.

Classify the user question into exactly one category.

Categories:
- image  : user wants to SEE a figure, diagram, chart,
           formula, architecture, plot, or any visual
- table  : user EXPLICITLY asks for a table, data grid,
           or structured comparison. Strong signals:
           "give me the table", "show me the table",
           "give me the results table", "table for",
           "show the comparison table"
- text   : everything else — explanation, definition,
           what is X, how does X work, accuracy of X,
           performance of X, results of X, metrics of X,
           what algorithms, which models, comparisons.
           Even metric questions are text unless user
           explicitly says "table"
- greeting: hi, hello, bye, thanks, who are you
- error  : unrelated or incomprehensible

Additionally set needs_table_augment=true if the question asks
about numeric results, scores, benchmarks, accuracy, or
performance comparisons between models. Otherwise false.

needs_table_augment examples:
'what was BERT accuracy on MNLI'        → true
'which model performed best on GLUE'    → true
'compare BERT and RoBERTa F1 scores'    → true
'what is BERT'                          → false
'how does attention work'               → false
'show me CNN architecture'              → false

Additionally when intent=greeting, set greeting_subtype:
- 'farewell' if the message is a goodbye, bye, thanks/thank you,
  see you, take care, or any closing statement
- 'greeting' for hello, hi, who are you, or any opening statement

greeting_subtype examples:
'hi'                → greeting
'hello there'       → greeting
'who are you'       → greeting
'bye'               → farewell
'goodbye'           → farewell
'thanks'            → farewell
'thank you so much' → farewell
'see you later'     → farewell
'cheers'            → farewell

Examples:
"show me CNN architecture"              → image
"give me the BERT figure"               → image
"show me naive bayes formula"           → image
"show me the results table"             → table
"give me the table for BERT accuracy"   → table
"give me the table for it"              → table
"show me the comparison table"          → table
"what was BERT accuracy on MNLI"        → text
"what is BERT accuracy on MNLI"         → text
"what algorithms were compared"         → text
"what is the F1 score of SVM"           → text
"which model performed best"            → text
"what is BERT"                          → text
"explain how BERT works"                → text
"how does attention mechanism work"     → text
"what is masked language modeling"      → text
"hi there"                              → greeting
"hi"                                    → greeting
"hello"                                 → greeting
"thanks bye"                            → greeting
"what is the weather"                   → error

CRITICAL RULE:
Only classify as table when user explicitly uses the
word "table" or asks for structured data grid.
Never classify as table just because question mentions
accuracy, score, metrics, or results.

Question: "{question}"

Reply with ONLY one word: image, table, text,
greeting, or error"""


def resolve_references_prompt(question: str, history: str) -> str:
    turns = re.split(r'\n(?=User:|Assistant:)', history.strip())
    recent = " | ".join(t.replace("\n", " ") for t in turns[-4:])
    sliced_history = recent[:400]
    return f"""Rewrite this question by replacing
pronouns like "it", "this", "that", "the same" with
the actual subject from the conversation history.
Keep the question natural and concise.
Reply with ONLY the rewritten question. Max 15 words.

Examples:
History: "User: what is BERT | Assistant: BERT is..."
Question: "what is the accuracy of it in MNLI"
Rewritten: "what is the accuracy of BERT in MNLI"

History: "User: explain CNN architecture | Assistant: CNN..."
Question: "show me the figure for it"
Rewritten: "show me the CNN architecture figure"

History:
{sliced_history}

Question: "{question}"
Rewritten:"""


def normalize_image_query_prompt(question: str, context_line: str) -> str:
    return f"""You are a search query normalizer for an
academic paper image database.
Expand only clear abbreviations. Do NOT add extra domain words.
Keep the core subject exactly as stated.
If the query contains "it", "this", "that", or "the topic",
resolve the reference using the recent conversation context.
Respond with ONLY the rewritten query, nothing else. Max 8 words.

Examples:
"cnn arch image" → "CNN architecture diagram"
"bert fig" → "BERT architecture figure"
"show me the figure for it" (context: BERT arch) → "BERT architecture figure"
"give me the diagram for it" (context: confusion matrix) → "confusion matrix EEG"
"conf matrix eeg" → "confusion matrix EEG"
"electrode img" → "electrode placement diagram"
"algo chart" → "algorithm comparison chart"

{context_line}Query: "{question}"
Rewritten:"""


def table_summary_prompt(question: str, history: str, table_context: str) -> str:
    return (
        f"Answer this question in ONE sentence using only the table data below. "
        f"Quote the exact number. Be direct.\n\n"
        f"{history + chr(10) + chr(10) if history else ''}"
        f"Question: {question}\n\n"
        f"Table data:\n{table_context}\n\n"
        f"One-sentence answer:"
    )


def generate_answer_prompt(question: str, history: str, context: str) -> str:
    system_prompt = (
        "You are PaperBot, an academic research assistant with access to "
        "5 indexed research papers: BERT, RoBERTa, a BCI/EEG paper, and "
        "two sentiment analysis papers. Answer using the provided context. "
        "Be direct and specific. For explanation questions, answer as text. "
        "Do not promise to show or fetch an image unless the current state "
        "already has image_path or needs_image=True."
    )
    return (
        f"{system_prompt}\n\n"
        f"{history + chr(10) + chr(10) if history else ''}"
        f"Question:\n{question}\n\n"
        f"PDF context:\n{context}\n\n"
        "Answer:"
    )


def content_suggestions_prompt(question: str, answer_summary: str) -> str:
    return f"""Decide if there is likely a relevant TABLE
or IMAGE available in the academic papers for this question.

Rules — be strict but practical:
- has_image = true if the topic has a known diagram,
  architecture, formula, workflow, or visual in papers:
  "what is BERT" → has_image=true (architecture diagrams exist)
  "what is CNN" → has_image=true
  "what is masked language modeling" → has_image=true
  "how does attention work" → has_image=true
  "what is softmax" → has_image=true
  "what is the BCI system" → has_image=true
  "what is sentiment analysis" → has_image=false (no diagram)
  "what is transfer learning" → has_image=false
  "what is the difference between BERT BASE and BERT LARGE" → has_image=false (numerical difference, no comparison figure)
  "how many parameters does BERT have" → has_image=false

- has_table = true if the topic has numeric benchmark
  results, accuracy scores, or performance comparisons:
  "what was BERT accuracy on MNLI" → has_table=true
  "what is the F1 score of SVM" → has_table=true
  "which model performed best" → has_table=true
  "what is the difference between BERT BASE and BERT LARGE" → has_table=true (performance comparison table exists)
  "what algorithms were compared" → has_table=true
  "what is BERT" → has_table=false (general explanation)
  "how does BERT work" → has_table=false
  "what is sentiment analysis" → has_table=false

- Both can be true:
  "what is the BCI system performance" → both true
  "what are the BERT results" → has_table=true, has_image=true

Question: "{question}"
Answer summary: "{answer_summary}"
"""


def select_compatible_image_prompt(question: str, candidates: list[dict]) -> str:
    candidate_lines = []
    for i, c in enumerate(candidates):
        candidate_lines.append(
            f"[{i}] region_type={c.get('region_type','figure')} "
            f"caption={c.get('caption','')[:120]}"
        )
    candidates_text = "\n".join(candidate_lines)
    return (
        f"You are selecting the most relevant image for a user query "
        f"from a list of candidates retrieved from academic papers.\n\n"
        f"User query: \"{question}\"\n\n"
        f"Candidates:\n{candidates_text}\n\n"
        f"Rules:\n"
        f"- If query asks for a formula/equation, prefer region_type=isolate_formula\n"
        f"- If query asks for architecture/diagram/workflow, prefer region_type=figure\n"
        f"- If query asks for a confusion matrix, prefer candidates whose caption "
        f"mentions matrix or confusion\n"
        f"- If no candidate is relevant to the query, reply with: NONE\n\n"
        f"Reply with ONLY the index number of the best candidate, "
        f"or NONE if nothing matches.\n"
        f"Examples: 0  or  3  or  NONE"
    )
