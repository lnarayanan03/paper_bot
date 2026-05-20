"""LangGraph RAG workflow definitions for PaperBot."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from prompts import (
    classification_prompt,
    resolve_references_prompt,
    normalize_image_query_prompt,
    table_summary_prompt,
    generate_answer_prompt,
    content_suggestions_prompt,
    select_compatible_image_prompt,
)
from schema import ContentSuggestions, IntentClassification

load_dotenv()


TOP_K = 5
IMAGE_TOP_K = int(os.getenv("IMAGE_TOP_K", "10"))
IMAGE_MIN_SCORE = float(os.getenv("IMAGE_MIN_SCORE", "0.30"))
BASE_DIR = Path(__file__).resolve().parent

_EMBEDDING_MODEL: Any | None = None
_LLM_CLIENT: Any | None = None
_QDRANT_CLIENT: Any | None = None


class GraphState(TypedDict, total=False):
    question: str
    conversation_history: str
    intent: str  # "image" | "table" | "greeting" | "text"
    needs_table_augment: bool
    greeting_subtype: str
    chunks: list[dict]
    answer: str
    needs_image: bool
    has_relevant_image: bool
    has_relevant_table: bool
    image_source: str
    image_page: int
    image_path: str | None
    image_score: float
    image_caption: str
    image_region_type: str
    image_detector: str
    image_detector_confidence: float
    image_bbox: list
    session_id: str


def _get_embedding_model() -> Any:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from fastembed import TextEmbedding

        _EMBEDDING_MODEL = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDING_MODEL


def _embed_text(text: str) -> list[float]:
    return list(_get_embedding_model().embed([text]))[0].tolist()


def _get_qdrant_client() -> Any:
    global _QDRANT_CLIENT
    from qdrant_client import QdrantClient

    if _QDRANT_CLIENT is None:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        _QDRANT_CLIENT = QdrantClient(host=qdrant_host, port=qdrant_port)
    return _QDRANT_CLIENT


def _get_llm() -> Any:
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        _LLM_CLIENT = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY", "")
        )
    return _LLM_CLIENT


def _message_content(response: Any) -> str:
    return str(getattr(response, "content", response))


def _retrieve_hits(
    client: Any,
    collection_name: str,
    query_vector: list[float],
    limit: int = TOP_K,
) -> list[Any]:
    if hasattr(client, "search"):
        return client.search(collection_name=collection_name, query_vector=query_vector, limit=limit)

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )
    return list(response.points)


def _hit_to_chunk(hit: Any) -> dict:
    payload = getattr(hit, "payload", None) or {}
    score = getattr(hit, "score", None)
    chunk = dict(payload)
    if score is not None:
        chunk["score"] = score
    return chunk


def _format_context(chunks: list[dict]) -> str:
    formatted_chunks = []
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown source")
        page = chunk.get("page", "unknown page")
        text = chunk.get("text", "")
        formatted_chunks.append(f"[{index}] Source: {source}, page {page}\n{text}")
    return "\n\n".join(formatted_chunks)


def _clean_markdown_table(markdown: str | None) -> str:
    if not markdown:
        return ""
    lines = []
    for line in str(markdown).splitlines():
        clean = " ".join(line.split())
        if not clean or clean == "NO_TABLE":
            continue
        lines.append(clean)
    return "\n".join(lines)


def _is_non_figure_image_record(caption: str, search_text: str = "") -> bool:
    combined = " ".join(f"{caption} {search_text}".lower().split())
    negative_phrases = [
        "no figure",
        "does not contain a figure",
        "no visual figure",
        "blank page",
        "no visible figures",
        "no visible visual elements",
        "title page with text",
        "not a figure",
        "text block with no figures",
        "page from a journal article with no diagram",
        "does not contain a visual figure",
        "no visible diagram",
        "author profile image",
        "journal header",
        "title page/banner",
        "title page banner",
        "logo only",
        "page header",
        "page footer",
        "header/footer",
        "text block with no research figure",
    ]
    return any(phrase in combined for phrase in negative_phrases)


def _no_matching_image_response() -> dict:
    return {
        "answer": "No matching figure/image was found.",
        "needs_image": False,
        "image_path": None,
        "has_relevant_image": False,
        "has_relevant_table": False,
    }


# ── Nodes ─────────────────────────────────────────────────────────────────────


def classify_intent(state: GraphState) -> dict:
    question = state.get("question", "").strip()
    if not question:
        return {"intent": "error"}
    try:
        chain = _get_llm().with_structured_output(IntentClassification)
        result = chain.invoke(classification_prompt(question))
        return {
            "intent": result.intent,
            "needs_table_augment": result.needs_table_augment,
            "greeting_subtype": result.greeting_subtype,
        }
    except Exception as e:
        print(f"[classify_intent error] {e}")
        return {"intent": "error"}


def error_node(state: GraphState) -> dict:
    return {
        "answer": (
            "I could not confidently understand that request. "
            "Please rephrase it as a question about text, tables, or figures."
        )
    }


def greeting_node(state: GraphState) -> dict:
    is_farewell = state.get("greeting_subtype", "greeting") == "farewell"
    if is_farewell:
        msg = ("Goodbye! Hope PaperBot helped you understand "
               "the research papers. Come back anytime! 👋")
    else:
        msg = ("Hi! I am PaperBot 🤖 I can help you explore "
               "5 indexed research papers on BERT, RoBERTa, "
               "BCI/EEG, and Sentiment Analysis. "
               "Ask me anything or say 'show me a figure'!")
    return {"answer": msg}


def _is_garbage_table(markdown: str) -> bool:
    """
    Returns True if markdown table is pdfplumber noise —
    figure annotation tokens mistaken for table cells.
    """
    if not markdown:
        return True
    lines = [l for l in markdown.splitlines() if "|" in l]
    if len(lines) < 2:
        return True

    cells = []
    for line in lines:
        if "---" in line:
            continue
        parts = [p.strip() for p in line.split("|") if p.strip()]
        cells.extend(parts)

    if not cells:
        return True

    garbage_tokens = {
        "[sep]", "[cls]", "[mask]", "e e", "t t",
        "t n", "e n", "tok", "emb", "...",
    }
    garbage_prefixes = ("e e", "t t", "t n", "e n")
    garbage_count = sum(
        1 for c in cells
        if c.lower() in garbage_tokens
        or c.lower().startswith(garbage_prefixes)
        or (len(c) <= 2 and not c.replace(".", "").isdigit())
        or c.lower().startswith("e\n")
        or c.lower().startswith("t\n")
    )

    if len(cells) > 0 and garbage_count / len(cells) > 0.40:
        return True

    has_numeric = any(
        bool(re.search(r"\d+\.?\d*", c)) for c in cells
    )
    if not has_numeric:
        return True

    return False


def _resolve_query_references(
    question: str,
    history: str,
) -> str:
    """
    Resolve pronouns like 'it', 'this', 'that' using
    conversation history before embedding for retrieval.
    """
    if not history:
        return question

    try:
        prompt = resolve_references_prompt(question, history)
        response = _get_llm().invoke(prompt)
        resolved = response.content.strip()
        if resolved and len(resolved) < 200:
            print(f"  [resolve] '{question}' → '{resolved}'")
            return resolved
    except Exception as e:
        print(f"[resolve_references error] {e}")
    return question


def retrieve(state: GraphState) -> dict:
    collection_text = os.getenv("COLLECTION_TEXT", "paperbot_chunks")
    try:
        history = state.get("conversation_history", "")
        resolved = _resolve_query_references(
            state["question"], history
        )
        query_vector = _embed_text(resolved)
        hits = _retrieve_hits(
            _get_qdrant_client(),
            collection_text,
            query_vector
        )
    except Exception as e:
        resolved = state.get("question", "")
        hits = []
    chunks = [_hit_to_chunk(hit) for hit in hits]

    if state.get("needs_table_augment", False):
        try:
            collection_tables = os.getenv(
                "COLLECTION_TABLES", "paperbot_tables"
            )
            table_hits = _retrieve_hits(
                _get_qdrant_client(),
                collection_tables,
                _embed_text(resolved),
                limit=3,
            )
            for h in table_hits:
                payload = getattr(h, "payload", None) or {}
                md = _clean_markdown_table(
                    payload.get("markdown", "")
                )
                if not md:
                    continue
                if _is_garbage_table(md):
                    continue
                chunks.append({
                    "source": payload.get("source", ""),
                    "page": payload.get("page", 0),
                    "text": f"Table data:\n{md}",
                    "markdown": md,
                    "score": getattr(h, "score", 0),
                    "type": "table",
                })
        except Exception as e:
            print(f"[retrieve table augment error] {e}")

    return {"chunks": chunks}


def retrieve_tables(state: GraphState) -> dict:
    collection_tables = os.getenv("COLLECTION_TABLES", "paperbot_tables")
    history = state.get("conversation_history", "")
    resolved = _resolve_query_references(state["question"], history)
    query_vector = _embed_text(resolved)
    hits = _retrieve_hits(
        _get_qdrant_client(),
        collection_tables,
        query_vector,
        limit=TOP_K,
    )
    chunks = []
    for h in hits:
        payload = getattr(h, "payload", None) or {}
        markdown = _clean_markdown_table(payload.get("markdown", ""))
        if not markdown:
            continue
        if _is_garbage_table(markdown):
            continue
        chunks.append(
            {
                "source": payload.get("source", ""),
                "page": payload.get("page", 0),
                "text": markdown,
                "markdown": markdown,
                "score": getattr(h, "score", 0),
                "type": "table",
            }
        )
    return {"chunks": chunks}


def _check_content_suggestions(
    question: str,
    answer: str,
) -> dict:
    prompt = content_suggestions_prompt(question, answer[:200])

    try:
        chain = _get_llm().with_structured_output(ContentSuggestions)
        result = chain.invoke(prompt)
        return {
            "has_relevant_image": result.has_image,
            "has_relevant_table": result.has_table,
        }
    except Exception as e:
        print(f"[content_suggestions error] {e}")
        return {
            "has_relevant_image": False,
            "has_relevant_table": False,
        }


def generate_answer(state: GraphState) -> dict:
    question = state.get("question", "")
    intent = state.get("intent", "text")
    history = state.get("conversation_history", "")

    if intent == "image":
        answer = state.get("answer") or "Here is the figure from the paper."
        return {
            "answer": answer,
            "has_relevant_image": False,
            "has_relevant_table": False,
        }

    if intent == "table":
        # Return raw markdown tables directly — no LLM paraphrasing
        chunks = state.get("chunks", [])
        table_chunks = [
            c for c in chunks
            if _clean_markdown_table(c.get("markdown", ""))
        ]
        if table_chunks:
            table_chunks.sort(key=lambda c: float(c.get("score", 0) or 0), reverse=True)
            top_score = float(table_chunks[0].get("score", 0) or 0)
            min_score = max(0.35, top_score - 0.06)
            table_chunks = [
                c for c in table_chunks
                if float(c.get("score", 0) or 0) >= min_score
            ][:2]
            if not table_chunks:
                return {
                    "answer": "No relevant table was found for that request.",
                    "has_relevant_image": False,
                    "has_relevant_table": False,
                }
            # Get one-sentence LLM answer first
            table_context = "\n\n".join(
                _clean_markdown_table(c.get("markdown", "")) for c in table_chunks
            )
            summary_prompt = table_summary_prompt(question, history, table_context)
            try:
                summary = _message_content(_get_llm().invoke(summary_prompt)).strip()
            except Exception as e:
                summary = ""

            parts = []
            for c in table_chunks:
                src = c.get("source", "")
                page = c.get("page", "")
                md = _clean_markdown_table(c.get("markdown", ""))
                parts.append(f"**{src} · p.{page}**\n\n{md}")

            table_block = "\n\n---\n\n".join(parts)
            full_answer = f"{summary}\n\n{table_block}" if summary else table_block
            image_suggestion = _check_content_suggestions(question, full_answer)
            return {
                "answer": full_answer,
                "has_relevant_image": image_suggestion.get(
                    "has_relevant_image", False
                ),
                "has_relevant_table": False,
            }
        return {
            "answer": "No relevant table was found for that request.",
            "has_relevant_image": False,
            "has_relevant_table": False,
        }
    else:
        context = _format_context(state.get("chunks", []))
        prompt = generate_answer_prompt(question, history, context)
        try:
            response = _get_llm().invoke(prompt)
            answer = _message_content(response)
        except Exception as e:
            print(f"[generate_answer error] {e}")
            answer = "I could not generate an answer."
        suggestions = _check_content_suggestions(question, answer)
        return {"answer": answer, **suggestions}


# ── Routing ───────────────────────────────────────────────────────────────────


def route_after_classify(state: GraphState) -> str:
    intent = state.get("intent")
    if intent == "error":
        return "error"
    if intent == "greeting":
        return "greeting"
    if intent == "table":
        return "retrieve_tables"
    if intent == "image":
        return "image_decision"
    return "retrieve"


def _image_hit_to_dict(hit: Any) -> dict:
    payload = getattr(hit, "payload", None) or {}
    return {
        "source": str(payload.get("source", "")),
        "page": payload.get("page", 0),
        "image_path": str(payload.get("image_path", "")),
        "score": getattr(hit, "score", 0),
        "caption": str(payload.get("caption", "")),
        "search_text": str(payload.get("search_text", "")),
        "caption_context": str(payload.get("caption_context", "")),
        "region_type": str(payload.get("region_type", "figure")),
        "detector": str(payload.get("detector", "")),
        "detector_confidence": payload.get("detector_confidence", 0),
        "bbox": payload.get("bbox", []),
    }


def _search_images_collection(question: str) -> dict:
    """Find the most relevant image using pure vector similarity."""
    collection_images = os.getenv("COLLECTION_IMAGES", "paperbot_images")
    try:
        print(f"[image_search] query='{question}'")
        query_vector = _embed_text(question)
        hits = _retrieve_hits(
            _get_qdrant_client(),
            collection_images,
            query_vector,
            limit=IMAGE_TOP_K,
        )
        if not hits:
            return {}
        hit_dicts = [_image_hit_to_dict(hit) for hit in hits]
        result = dict(hit_dicts[0])
        result["_hits"] = hit_dicts
        return result

    except Exception as e:
        return {}


def _normalize_image_query(
    question: str,
    history: str = "",
) -> str:
    """Use LLM to expand abbreviations and normalize image queries."""
    try:
        context_line = ""
        if history:
            turns = re.split(r'\n(?=User:|Assistant:)', history.strip())
            recent_turns = turns[-4:]
            recent = " | ".join(t.replace("\n", " ") for t in recent_turns)
            context_line = f"Recent conversation: {recent[:400]}\n"

        prompt = normalize_image_query_prompt(question, context_line)
        response = _get_llm().invoke(prompt)
        normalized = response.content.strip()
        if normalized and len(normalized) < 100:
            print(f"  [normalize] '{question}' → '{normalized}'")
            return normalized
    except Exception as e:
        print(f"[normalize_image_query error] {e}")
    return question


def image_decision(state: GraphState) -> dict:
    question = state.get("question", "")
    history = state.get("conversation_history", "")
    normalized = _normalize_image_query(question, history)
    best = _search_images_collection(normalized)

    if not best:
        return _no_matching_image_response()

    candidates = best.get("_hits", [best])
    compatible = None
    try:
        prompt = select_compatible_image_prompt(question, candidates)
        response = _get_llm().invoke(prompt)
        raw = response.content.strip()
        if raw.upper() != "NONE":
            idx = int(raw)
            if 0 <= idx < len(candidates):
                compatible = candidates[idx]
    except Exception as e:
        print(f"[image_compatibility error] {e}")
        compatible = candidates[0] if candidates else None

    if compatible is None:
        return _no_matching_image_response()

    best = compatible

    score = float(best.get("score", 0) or 0)
    caption = str(best.get("caption", ""))
    search_text = str(best.get("search_text", ""))
    if score < IMAGE_MIN_SCORE or _is_non_figure_image_record(
        caption, search_text
    ):
        return _no_matching_image_response()

    source = str(best.get("source", ""))
    stored_path = str(best.get("image_path", ""))
    final_path = (
        stored_path
        if stored_path and Path(stored_path).exists()
        else None
    )
    region_type = str(best.get("region_type", "figure"))
    detector = str(best.get("detector", ""))
    try:
        page = int(best.get("page", 0))
    except (TypeError, ValueError):
        page = 0
    print(
        f"[image_decision] selected source={source} page={page} "
        f"score={score} region_type={region_type} detector={detector}"
    )
    return {
        "needs_image": True,
        "image_source": source,
        "image_page": page,
        "image_path": final_path,
        "image_score": score,
        "image_caption": caption,
        "image_region_type": region_type,
        "image_detector": detector,
        "image_detector_confidence": float(best.get("detector_confidence", 0) or 0),
        "image_bbox": best.get("bbox", []),
        "has_relevant_image": True,
        "has_relevant_table": False,
    }


# ── Graph assembly ─────────────────────────────────────────────────────────────


def build_graph() -> Any:
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, StateGraph

    graph = StateGraph(GraphState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("error", error_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("retrieve", retrieve)
    graph.add_node("retrieve_tables", retrieve_tables)
    graph.add_node("image_decision", image_decision)
    graph.add_node("generate_answer", generate_answer)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "error":           "error",
            "greeting":        "greeting",
            "retrieve":        "retrieve",
            "retrieve_tables": "retrieve_tables",
            "image_decision":  "image_decision",
        }
    )

    graph.add_edge("error", END)
    graph.add_edge("greeting", END)
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("retrieve_tables", "generate_answer")
    graph.add_edge("image_decision", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile(checkpointer=MemorySaver())


app = build_graph()


def _get_graph() -> Any:
    return app


def _load_conversation_history(session_id: str) -> tuple[Any | None, str]:
    try:
        import redis

        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        history_key = f"paperbot:history:{session_id}"
        client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        raw_messages = client.lrange(history_key, -10, -1)
    except Exception as e:
        return None, ""

    lines = []
    for item in raw_messages:
        role, _, content = str(item).partition(":")
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
    if not lines:
        return client, ""
    return client, "Previous conversation:\n" + "\n".join(lines)


def _store_conversation_pair(
    client: Any | None,
    session_id: str,
    question: str,
    answer: str,
) -> None:
    if client is None:
        return
    try:
        history_key = f"paperbot:history:{session_id}"
        client.rpush(history_key, f"user:{question}")
        client.rpush(history_key, f"assistant:{answer}")
        client.ltrim(history_key, -10, -1)
    except Exception as e:
        pass


def run_graph(question: str, session_id: str, human_response: dict | None = None) -> dict:
    """Run the PaperBot RAG graph."""
    graph = _get_graph()
    config = {"configurable": {"thread_id": session_id}}
    redis_client, conversation_history = _load_conversation_history(session_id)

    initial_state: GraphState = {
        "question": question,
        "conversation_history": conversation_history,
        "session_id": session_id,
        "intent": "",
        "needs_table_augment": False,
        "greeting_subtype": "greeting",
        "chunks": [],
        "answer": "",
        "needs_image": False,
        "has_relevant_image": False,
        "has_relevant_table": False,
        "image_source": "",
        "image_page": 0,
        "image_path": None,
    }
    result = graph.invoke(initial_state, config=config)

    final_state = dict(result or {})
    final_state["interrupt_pending"] = False
    final_state.setdefault("answer", "")
    final_state.setdefault("needs_image", False)
    final_state.setdefault("has_relevant_image", False)
    final_state.setdefault("has_relevant_table", False)
    final_state.setdefault("image_path", None)
    final_state.setdefault("chunks", [])
    _store_conversation_pair(
        redis_client,
        session_id,
        question,
        str(final_state.get("answer", "")),
    )
    return final_state
