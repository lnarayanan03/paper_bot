"""PDF ingestion pipeline for PaperBot."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

VECTOR_SIZE = 384
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
DL_CONF_THRESHOLD = 0.5
DL_RENDER_DPI = 150

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "data" / "pdfs"
IMAGE_DIR = BASE_DIR / "data" / "images"

_DL_MODEL: Any | None = None


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text with a simple character sliding window."""
    clean_text = " ".join(text.split())
    if not clean_text:
        return []

    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(clean_text), step):
        chunk = clean_text[start : start + chunk_size]
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(clean_text):
            break

    return chunks


def recreate_collection(client: Any, collection_name: str) -> None:
    """Delete a collection if it exists, then create it with PaperBot vector settings."""
    from qdrant_client.models import Distance, VectorParams

    try:
        client.delete_collection(collection_name=collection_name)
    except Exception:
        pass

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def embed_texts(model: Any, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    embeddings = list(model.embed(texts))
    return [embedding.tolist() for embedding in embeddings]


def clean_table_cell(value: Any) -> str:
    if value is None:
        return ""
    clean = re.sub(r"\s+", " ", str(value)).strip()
    return clean.replace("|", r"\|")


def table_to_markdown(table: list[list[Any]]) -> str:
    if not table:
        return ""
    column_count = max((len(row) for row in table if row), default=0)
    if column_count == 0:
        return ""

    rows = []
    for row in table:
        cells = [clean_table_cell(cell) for cell in (row or [])]
        cells.extend([""] * (column_count - len(cells)))
        rows.append(cells[:column_count])

    header = rows[0]
    separator = ["---"] * column_count
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def clean_extracted_markdown_table(markdown: str | None) -> str | None:
    if not markdown:
        return None
    lines = []
    for line in markdown.splitlines():
        clean = " ".join(line.split())
        if not clean or clean == "NO_TABLE":
            continue
        if "|" in clean:
            lines.append(clean)
    if len(lines) < 2:
        return None
    if not any("---" in line for line in lines):
        return None
    return "\n".join(lines)


def _get_dl_model(model_path: str):
    global _DL_MODEL
    if _DL_MODEL is None:
        from doclayout_yolo import YOLOv10

        _DL_MODEL = YOLOv10(model_path)
    return _DL_MODEL


def _box_scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _box_values(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if len(value) > 0 and isinstance(value[0], list):
        value = value[0]
    return [float(v) for v in value]


def detect_figure_regions(page: Any, page_num: int, pdf_stem: str) -> list[dict]:
    model_path = os.getenv("PAPERBOT_DL_MODEL_PATH", "")
    if not model_path:
        return []

    import fitz

    scale = DL_RENDER_DPI / 72
    tmp_path = Path(f"/tmp/paperbot_dl_{pdf_stem}_page_{page_num}_{uuid4().hex}.png")
    try:
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        pix.save(str(tmp_path))

        model = _get_dl_model(model_path)
        results = model.predict(
            str(tmp_path),
            imgsz=1024,
            conf=DL_CONF_THRESHOLD,
            verbose=False,
        )
        names = getattr(model, "names", {}) or {}
        regions = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(_box_scalar(box.cls))
                if isinstance(names, dict):
                    label = str(names.get(cls_id, cls_id))
                else:
                    label = str(names[cls_id] if cls_id < len(names) else cls_id)
                if label not in {"figure", "isolate_formula"}:
                    continue

                x0, y0, x1, y1 = _box_values(box.xyxy)
                score = _box_scalar(box.conf)
                regions.append(
                    {
                        "label": label,
                        "score": score,
                        "bbox_page": [
                            x0 / scale,
                            y0 / scale,
                            x1 / scale,
                            y1 / scale,
                        ],
                        "detector": "dl_layout",
                        "region_type": label,
                    }
                )
        return sorted(regions, key=lambda r: float(r["score"]), reverse=True)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _bbox_distance(a: list[float] | tuple[float, float, float, float], b: Any) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0)
    dy = max(by0 - ay1, ay0 - by1, 0)
    return (dx * dx + dy * dy) ** 0.5


def extract_nearest_caption_context(
    page: Any,
    region_bbox: list[float] | tuple[float, float, float, float],
    region_type: str,
    max_chars: int = 350,
) -> str:
    marker_pattern = re.compile(r"\b(fig\.?|figure|table|equation|formula|caption)\b", re.I)
    page_h = page.rect.height
    max_distance = page_h * (0.25 if region_type in ("formula", "isolate_formula") else 0.35)
    candidates = []

    for block in page.get_text("blocks"):
        if len(block) < 5:
            continue
        text = " ".join(str(block[4]).split())
        if not text:
            continue
        block_bbox = (float(block[0]), float(block[1]), float(block[2]), float(block[3]))
        distance = _bbox_distance(region_bbox, block_bbox)
        overlaps_x = not (block_bbox[2] < region_bbox[0] or block_bbox[0] > region_bbox[2])
        has_marker = bool(marker_pattern.search(text))
        if not has_marker:
            continue
        if distance > max_distance and not overlaps_x:
            continue
        candidates.append((0 if has_marker else 1, distance, text))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2][:max_chars]


def _search_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 4
    }


def should_keep_caption_context_for_region(
    caption_context: str,
    vision_caption: str,
    region_type: str,
) -> bool:
    if not caption_context:
        return False
    if region_type not in ("formula", "isolate_formula"):
        return True

    context = caption_context.lower()
    caption = vision_caption.lower()
    formula_indicators = (
        "formula", "equation", "function", "activation", "metric",
        "normalization", "frequency", "precision", "recall", "score",
    )
    unrelated_visual_terms = (
        "figure architecture", "three-tier", "three tier", "block diagram",
        "system architecture", "module", "component", "client",
        "mouse control", "model architecture",
    )
    context_tokens = _search_tokens(context)
    caption_tokens = _search_tokens(caption)
    overlap = context_tokens & caption_tokens
    strong_overlap = len(overlap) >= 2 or (
        bool(caption_tokens) and len(overlap) / len(caption_tokens) >= 0.25
    )

    has_formula_indicator = any(term in context for term in formula_indicators)
    has_unrelated_visual = any(term in context for term in unrelated_visual_terms)
    visual_terms_in_caption = any(term in caption for term in unrelated_visual_terms)

    if has_unrelated_visual and not visual_terms_in_caption and not strong_overlap:
        return False
    return has_formula_indicator or strong_overlap


def build_image_search_text(
    pdf_path: Path,
    page_num: int,
    vision_caption: str,
    region_type: str = "figure",
    caption_context: str = "",
    detector: str = "",
) -> str:
    source_text = re.sub(r"[_\s]+", " ", pdf_path.stem).strip()
    if not should_keep_caption_context_for_region(
        caption_context,
        vision_caption,
        region_type,
    ):
        caption_context = ""

    parts = [
        f"source document: {source_text}",
        f"region type: {region_type}",
        f"detector: {detector}" if detector else "",
        f"local caption context: {caption_context}" if caption_context else "",
        f"vision caption: {vision_caption}",
    ]
    search_text = " ".join(part for part in parts if part)
    return " ".join(search_text.split())[:1000]


def is_non_figure_caption(caption: str) -> bool:
    clean = " ".join(caption.lower().split())
    if not clean:
        return True
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
        "portrait photograph",
        "cropped portrait",
        "photograph of a person",
        "headshot",
        "profile photo",
        "person with",
        "journal cover",
        "cover page",
        "table of contents",
        "abstract only",
        "references list",
        "bibliography",
    ]
    return any(phrase in clean for phrase in negative_phrases)


def _anthropic_vision_text(
    image_path: str,
    prompt: str,
    api_key: str,
    max_tokens: int = 500,
) -> str:
    import base64
    import mimetypes
    import anthropic

    img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
    media_type = mimetypes.guess_type(image_path)[0]
    if media_type not in ("image/png", "image/jpeg"):
        media_type = "image/png"

    client = anthropic.Anthropic(api_key=api_key)
    model = os.getenv("ANTHROPIC_VISION_MODEL", "claude-sonnet-4-6")
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": img_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )
    except Exception as e:
        status_code = getattr(e, "status_code", None)
        if status_code == 404 or e.__class__.__name__ == "NotFoundError":
            raise RuntimeError(
                "Anthropic model not found. Set ANTHROPIC_VISION_MODEL to a "
                "model available for your account, e.g. claude-sonnet-4-6."
            ) from e
        raise

    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            return str(getattr(block, "text", ""))
        if isinstance(block, dict) and block.get("type") == "text":
            return str(block.get("text", ""))
    return ""


def anthropic_describe_only(image_path: str, api_key: str) -> str:
    prompt = (
        "You are describing a figure from an academic research paper "
        "for a semantic search index. Your description will be used "
        "as a search vector — it must be specific enough to distinguish "
        "this figure from other similar figures in the same paper. "
        "\n\n"
        "Rules:\n"
        "1. Start with the figure type: architecture diagram, "
        "workflow diagram, confusion matrix, electrode placement diagram, "
        "input representation diagram, line chart, bar chart, "
        "formula block, or comparison diagram.\n"
        "2. Name the specific model, algorithm, or system shown "
        "if labels are visible. For example: BERT pre-training, "
        "BERT fine-tuning tasks, CNN layer architecture, "
        "EEG electrode placement, sentiment analysis workflow.\n"
        "3. Describe the dominant layout: parallel stacks, "
        "sequential pipeline, grid, head map, multi-panel, "
        "single stack, left-to-right flow.\n"
        "4. Mention unique visible labels or text that distinguish "
        "this figure: e.g. NSP and Mask LM outputs, "
        "MNLI/NER/SQuAD task labels, confusion matrix class names, "
        "electrode position names like AF3 Fz C3, "
        "layer names like Conv1D MaxPool Flatten Dense.\n"
        "5. For pre-training figures mention: masked tokens, "
        "NSP, sentence pairs, MLM.\n"
        "6. For fine-tuning figures mention: the specific downstream "
        "tasks shown like MNLI, NER, SQuAD, SST-2, CoLA.\n"
        "7. Do NOT mention paper title, authors, page numbers, "
        "journal name, or citation text.\n"
        "8. Do NOT describe text paragraphs, reference lists, "
        "or page headers as figures.\n"
        "9. Respond in ONE specific sentence of 30-60 words. "
        "More specific = better retrieval."
    )
    try:
        return _anthropic_vision_text(image_path, prompt, api_key).strip()[:500]
    except Exception as e:
        print(f"  [Anthropic desc failed: {e}]")
        return ""


def anthropic_extract_table(page_image_path: str, api_key: str) -> str | None:
    prompt = (
        "This page from an academic paper may contain a results "
        "table. If you see a table with numeric data, extract it as a "
        "markdown table with all rows and columns. If there is no clear "
        "data table, respond with exactly: NO_TABLE"
    )
    try:
        content = _anthropic_vision_text(
            page_image_path,
            prompt,
            api_key,
            max_tokens=1000,
        ).strip()
        if content == "NO_TABLE":
            return None
        return content
    except Exception as e:
        print(f"  [Anthropic table failed: {e}]")
        return None


def ingest(only_pdf: str | None = None, wipe: bool = True) -> None:
    import fitz
    import pdfplumber
    import shutil
    from dotenv import load_dotenv
    from fastembed import TextEmbedding
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    load_dotenv()

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    collection_text = os.getenv("COLLECTION_TEXT", "paperbot_chunks")
    collection_images = os.getenv("COLLECTION_IMAGES", "paperbot_images")
    collection_tables = os.getenv("COLLECTION_TABLES", "paperbot_tables")

    if wipe:
        if IMAGE_DIR.exists():
            shutil.rmtree(IMAGE_DIR)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(host=qdrant_host, port=qdrant_port)
    if wipe:
        recreate_collection(client, collection_text)
        recreate_collection(client, collection_images)
        recreate_collection(client, collection_tables)

    model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    if only_pdf:
        pdf_paths = [p for p in pdf_paths if p.name == only_pdf]
        if not pdf_paths:
            print(f"PDF not found: {only_pdf}")
            return

    for pdf_path in pdf_paths:
        with fitz.open(pdf_path) as document, pdfplumber.open(str(pdf_path)) as plumber_pdf:
            total_pages = document.page_count

            for page_index in range(total_pages):
                page_num = page_index + 1
                print(f"Ingesting {pdf_path.name} page {page_num}/{total_pages}")

                page = document.load_page(page_index)
                page_text = page.get_text("text")

                # Text chunk pipeline — unchanged.
                chunks = chunk_text(page_text)
                chunk_vectors = embed_texts(model, chunks)
                chunk_points = [
                    PointStruct(
                        id=str(uuid4()),
                        vector=vector,
                        payload={
                            "source": pdf_path.name,
                            "page": page_num,
                            "chunk_index": chunk_index,
                            "text": chunk,
                        },
                    )
                    for chunk_index, (chunk, vector) in enumerate(zip(chunks, chunk_vectors))
                ]
                if chunk_points:
                    client.upsert(collection_name=collection_text, points=chunk_points)

                # Table extraction pipeline.
                plumber_page = plumber_pdf.pages[page_index]
                tables = plumber_page.extract_tables()

                if len(tables) == 0 and anthropic_api_key:
                    has_result_table = (
                        "table" in page_text.lower()
                        and page_num <= 12
                        and any(kw in page_text.lower() for kw in [
                            "accuracy", "f1", "precision", "recall", "mnli",
                            "squad", "glue", "score", "benchmark", "result"
                        ])
                        and bool(re.search(r"\d+\.\d+", page_text))
                    )
                    if has_result_table:
                        mat = fitz.Matrix(150/72, 150/72)
                        pix = page.get_pixmap(matrix=mat, alpha=False)
                        tmp_path = Path(f"/tmp/table_{pdf_path.stem}_{page_num}.png")
                        pix.save(str(tmp_path))
                        md = anthropic_extract_table(str(tmp_path), anthropic_api_key)
                        md = clean_extracted_markdown_table(md)
                        if md:
                            print(f"  [AnthropicTable] extracted table via Vision p.{page_num}")
                            table_text = f"Table from {pdf_path.name} page {page_num}:\n{md}"
                            table_vector = embed_texts(model, [table_text])[0]
                            table_point = PointStruct(
                                id=str(uuid4()),
                                vector=table_vector,
                                payload={
                                    "source": pdf_path.name,
                                    "page": page_num,
                                    "table_index": -1,
                                    "markdown": md,
                                    "text": table_text,
                                    "type": "table",
                                }
                            )
                            client.upsert(
                                collection_name=collection_tables,
                                points=[table_point]
                            )
                        if tmp_path.exists():
                            tmp_path.unlink()

                for table_index, table in enumerate(tables):
                    if not table:
                        continue
                    if len(table) < 2:
                        continue
                    if not table[0] or len(table[0]) < 2:
                        continue
                    header = [clean_table_cell(c) for c in table[0] if c]
                    real_cells = [h for h in header if len(h) > 1 and
                                  not h.startswith('E\n') and
                                  'T\n' not in h and
                                  '[SEP]' not in h and
                                  '[CLS]' not in h]
                    if len(real_cells) < 2:
                        continue

                    markdown_table = table_to_markdown(table)

                    table_text = f"Table from {pdf_path.name} page {page_num}:\n{markdown_table}"
                    table_vector = embed_texts(model, [table_text])[0]

                    table_point = PointStruct(
                        id=str(uuid4()),
                        vector=table_vector,
                        payload={
                            "source": pdf_path.name,
                            "page": page_num,
                            "table_index": table_index,
                            "markdown": markdown_table,
                            "text": table_text,
                        }
                    )
                    client.upsert(
                        collection_name=collection_tables,
                        points=[table_point]
                    )
                    print(f"  → Table p.{page_num} t.{table_index}: {real_cells[:3]}")

                # Image pipeline — DL region crop, Anthropic description only.
                safe_stem = pdf_path.stem.replace(" ", "_")

                dl_regions = detect_figure_regions(page, page_num, safe_stem)
                if not dl_regions:
                    print("  [ImageSkip] YOLO found no regions; skipping")
                    continue

                for region_index, region in enumerate(dl_regions):
                    region_type = str(region.get("region_type", region.get("label", "figure")))
                    image_path = (
                        IMAGE_DIR /
                        f"{safe_stem}_page_{page_num}_{region_type}_{region_index}.png"
                    )
                    x0, y0, x1, y1 = region["bbox_page"]
                    clip_rect = fitz.Rect(x0, y0, x1, y1)
                    mat = fitz.Matrix(300/72, 300/72)
                    pix = page.get_pixmap(matrix=mat, clip=clip_rect, alpha=False)
                    print(f"  [Crop] bbox=({x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f})")
                    pix.save(str(image_path))

                    if not anthropic_api_key:
                        print("  [ImageSkip] no ANTHROPIC_API_KEY; skipping image")
                        continue
                    caption = anthropic_describe_only(str(image_path), anthropic_api_key)

                    print(f"  → Vision desc: {caption[:80]}")
                    if is_non_figure_caption(caption):
                        print("  [ImageSkip] Vision says no figure; skipping image indexing")
                        continue

                    caption_context = extract_nearest_caption_context(
                        page,
                        region["bbox_page"],
                        region_type,
                    )
                    if not should_keep_caption_context_for_region(
                        caption_context,
                        caption,
                        region_type,
                    ):
                        caption_context = ""
                    search_text = build_image_search_text(
                        pdf_path,
                        page_num,
                        caption,
                        region_type=region_type,
                        caption_context=caption_context,
                        detector=str(region.get("detector", "")),
                    )
                    image_vector = embed_texts(model, [search_text])[0]
                    image_point = PointStruct(
                        id=str(uuid4()),
                        vector=image_vector,
                        payload={
                            "source": pdf_path.name,
                            "page": page_num,
                            "image_path": str(image_path),
                            "caption": caption,
                            "search_text": search_text,
                            "caption_context": caption_context,
                            "has_figures": True,
                            "bbox": region["bbox_page"],
                            "detector": region["detector"],
                            "detector_confidence": region["score"],
                            "region_type": region_type,
                        },
                    )
                    client.upsert(collection_name=collection_images, points=[image_point])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, default=None,
        help="Process only this PDF filename")
    parser.add_argument("--no-wipe", action="store_true",
        help="Skip recreating collections (append mode)")
    args = parser.parse_args()
    ingest(only_pdf=args.pdf, wipe=not args.no_wipe)
