# PaperBot — Multimodal RAG chatbot over academic research papers

## Demo screenshot placeholder

![PaperBot Demo](docs/demo.png)

## What is PaperBot

PaperBot is a multimodal retrieval-augmented chatbot for five indexed academic research papers: `1_BERT.pdf`, `2_RoBERTa.pdf`, `2_joels_paper_240428_181813.pdf`, `A Comparative Study of Sentiment Analysis Using NLP and Different Machine Learning Techniques on US Airline Twitter Data.pdf`, and `SENTIMENT ANALYSIS USING NATURAL LANGUAGE PROCESSING AND MACHINE LEARNING.pdf`.

Users can ask natural-language questions about definitions, methods, benchmark results, tables, figures, formulas, diagrams, and cited source pages from the papers.

The backend routes each request through a LangGraph workflow that classifies the intent, retrieves relevant text, tables, or images from Qdrant, and generates grounded answers with Groq plus Anthropic Vision captions for visual assets.

## Architecture overview

```text
User
  |
  v
Frontend (React/Vite)
  |
  v
FastAPI backend
  |
  v
LangGraph pipeline
  |
  v
Qdrant vector database
  |-- paperbot_chunks
  |-- paperbot_tables
  |-- paperbot_images
  |
  v
Groq LLM / Anthropic Vision
```

## Tech stack table

| Component | Technology |
|---|---|
| Backend | FastAPI, Uvicorn, Python, Pydantic, python-dotenv, httpx |
| Frontend | React, React DOM, Vite, @vitejs/plugin-react, axios, lucide-react, react-markdown, remark-gfm, Three.js through `window.THREE` |
| Vector DB | Qdrant, qdrant-client |
| LLM | Groq through LangChain ChatGroq using `llama-3.3-70b-versatile` |
| Vision | Anthropic Messages API using `ANTHROPIC_VISION_MODEL` |
| Embeddings | fastembed `TextEmbedding`, sentence-transformers/all-MiniLM-L6-v2, sentence-transformers |
| Image Detection | DocLayout-YOLO `YOLOv10`, Pillow, PyMuPDF rendering |
| PDF Parsing | PyMuPDF, pdfplumber |
| Graph Framework | LangGraph, LangChain |
| Conversation Memory | Redis, LangGraph `MemorySaver` checkpointer |
| Runtime | Python 3.11, Node.js 18+, Docker, Redis, browser-served Vite build |

## How it works — three pipelines

TEXT pipeline:

```text
PyMuPDF page text
  -> normalize whitespace
  -> 500-character chunks with 50-character overlap
  -> MiniLM embeddings through fastembed
  -> Qdrant collection: paperbot_chunks
```

TABLE pipeline:

```text
pdfplumber path
  -> extract_tables()
  -> clean cells
  -> convert table rows to Markdown
  -> embed "Table from source PDF page number: markdown table"
  -> Qdrant collection: paperbot_tables
```

```text
Anthropic Vision fallback path
  -> detect likely result-table pages from page text
  -> PyMuPDF render page image at 150 DPI
  -> Anthropic Vision extracts Markdown table
  -> clean extracted Markdown
  -> MiniLM embedding
  -> Qdrant collection: paperbot_tables
```

IMAGE pipeline:

```text
PyMuPDF render page
  -> DocLayout-YOLO region detection
  -> keep figure and isolate_formula boxes
  -> crop region at 300 DPI
  -> Anthropic Vision caption
  -> combine source, region type, detector, caption context, and vision caption
  -> MiniLM embedding
  -> Qdrant collection: paperbot_images
```

## Query flow

All questions enter the graph at `classify_intent`, which calls Groq with a structured `IntentClassification` schema.

Intent type: `text`

```text
classify_intent
  -> retrieve
  -> generate_answer
  -> END
```

The text route resolves pronouns from Redis-backed conversation history, embeds the resolved question with MiniLM, searches `paperbot_chunks`, optionally augments with `paperbot_tables` when `needs_table_augment=true`, and asks Groq to answer using retrieved context.

Intent type: `image`

```text
classify_intent
  -> image_decision
  -> generate_answer
  -> END
```

The image route normalizes short image requests, searches `paperbot_images`, asks Groq to select the compatible candidate, filters low-score or non-figure records, returns a PNG data URL when a stored crop exists, and uses `generate_answer` to produce the short image response.

Intent type: `table`

```text
classify_intent
  -> retrieve_tables
  -> generate_answer
  -> END
```

The table route searches `paperbot_tables`, removes empty and garbage Markdown tables, keeps the strongest matching table chunks, asks Groq for a one-sentence numeric summary, and returns the raw Markdown table with source filename and page number.

Intent type: `greeting`

```text
classify_intent
  -> greeting
  -> END
```

The greeting route answers directly for opening messages and farewell messages without retrieval.

Intent type: `error`

```text
classify_intent
  -> error
  -> END
```

The error route returns a rephrase prompt when the question is empty, unrelated, or cannot be confidently classified.

## Project structure

```text
paper_bot/
|-- README.md
|   Production project documentation.
|-- paperbot/
|   Main application directory.
|-- paperbot/backend/
|   FastAPI, LangGraph, ingestion, prompts, schemas, data, and built frontend.
|-- paperbot/backend/.agents/
|   Local agent metadata directory present in this checkout.
|-- paperbot/backend/.codex/
|   Local Codex metadata directory present in this checkout.
|-- paperbot/backend/.env
|   Local backend environment file with runtime secrets and settings.
|-- paperbot/backend/.env.example
|   Example environment variables for API keys, Qdrant, Redis, collections, image search, and DocLayout-YOLO weights.
|-- paperbot/backend/__pycache__/
|   Generated Python bytecode cache.
|-- paperbot/backend/__pycache__/graph.cpython-311.pyc
|   Generated bytecode for `graph.py` under Python 3.11.
|-- paperbot/backend/__pycache__/graph.cpython-312.pyc
|   Generated bytecode for `graph.py` under Python 3.12.
|-- paperbot/backend/__pycache__/ingest.cpython-311.pyc
|   Generated bytecode for `ingest.py` under Python 3.11.
|-- paperbot/backend/__pycache__/ingest.cpython-312.pyc
|   Generated bytecode for `ingest.py` under Python 3.12.
|-- paperbot/backend/__pycache__/main.cpython-311.pyc
|   Generated bytecode for `main.py` under Python 3.11.
|-- paperbot/backend/__pycache__/main.cpython-312.pyc
|   Generated bytecode for `main.py` under Python 3.12.
|-- paperbot/backend/__pycache__/post_ingest.cpython-311.pyc
|   Generated bytecode for a local post-ingest helper under Python 3.11.
|-- paperbot/backend/__pycache__/post_ingest.cpython-312.pyc
|   Generated bytecode for a local post-ingest helper under Python 3.12.
|-- paperbot/backend/__pycache__/prompts.cpython-311.pyc
|   Generated bytecode for `prompts.py` under Python 3.11.
|-- paperbot/backend/__pycache__/prompts.cpython-312.pyc
|   Generated bytecode for `prompts.py` under Python 3.12.
|-- paperbot/backend/__pycache__/schema.cpython-311.pyc
|   Generated bytecode for `schema.py` under Python 3.11.
|-- paperbot/backend/__pycache__/schema.cpython-312.pyc
|   Generated bytecode for `schema.py` under Python 3.12.
|-- paperbot/backend/__pycache__/visualize_graph.cpython-311.pyc
|   Generated bytecode for a local graph visualization helper under Python 3.11.
|-- paperbot/backend/__pycache__/visualize_graph.cpython-312.pyc
|   Generated bytecode for a local graph visualization helper under Python 3.12.
|-- paperbot/backend/data/
|   Source PDFs and generated image crops.
|-- paperbot/backend/data/images/
|   Generated figure and formula crops produced by ingestion.
|-- paperbot/backend/data/images/1_BERT_page_13_figure_0.png
|   Cropped BERT figure from page 13.
|-- paperbot/backend/data/images/1_BERT_page_15_figure_0.png
|   Cropped BERT figure from page 15.
|-- paperbot/backend/data/images/1_BERT_page_16_figure_0.png
|   Cropped BERT figure from page 16.
|-- paperbot/backend/data/images/1_BERT_page_3_figure_0.png
|   Cropped BERT figure from page 3.
|-- paperbot/backend/data/images/1_BERT_page_5_figure_0.png
|   Cropped BERT figure from page 5.
|-- paperbot/backend/data/images/2_joels_paper_240428_181813_page_3_figure_0.png
|   Cropped figure from Joel's paper page 3.
|-- paperbot/backend/data/images/2_joels_paper_240428_181813_page_3_figure_2.png
|   Cropped figure from Joel's paper page 3.
|-- paperbot/backend/data/images/2_joels_paper_240428_181813_page_3_isolate_formula_1.png
|   Cropped formula from Joel's paper page 3.
|-- paperbot/backend/data/images/2_joels_paper_240428_181813_page_4_figure_0.png
|   Cropped figure from Joel's paper page 4.
|-- paperbot/backend/data/images/2_joels_paper_240428_181813_page_4_figure_1.png
|   Cropped figure from Joel's paper page 4.
|-- paperbot/backend/data/images/2_joels_paper_240428_181813_page_4_isolate_formula_2.png
|   Cropped formula from Joel's paper page 4.
|-- paperbot/backend/data/images/2_joels_paper_240428_181813_page_4_isolate_formula_3.png
|   Cropped formula from Joel's paper page 4.
|-- paperbot/backend/data/images/2_joels_paper_240428_181813_page_4_isolate_formula_4.png
|   Cropped formula from Joel's paper page 4.
|-- paperbot/backend/data/images/A_Comparative_Study_of_Sentiment_Analysis_Using_NLP_and_Different_Machine_Learning_Techniques_on_US_Airline_Twitter_Data_page_3_figure_0.png
|   Cropped sentiment-analysis figure from page 3.
|-- paperbot/backend/data/images/A_Comparative_Study_of_Sentiment_Analysis_Using_NLP_and_Different_Machine_Learning_Techniques_on_US_Airline_Twitter_Data_page_3_figure_1.png
|   Cropped sentiment-analysis figure from page 3.
|-- paperbot/backend/data/images/A_Comparative_Study_of_Sentiment_Analysis_Using_NLP_and_Different_Machine_Learning_Techniques_on_US_Airline_Twitter_Data_page_4_isolate_formula_0.png
|   Cropped sentiment-analysis formula from page 4.
|-- paperbot/backend/data/images/A_Comparative_Study_of_Sentiment_Analysis_Using_NLP_and_Different_Machine_Learning_Techniques_on_US_Airline_Twitter_Data_page_4_isolate_formula_1.png
|   Cropped sentiment-analysis formula from page 4.
|-- paperbot/backend/data/images/A_Comparative_Study_of_Sentiment_Analysis_Using_NLP_and_Different_Machine_Learning_Techniques_on_US_Airline_Twitter_Data_page_4_isolate_formula_2.png
|   Cropped sentiment-analysis formula from page 4.
|-- paperbot/backend/data/images/A_Comparative_Study_of_Sentiment_Analysis_Using_NLP_and_Different_Machine_Learning_Techniques_on_US_Airline_Twitter_Data_page_4_isolate_formula_3.png
|   Cropped sentiment-analysis formula from page 4.
|-- paperbot/backend/data/images/SENTIMENT_ANALYSIS_USING_NATURAL_LANGUAGE_PROCESSING_AND_MACHINE_LEARNING_page_1_figure_0.png
|   Cropped sentiment-analysis figure from page 1.
|-- paperbot/backend/data/images/SENTIMENT_ANALYSIS_USING_NATURAL_LANGUAGE_PROCESSING_AND_MACHINE_LEARNING_page_3_figure_0.png
|   Cropped sentiment-analysis figure from page 3.
|-- paperbot/backend/data/images/SENTIMENT_ANALYSIS_USING_NATURAL_LANGUAGE_PROCESSING_AND_MACHINE_LEARNING_page_4_figure_0.png
|   Cropped sentiment-analysis figure from page 4.
|-- paperbot/backend/data/images/SENTIMENT_ANALYSIS_USING_NATURAL_LANGUAGE_PROCESSING_AND_MACHINE_LEARNING_page_4_figure_1.png
|   Cropped sentiment-analysis figure from page 4.
|-- paperbot/backend/data/images/SENTIMENT_ANALYSIS_USING_NATURAL_LANGUAGE_PROCESSING_AND_MACHINE_LEARNING_page_6_figure_0.png
|   Cropped sentiment-analysis figure from page 6.
|-- paperbot/backend/data/pdfs/
|   Indexed source PDF directory consumed by `ingest.py`.
|-- paperbot/backend/data/pdfs/.gitkeep
|   Keeps the PDF directory in version control when empty.
|-- paperbot/backend/data/pdfs/1_BERT.pdf
|   BERT research paper PDF.
|-- paperbot/backend/data/pdfs/2_RoBERTa.pdf
|   RoBERTa research paper PDF.
|-- paperbot/backend/data/pdfs/2_joels_paper_240428_181813.pdf
|   BCI/EEG research paper PDF.
|-- paperbot/backend/data/pdfs/A Comparative Study of Sentiment Analysis Using NLP and Different Machine Learning Techniques on US Airline Twitter Data.pdf
|   Sentiment analysis comparison paper PDF.
|-- paperbot/backend/data/pdfs/SENTIMENT ANALYSIS USING NATURAL LANGUAGE PROCESSING AND MACHINE LEARNING.pdf
|   Sentiment analysis with NLP and machine learning paper PDF.
|-- paperbot/backend/graph.py
|   LangGraph workflow, intent routing, retrieval, answer generation, image selection, and Redis conversation history.
|-- paperbot/backend/ingest.py
|   PDF ingestion CLI for text chunks, tables, figure crops, captions, embeddings, and Qdrant upserts.
|-- paperbot/backend/ingest_anthropic.log
|   Local ingestion log from Anthropic-assisted indexing.
|-- paperbot/backend/main.py
|   FastAPI app with health, ask, resume, API aliases, static asset serving, and frontend fallback.
|-- paperbot/backend/prompts.py
|   Prompt builders for intent classification, reference resolution, image query normalization, table summary, answer generation, suggestions, and image candidate selection.
|-- paperbot/backend/requirements.txt
|   Backend Python dependency list.
|-- paperbot/backend/schema.py
|   Pydantic models for structured LLM output and API requests/responses.
|-- paperbot/backend/static/
|   Built frontend output served by FastAPI.
|-- paperbot/backend/static/assets/
|   Built CSS and JavaScript bundles.
|-- paperbot/backend/static/assets/index-ByyvC2i9.css
|   Built frontend stylesheet bundle.
|-- paperbot/backend/static/assets/index-CYtxQ2bj.js
|   Built frontend JavaScript bundle.
|-- paperbot/backend/static/index.html
|   Built frontend HTML entry point.
|-- paperbot/frontend/
|   React/Vite frontend source.
|-- paperbot/frontend/index.html
|   Vite HTML entry point.
|-- paperbot/frontend/package-lock.json
|   Locked npm dependency graph for the frontend.
|-- paperbot/frontend/package.json
|   Frontend scripts and dependencies.
|-- paperbot/frontend/src/
|   Frontend React source directory.
|-- paperbot/frontend/src/App.jsx
|   Main React application shell, landing content, health check, Three.js particle background, chat launcher, and image modal state.
|-- paperbot/frontend/src/components/
|   Reusable chat UI components.
|-- paperbot/frontend/src/components/ChatWindow.jsx
|   Floating chat panel, message state, `/api/ask` calls, streaming text effect, suggestions, and inline image messages.
|-- paperbot/frontend/src/components/ImageModal.jsx
|   Full-size retrieved image modal.
|-- paperbot/frontend/src/components/MessageBubble.jsx
|   Markdown, table, image, typing, source, and suggestion rendering for chat messages.
|-- paperbot/frontend/src/components/SourcePanel.jsx
|   Source chips for retrieved file and page citations.
|-- paperbot/frontend/src/index.css
|   Frontend styles for the site, chat UI, markdown, tables, modal, and responsive behavior.
|-- paperbot/frontend/src/main.jsx
|   React root mount.
|-- paperbot/frontend/vite.config.js
|   Vite config that builds into `../backend/static` and proxies `/api` to FastAPI during development.
```

## Prerequisites

- Python 3.11.
- Node.js 18+.
- Docker.
- Redis installed natively and reachable on `localhost:6379`.
- Groq API key for text classification and answer generation.
- Anthropic API key for vision captioning and table fallback extraction.
- DocLayout-YOLO model weights as a `.pt` file, such as `doclayout_yolo_docstructbench_imgsz1024.pt`.
- The five source PDFs present under `paperbot/backend/data/pdfs`.

## Setup and installation

Clone the repository:

```bash
git clone https://github.com/lnarayanan03/paper_bot.git
cd paper_bot
```

Create a backend virtual environment and install the backend requirements:

```bash
cd paperbot/backend
python3.11 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Install the additional runtime packages imported by the current ingestion and graph code:

```bash
python3 -m pip install langchain-groq pdfplumber doclayout-yolo
```

Copy the example environment file and fill in real API keys:

```bash
cp .env.example .env
```

Edit `paperbot/backend/.env` so these values are set:

```bash
nano .env
```

Set `GROQ_API_KEY` to a real Groq API key, set `ANTHROPIC_API_KEY` to a real Anthropic API key, and set `PAPERBOT_DL_MODEL_PATH` to the absolute path of the DocLayout-YOLO weights file.

Start Qdrant with Docker:

```bash
docker run -d --name paperbot-qdrant -p 6333:6333 qdrant/qdrant
```

Verify Redis is running:

```bash
redis-cli ping
```

If Redis does not return `PONG`, start it with your native service manager:

```bash
sudo systemctl start redis
```

Set `PAPERBOT_DL_MODEL_PATH` in `.env` to the absolute path of the DocLayout-YOLO `.pt` file:

```bash
grep '^PAPERBOT_DL_MODEL_PATH=' .env
```

Run ingestion for all PDFs:

```bash
cd /home/ranga/paper_bot/paperbot/backend
. .venv/bin/activate
python3 ingest.py
```

Install and build the frontend into the FastAPI static directory:

```bash
cd /home/ranga/paper_bot/paperbot/frontend
npm install
npm run build
```

Start the backend:

```bash
cd /home/ranga/paper_bot/paperbot/backend
. .venv/bin/activate
python3 main.py
```

## Running the app

After setup, start PaperBot with this single command:

```bash
cd /home/ranga/paper_bot/paperbot/backend && . .venv/bin/activate && python3 main.py
```

Open the app at http://localhost:8000.

## Ingest options

Full ingest of all PDFs:

```bash
python3 ingest.py
```

Single-PDF ingest:

```bash
python3 ingest.py --pdf 1_BERT.pdf
```

Append mode without recreating Qdrant collections:

```bash
python3 ingest.py --no-wipe
```

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API key used by `ChatGroq` for intent classification, reference resolution, image query normalization, table summaries, answer generation, content suggestions, and image candidate selection. | `your_groq_api_key_here` |
| `ANTHROPIC_API_KEY` | Anthropic API key used by ingestion for image captioning and table extraction fallback. | `your_anthropic_api_key_here` |
| `ANTHROPIC_VISION_MODEL` | Anthropic model used for vision requests in ingestion. | `claude-sonnet-4-5` |
| `QDRANT_HOST` | Hostname for the Qdrant vector database. | `localhost` |
| `QDRANT_PORT` | Port for the Qdrant vector database. | `6333` |
| `REDIS_HOST` | Hostname for Redis conversation history storage. | `localhost` |
| `REDIS_PORT` | Port for Redis conversation history storage. | `6379` |
| `COLLECTION_TEXT` | Qdrant collection for embedded text chunks. | `paperbot_chunks` |
| `COLLECTION_IMAGES` | Qdrant collection for embedded image captions and search text. | `paperbot_images` |
| `COLLECTION_TABLES` | Qdrant collection for embedded Markdown tables. | `paperbot_tables` |
| `PAPERBOT_DL_MODEL_PATH` | Absolute path to the DocLayout-YOLO `.pt` weights file used by `YOLOv10`. | `/path/to/doclayout_yolo_docstructbench_imgsz1024.pt` |
| `IMAGE_TOP_K` | Number of image candidates retrieved from Qdrant before compatibility selection. | `10` |
| `IMAGE_MIN_SCORE` | Minimum vector similarity score accepted for image results. | `0.30` |

## Qdrant collections

| Collection | Content | Vector source |
|---|---|---|
| `paperbot_chunks` | Page text split into 500-character chunks with 50-character overlap. | MiniLM embedding of each cleaned text chunk through fastembed. |
| `paperbot_tables` | Markdown tables extracted by pdfplumber or Anthropic Vision fallback. | MiniLM embedding of the source PDF name, page number, and Markdown table text. |
| `paperbot_images` | Cropped figure and formula regions with source, page, caption, caption context, detector metadata, bounding box, and image path. | MiniLM embedding of source document name, region type, detector, local caption context, and Anthropic Vision caption. |

## Known limitations

- Groq free tier: 100k tokens/day.
- `IMAGE_MIN_SCORE` is set low at `0.30` due to query-length asymmetry between short user prompts and longer indexed image descriptions.
- Qdrant data is stored inside the Docker container by default because the setup command does not mount a host volume.
- Re-ingest is required if the embedding model changes because all stored vectors are generated with `sentence-transformers/all-MiniLM-L6-v2`.

## License

MIT
