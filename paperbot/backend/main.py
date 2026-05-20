"""FastAPI application entry point for PaperBot."""

from __future__ import annotations

import base64
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from graph import run_graph
from schema import (
    AskRequest,
    AskResponse,
    HealthResponse,
    ImageResponse,
    ResumeRequest,
    SourceChunk,
)

load_dotenv()


app = FastAPI(title="PaperBot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def encode_png_data_url(image_path: str | None) -> str | None:
    if not image_path:
        return None

    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return None

    image_base64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{image_base64}"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    state = run_graph(request.question, request.session_id)

    sources = []
    for chunk in state.get("chunks", []):
        text = str(chunk.get("text", ""))
        sources.append(SourceChunk(
            source=str(chunk.get("source", "")),
            page=int(chunk.get("page", 0) or 0),
            text_preview=text[:120],
        ))

    image_base64 = None
    if state.get("image_path") and state.get("needs_image"):
        image_base64 = encode_png_data_url(state["image_path"])

    return AskResponse(
        answer=state.get("answer", ""),
        needs_image=bool(state.get("needs_image", False)),
        has_relevant_image=bool(state.get("has_relevant_image", False)),
        has_relevant_table=bool(state.get("has_relevant_table", False)),
        image_base64=image_base64,
        image_source=str(state.get("image_source", "")),
        image_page=int(state.get("image_page", 0) or 0),
        sources=sources,
        interrupt_pending=False,
        session_id=request.session_id,
    )


@app.post("/resume", response_model=ImageResponse)
def resume(request: ResumeRequest) -> ImageResponse:
    state = run_graph(
        "", request.session_id,
        human_response={"approved": request.approved}
    )
    image_path = state.get("image_path")
    image_base64 = encode_png_data_url(image_path)

    return ImageResponse(
        image_path=image_path,
        image_base64=image_base64,
        image_source=str(state.get("image_source", "")),
        image_page=int(state.get("image_page", 0) or 0),
    )


@app.get("/api/health", response_model=HealthResponse)
def api_health() -> HealthResponse:
    return health()


@app.post("/api/ask", response_model=AskResponse)
def api_ask(request: AskRequest) -> AskResponse:
    return ask(request)


@app.post("/api/resume", response_model=ImageResponse)
def api_resume(request: ResumeRequest) -> ImageResponse:
    return resume(request)


_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_static_dir / "assets")), name="assets")


@app.get("/{full_path:path}", response_model=None)
def serve_frontend(full_path: str):
    if full_path.startswith("api"):
        return JSONResponse({"error": "not found"}, status_code=404)
    index = _static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"error": "Frontend not built. Run: cd frontend && npm run build"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
