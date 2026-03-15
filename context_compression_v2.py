"""
Automatic Context Compression Pipeline  ·  v2
LangChain + Google Gemini 2.5 Pro  +  ChromaDB Vector Store

NEW IN V2
─────────
• ChromaDB persistent vector store  (./chroma_store)
• Google text-embedding-004 embeddings for all chunks
• Document ingestion pipeline with overlap chunking
• search_documents now requires a document to be loaded first
  — the user is prompted to provide a file/text before any search

COMPRESSION LAYERS (unchanged from v1)
──────────────────────────────────────
  Layer 1 — Tool Result Offloading   (result > 2 000 tokens)
  Layer 2 — Tool Input Eviction      (context > 85 % window)
  Layer 3 — LLM Summarization        (context > 85 % after L2)
  Bonus   — Autonomous Compact Tool  (agent calls at task boundaries)
"""

from __future__ import annotations

import os
import re
import sys
import time
import json
import uuid
import textwrap
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

# LangChain / LangGraph
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage,
)
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict, Annotated


# Optional tiktoken — falls back to char-based estimate
def _load_encoder():
    try:
        import tiktoken as _tk
        return _tk.get_encoding("cl100k_base")
    except Exception:
        return None

_enc = _load_encoder()


# =============================================================================
# 0.  CONFIGURATION
# =============================================================================

@dataclass
class Config:
    # Context window
    model_context_window: int    = 1_000_000
    compression_threshold: float = 0.85
    tool_result_offload_threshold: int = 2_000
    recent_messages_keep: float  = 0.10
    max_summary_tokens: int      = 800

    # ChromaDB + Embeddings
    chroma_dir: Path       = Path("./chroma_store")
    collection_name: str   = "documents"
    embedding_model: str   = "models/gemini-embedding-001"

    # Chunking
    chunk_size: int        = 800   # characters per chunk
    chunk_overlap: int     = 150   # overlap between chunks
    min_chunk_size: int    = 100   # discard shorter chunks

    # Filesystem offload
    offload_dir: Path      = Path("./offload_store")

    # UX
    verbose: bool = True


CFG = Config()
CFG.chroma_dir.mkdir(parents=True, exist_ok=True)
CFG.offload_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1.  TOKEN COUNTING
# =============================================================================

def count_tokens(text: str) -> int:
    if _enc is not None:
        return len(_enc.encode(text, disallowed_special=()))
    return max(1, len(text) // 4)


def message_tokens(msg: BaseMessage) -> int:
    content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
    overhead = 0
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        overhead = sum(count_tokens(json.dumps(tc)) for tc in msg.tool_calls)
    return count_tokens(content) + overhead + 4


def context_tokens(messages: list[BaseMessage]) -> int:
    return sum(message_tokens(m) for m in messages)


def context_fill(messages: list[BaseMessage]) -> float:
    return context_tokens(messages) / CFG.model_context_window


# =============================================================================
# 2.  OFFLOAD STORE
# =============================================================================

def offload_to_disk(content: str, label: str = "content") -> str:
    fname = CFG.offload_dir / f"{label}_{uuid.uuid4().hex[:8]}.txt"
    fname.write_text(content, encoding="utf-8")
    return str(fname)


def read_offload(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def preview(text: str, lines: int = 10) -> str:
    return "\n".join(text.splitlines()[:lines])


# =============================================================================
# 3.  CHUNKING ENGINE
#     Overlap-aware character chunker with sentence-boundary snapping.
# =============================================================================

def _snap_to_sentence(text: str, pos: int, window: int = 80) -> int:
    """Move pos backward to the nearest sentence boundary within window chars."""
    region = text[max(0, pos - window): pos + window]
    matches = list(re.finditer(r"[.!?]\s", region))
    if matches:
        best = matches[-1]
        return max(0, pos - window) + best.end()
    return pos


def chunk_text(text: str, source_id: str = "doc") -> list[Document]:
    """
    Split text into overlapping Documents for embedding.

    Each Document carries metadata:
        source_id  — caller-supplied label
        chunk_idx  — zero-based chunk position
        char_start — character offset in original text
        chunk_len  — length of this chunk
    """
    text = text.strip()
    if not text:
        return []

    docs: list[Document] = []
    start = 0
    idx = 0

    while start < len(text):
        end = start + CFG.chunk_size
        if end < len(text):
            end = _snap_to_sentence(text, end)

        chunk = text[start:end].strip()
        if len(chunk) >= CFG.min_chunk_size:
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source_id":  source_id,
                    "chunk_idx":  idx,
                    "char_start": start,
                    "chunk_len":  len(chunk),
                },
            ))
            idx += 1

        start = end - CFG.chunk_overlap
        if start <= 0 or end >= len(text):
            break

    return docs


def chunk_file(path: str) -> list[Document]:
    """Read a UTF-8 text file and return chunks."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    text = p.read_text(encoding="utf-8", errors="replace")
    return chunk_text(text, source_id=p.name)


# =============================================================================
# 4.  VECTOR STORE  (ChromaDB + Google text-embedding-004)
# =============================================================================

class VectorStoreManager:
    """
    Persistent ChromaDB collection with Google text-embedding-004.

    Public API
    ----------
    ingest_text(text, source_id)  -> int          chunks stored
    ingest_file(path)             -> int          chunks stored
    search(query, k)              -> list[dict]   ranked results
    stats()                       -> dict
    clear()
    """

    def __init__(self, api_key: str):
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=CFG.embedding_model,
            google_api_key=api_key,
        )
        self._store = Chroma(
            collection_name=CFG.collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(CFG.chroma_dir),
        )
        if CFG.verbose:
            n = self._store._collection.count()
            print(f"  [VectorStore] '{CFG.collection_name}' loaded "
                  f"({n} existing chunks) from {CFG.chroma_dir}")

    # ── Ingestion ──────────────────────────────────────────────────────────

    def ingest_text(self, text: str, source_id: str = "inline") -> int:
        docs = chunk_text(text, source_id=source_id)
        if not docs:
            return 0
        self._store.add_documents(docs)
        if CFG.verbose:
            print(f"  [VectorStore] Ingested {len(docs)} chunks  source='{source_id}'")
        return len(docs)

    def ingest_file(self, path: str) -> int:
        docs = chunk_file(path)
        if not docs:
            return 0
        self._store.add_documents(docs)
        if CFG.verbose:
            print(f"  [VectorStore] Ingested {len(docs)} chunks  file='{path}'")
        return len(docs)

    # ── Retrieval ──────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5) -> list[dict]:
        hits = self._store.similarity_search_with_relevance_scores(query, k=k)
        return [
            {
                "content":    doc.page_content,
                "source_id":  doc.metadata.get("source_id", "unknown"),
                "chunk_idx":  doc.metadata.get("chunk_idx", -1),
                "char_start": doc.metadata.get("char_start", 0),
                "score":      round(score, 4),
            }
            for doc, score in hits
        ]

    # ── Metadata ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "total_chunks": self._store._collection.count(),
            "collection":   CFG.collection_name,
        }

    def clear(self):
        self._store._collection.delete(where={"chunk_idx": {"$gte": 0}})
        if CFG.verbose:
            print("  [VectorStore] Cleared.")


# Module-level singletons set during build_agent()
_VS: Optional[VectorStoreManager] = None
_GLOBAL_LLM: Optional[ChatGoogleGenerativeAI] = None
_DOCUMENTS_LOADED: bool = False     # gate: must be True before search is allowed


# =============================================================================
# 5.  THREE-LAYER COMPRESSION PIPELINE
# =============================================================================

# ── Layer 1 — Tool Result Offloading ─────────────────────────────────────

def layer1_offload_tool_results(messages: list[BaseMessage]) -> list[BaseMessage]:
    out = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            raw = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            tok = count_tokens(raw)
            if tok > CFG.tool_result_offload_threshold:
                path = offload_to_disk(raw, label=f"tool_{msg.name or 'result'}")
                replacement = (
                    f"[OFFLOADED — {tok:,} tokens → {path}]\n"
                    f"Preview (first 10 lines):\n{preview(raw)}\n"
                    f"Use read_offloaded_file('{path}') to retrieve full content."
                )
                msg = ToolMessage(content=replacement,
                                  tool_call_id=msg.tool_call_id, name=msg.name)
                if CFG.verbose:
                    print(f"  [Layer 1] Offloaded tool result ({tok:,} tok) → {path}")
        out.append(msg)
    return out


# ── Layer 2 — Tool Input Eviction ────────────────────────────────────────

_WRITE_TOOLS = {"write_file", "edit_file", "create_file", "save_content",
                "ingest_document", "ingest_file_path"}


def layer2_offload_tool_inputs(messages: list[BaseMessage]) -> list[BaseMessage]:
    out = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            new_tcs = []
            for tc in msg.tool_calls:
                if tc.get("name") in _WRITE_TOOLS:
                    args_str = json.dumps(tc.get("args", {}))
                    if count_tokens(args_str) > 500:
                        path = offload_to_disk(args_str, label=f"input_{tc['name']}")
                        tc = {**tc, "args": {"__offloaded__": path}}
                        if CFG.verbose:
                            print(f"  [Layer 2] Evicted input for '{tc['name']}' → {path}")
                new_tcs.append(tc)
            msg = AIMessage(content=msg.content, tool_calls=new_tcs, id=msg.id)
        out.append(msg)
    return out


# ── Layer 3 — LLM Summarization ──────────────────────────────────────────

_SUMMARIZATION_PROMPT = """You are a context compression assistant.
Produce a concise JSON summary so the agent can continue its task seamlessly.

Return ONLY valid JSON — no markdown fences, no extra text:
{
  "session_intent":    "<user goal in 1-2 sentences>",
  "progress_so_far":   "<bullet list of completed steps>",
  "key_facts":         "<critical findings/decisions the agent must remember>",
  "artifacts_created": "<files/outputs produced with paths/IDs>",
  "documents_loaded":  "<names/IDs ingested into the vector store>",
  "next_steps":        "<what the agent should do next>",
  "open_questions":    "<unresolved items needing user input>"
}"""


def layer3_summarize(messages: list[BaseMessage],
                     llm: ChatGoogleGenerativeAI) -> list[BaseMessage]:
    keep_budget = int(CFG.model_context_window * CFG.recent_messages_keep)

    tail: list[BaseMessage] = []
    tail_tok = 0
    for msg in reversed(messages):
        t = message_tokens(msg)
        if tail_tok + t > keep_budget:
            break
        tail.insert(0, msg)
        tail_tok += t

    head = messages[: len(messages) - len(tail)]
    if not head:
        return messages

    serialized = json.dumps(
        [{"role": m.__class__.__name__, "content": m.content} for m in head],
        ensure_ascii=False, indent=2,
    )
    archive_path = offload_to_disk(serialized, label="conversation_archive")

    history_text = "\n\n".join(
        f"[{m.__class__.__name__}]\n{m.content}"
        for m in head if isinstance(m.content, str)
    )[:40_000]

    try:
        resp = llm.invoke([
            SystemMessage(content=_SUMMARIZATION_PROMPT),
            HumanMessage(content=f"Summarize this conversation:\n\n{history_text}"),
        ])
        raw = resp.content.strip()
        raw = re.sub(r"^```(?:json)?", "", raw).rstrip("```").strip()
        summary_data = json.loads(raw)
    except Exception as e:
        summary_data = {"session_intent": "Unknown", "error": str(e)}

    summary_text = (
        "=== CONTEXT COMPRESSED (Layer 3 Summarization) ===\n"
        f"Archive: {archive_path}\n\n"
        + "\n".join(f"**{k.upper().replace('_',' ')}**: {v}"
                    for k, v in summary_data.items())
        + "\n\nUse read_offloaded_file() to recover any detail."
    )

    compressed = [SystemMessage(content=summary_text)] + tail
    saved = context_tokens(messages) - context_tokens(compressed)
    if CFG.verbose:
        print(f"  [Layer 3] Summarized {len(head)} msgs → saved ~{saved:,} tokens | {archive_path}")
    return compressed


# ── Orchestrator ─────────────────────────────────────────────────────────

class CompressionStats:
    def __init__(self):
        self.l1 = self.l2 = self.l3 = self.autonomous = self.tokens_saved = 0

    def report(self):
        print("\n" + "═" * 54)
        print("  COMPRESSION STATISTICS")
        print("═" * 54)
        print(f"  Layer 1 (tool result offloads):  {self.l1}")
        print(f"  Layer 2 (tool input evictions):  {self.l2}")
        print(f"  Layer 3 (summarizations):        {self.l3}")
        print(f"  Autonomous compact calls:        {self.autonomous}")
        print(f"  Total tokens saved (approx):     {self.tokens_saved:,}")
        print("═" * 54)


STATS = CompressionStats()


def run_compression_pipeline(
    messages: list[BaseMessage],
    llm: ChatGoogleGenerativeAI,
    force: bool = False,
) -> list[BaseMessage]:
    before = context_tokens(messages)

    # Layer 1 — always on
    pre = context_tokens(messages)
    messages = layer1_offload_tool_results(messages)
    d = pre - context_tokens(messages)
    if d > 0:
        STATS.l1 += 1
        STATS.tokens_saved += d

    if context_fill(messages) < CFG.compression_threshold and not force:
        return messages

    if CFG.verbose:
        print(f"\n  Compression triggered (fill={context_fill(messages):.1%})")

    # Layer 2
    pre = context_tokens(messages)
    messages = layer2_offload_tool_inputs(messages)
    d = pre - context_tokens(messages)
    if d > 0:
        STATS.l2 += 1
        STATS.tokens_saved += d

    # Layer 3
    if context_fill(messages) >= CFG.compression_threshold or force:
        pre = context_tokens(messages)
        messages = layer3_summarize(messages, llm)
        STATS.l3 += 1
        STATS.tokens_saved += pre - context_tokens(messages)

    saved = before - context_tokens(messages)
    if CFG.verbose and saved > 0:
        print(f"  Saved {saved:,} tokens | fill now {context_fill(messages):.1%}")

    return messages


# =============================================================================
# 6.  AGENT TOOLS
# =============================================================================

@tool
def ingest_document(text: str, source_name: str = "user_document") -> str:
    """
    Chunk raw text, embed each chunk with Google text-embedding-004,
    and store everything into ChromaDB.

    Chunking: 800-character chunks with 150-character overlap,
    snapped to sentence boundaries where possible.

    Args:
        text:        Full document text to ingest.
        source_name: Short label identifying this document in search results.

    Returns: Confirmation with chunk count and total collection size.
    """
    global _DOCUMENTS_LOADED
    if _VS is None:
        return "ERROR: Vector store not initialised. Call build_agent() first."
    n = _VS.ingest_text(text, source_id=source_name)
    _DOCUMENTS_LOADED = True
    st = _VS.stats()
    return (
        f"Document '{source_name}' ingested.\n"
        f"  Chunks created:       {n}\n"
        f"  Total chunks in store: {st['total_chunks']}\n"
        f"  Embedding model:       {CFG.embedding_model}"
    )


@tool
def ingest_file_path(file_path: str) -> str:
    """
    Load a local file, chunk it, embed with Google text-embedding-004,
    and store in ChromaDB. Supports any UTF-8 text file (.txt, .md, .py, .json, .csv …).

    Args:
        file_path: Absolute or relative path to the file.

    Returns: Confirmation with chunk count.
    """
    global _DOCUMENTS_LOADED
    if _VS is None:
        return "ERROR: Vector store not initialised."
    try:
        n = _VS.ingest_file(file_path)
        _DOCUMENTS_LOADED = True
        st = _VS.stats()
        return (
            f"File '{file_path}' ingested.\n"
            f"  Chunks created:       {n}\n"
            f"  Total chunks in store: {st['total_chunks']}"
        )
    except FileNotFoundError:
        return f"ERROR: File not found — '{file_path}'"
    except Exception as e:
        return f"ERROR ingesting file: {e}"


@tool
def search_documents(query: str, num_results: int = 5) -> str:
    """
    Semantic search over ingested documents using ChromaDB + Google embeddings.

    IMPORTANT: A document must be ingested first via ingest_document() or
    ingest_file_path(). If no document has been loaded this tool returns a
    clear instruction asking the user to provide one.

    Args:
        query:       Natural-language search query.
        num_results: Number of top chunks to return (default 5, max 20).

    Returns: Ranked result chunks with relevance scores and source metadata.
    """
    if _VS is None:
        return "ERROR: Vector store not initialised."

    if not _DOCUMENTS_LOADED:
        return (
            "NO DOCUMENTS LOADED YET.\n"
            "Please provide a document first using one of:\n"
            "  ingest_document(text, source_name)  — for raw text\n"
            "  ingest_file_path(file_path)          — for a local file\n"
            "Then call search_documents() again with your query."
        )

    st = _VS.stats()
    if st["total_chunks"] == 0:
        return "The vector store is empty. Please ingest a document first."

    results = _VS.search(query, k=min(num_results, 20))
    if not results:
        return f"No results found for: '{query}'"

    lines = [f"Search results for: '{query}'\n{'─' * 52}"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"\n[{i}] source={r['source_id']}  chunk={r['chunk_idx']}  "
            f"relevance={r['score']:.3f}\n{r['content']}"
        )
    lines.append(f"\n{'─' * 52}\n{len(results)} chunks from {st['total_chunks']} total.")
    return "\n".join(lines)


@tool
def vector_store_stats() -> str:
    """Show ChromaDB collection stats: chunk count, collection name, embedding model."""
    if _VS is None:
        return "Vector store not initialised."
    st = _VS.stats()
    return (
        f"Collection:       '{st['collection']}'\n"
        f"Total chunks:     {st['total_chunks']}\n"
        f"Embedding model:  {CFG.embedding_model}\n"
        f"Persist dir:      {CFG.chroma_dir}\n"
        f"Documents loaded: {_DOCUMENTS_LOADED}"
    )


@tool
def compact_context(reason: str = "") -> str:
    """
    Trigger autonomous context compression at a natural task boundary.
    Call this after completing a major phase, or before an unrelated new task.

    Args:
        reason: Why you are compacting now (logged for debugging).
    """
    if CFG.verbose:
        print(f"\n  [Autonomous] Agent requested compaction: {reason}")
    STATS.autonomous += 1
    return f"[AUTONOMOUS COMPACT REQUESTED at {datetime.now().isoformat()}] Reason: {reason}"


@tool
def read_offloaded_file(path: str) -> str:
    """
    Recover content offloaded to disk during context compression.
    Use the path from an [OFFLOADED] notice.

    Args:
        path: Filesystem path from a previous offload operation.
    """
    try:
        content = read_offload(path)
        if CFG.verbose:
            print(f"  [Recovery] Read {path} ({len(content):,} chars)")
        return content
    except FileNotFoundError:
        return f"ERROR: Not found — '{path}'"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def write_file(filename: str, content: str) -> str:
    """Write text content to a local file."""
    Path(filename).write_text(content, encoding="utf-8")
    return f"Wrote {len(content):,} chars to '{filename}'"


@tool
def calculate(expression: str) -> str:
    """Evaluate a safe mathematical expression (supports Python math module)."""
    try:
        allowed = {k: v for k, v in __import__("math").__dict__.items()
                   if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"


# =============================================================================
# 7.  SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a helpful AI assistant backed by a semantic document search system
(ChromaDB + Google text-embedding-004 + Gemini 2.5 Pro).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY DOCUMENT-FIRST WORKFLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Before ANY search, a document must be ingested.
   • If the user asks to search without providing a document,
     respond: "Please provide a document (raw text or file path) first."
   • Once they give you text or a path, call ingest_document() or ingest_file_path().
2. Only after ingestion, call search_documents() with the user's query.
3. If search_documents() returns "NO DOCUMENTS LOADED", ask for a document.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ingest_document(text, source_name) — chunk + embed + store raw text
ingest_file_path(file_path)        — chunk + embed + store a local file
search_documents(query, k)         — semantic retrieval (needs doc first)
vector_store_stats()               — show collection info
write_file(filename, content)      — save output to disk
calculate(expression)              — evaluate maths
compact_context(reason)            — autonomous compression
read_offloaded_file(path)          — recover offloaded content

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT MANAGEMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Call compact_context() when:
• You have finished ingesting and processing a large document.
• The user moves to a completely new, unrelated task.
• The conversation contains lots of stale intermediate content.
"""


# =============================================================================
# 8.  AGENT GRAPH
# =============================================================================

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    autonomous_compact_requested: bool


def build_agent(
    model_name: str = "gemini-2.5-pro",
    api_key: Optional[str] = None,
    extra_tools: list = [],
) -> tuple:
    """
    Build and return (compiled_graph, llm, vector_store_manager).

    Parameters
    ----------
    model_name   Gemini model (default "gemini-2.5-pro")
    api_key      Google API key; falls back to GOOGLE_API_KEY env var
    extra_tools  Additional LangChain tools to register
    """
    global _VS, _GLOBAL_LLM

    key = api_key or os.environ.get("GOOGLE_API_KEY", "")

    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=key,
        temperature=0,
        max_tokens=8192,
    )
    _GLOBAL_LLM = llm
    _VS = VectorStoreManager(api_key=key)

    all_tools = [
        ingest_document,
        ingest_file_path,
        search_documents,
        vector_store_stats,
        write_file,
        calculate,
        compact_context,
        read_offloaded_file,
    ] + extra_tools

    llm_with_tools = llm.bind_tools(all_tools)

    # ── Nodes ─────────────────────────────────────────────────────────────

    def call_model(state: AgentState) -> dict:
        msgs = run_compression_pipeline(state["messages"], llm)
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs
        response = llm_with_tools.invoke(msgs)
        return {"messages": [response], "autonomous_compact_requested": False}

    def run_tools(state: AgentState) -> dict:
        result = ToolNode(all_tools).invoke(state)
        compact_req = any(
            isinstance(m, ToolMessage) and "AUTONOMOUS COMPACT REQUESTED" in (m.content or "")
            for m in result.get("messages", [])
        )
        return {**result, "autonomous_compact_requested": compact_req}

    def compress_after_tools(state: AgentState) -> dict:
        force = state.get("autonomous_compact_requested", False)
        msgs = run_compression_pipeline(state["messages"], llm, force=force)
        return {"messages": msgs, "autonomous_compact_requested": False}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if (isinstance(last, AIMessage) and last.tool_calls) else END

    # ── Build ──────────────────────────────────────────────────────────────

    g = StateGraph(AgentState)
    g.add_node("model",    call_model)
    g.add_node("tools",    run_tools)
    g.add_node("compress", compress_after_tools)

    g.add_edge(START, "model")
    g.add_conditional_edges("model", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools",    "compress")
    g.add_edge("compress", "model")

    return g.compile(), llm, _VS


# =============================================================================
# 9.  CONVERSATION RUNNER
# =============================================================================

def chat(
    agent,
    history: list[BaseMessage],
    user_input: str,
) -> tuple[str, list[BaseMessage]]:
    """One conversation turn. Returns (response_text, updated_history)."""
    history = history + [HumanMessage(content=user_input)]

    tok_before = context_tokens(history)
    print(f"\n{'─' * 60}")
    print(f"User: {user_input}")
    print(f"Context: {tok_before:,} tokens ({context_fill(history):.2%} full)")

    result = agent.invoke({"messages": history, "autonomous_compact_requested": False})
    msgs = result["messages"]

    response_text = ""
    for msg in reversed(msgs):
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            content = msg.content
            if isinstance(content, list):
                # Extract text from content blocks: [{'type': 'text', 'text': '...'}, ...]
                response_text = "\n".join(
                    block["text"] for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            else:
                response_text = content
            break

    tok_after = context_tokens(msgs)
    delta = tok_before - tok_after
    print(f"\nAssistant: {response_text[:600]}{'...' if len(response_text) > 600 else ''}")
    print(f"Context: {tok_after:,} tokens ({context_fill(msgs):.2%} full)"
          + (f"  [saved {delta:,} tokens]" if delta > 0 else ""))

    return response_text, msgs


# =============================================================================
# 10.  INTERACTIVE SESSION  (document-first UX)
# =============================================================================

SAMPLE_DOCUMENT = textwrap.dedent("""
    Climate Change and Agriculture: A Comprehensive Overview

    Climate change poses one of the most significant threats to global food security.
    Rising temperatures, shifting precipitation patterns, and increasing frequency of
    extreme weather events are fundamentally altering agricultural systems worldwide.

    Temperature Effects on Crop Yields
    Studies have shown that for every 1 degree Celsius increase in global mean
    temperature, wheat yields decline by approximately 6%, rice by 3.2%, maize by
    7.4%, and soybean by 3.1% (Zhao et al., 2017). These reductions are primarily
    due to accelerated phenological development, heat stress during pollination, and
    increased evapotranspiration demand.

    Water Stress and Irrigation
    Changing precipitation patterns are creating both drought and flood risks for
    agriculture. In semi-arid regions, reduced rainfall is forcing farmers to rely
    more heavily on irrigation, depleting groundwater aquifers at unsustainable rates.
    The Colorado River basin supplies water to 40 million people and irrigates
    5.5 million acres of farmland, yet is experiencing historically low reservoir
    levels due to prolonged drought exacerbated by climate change.

    Adaptation Strategies
    Farmers worldwide are implementing various adaptation strategies:
    1. Drought-resistant crop varieties developed through traditional breeding and
       genetic modification offer promise for maintaining yields under water stress.
    2. Precision irrigation technologies, including drip irrigation and soil moisture
       sensors, reduce water usage by 30-50% compared to conventional flood irrigation.
    3. Agroforestry systems integrating trees with crops provide shade, reduce
       evaporation, sequester carbon, and improve soil health.
    4. Shifting planting dates to avoid peak temperature stress during critical
       growth phases has shown 10-15% yield improvements in some regions.

    Food Security Implications
    The World Food Programme estimates that climate change could push an additional
    80 million people into hunger by 2050. Sub-Saharan Africa and South Asia are
    particularly vulnerable, as these regions depend heavily on rain-fed agriculture
    and have limited adaptive capacity. Smallholder farmers, who produce 70% of the
    world's food, are disproportionately affected due to limited access to
    technology, credit, and climate information services.

    Policy Recommendations
    Addressing agricultural impacts of climate change requires action at multiple
    levels. International cooperation under the Paris Agreement must include
    agriculture-specific mitigation and adaptation targets. National governments
    should invest in agricultural research and extension services, climate-smart
    infrastructure, and social protection programs for vulnerable farming communities.
""").strip()


def run_interactive_session(api_key: str, model_name: str = "gemini-2.5-pro"):
    print("\n" + "=" * 64)
    print("  Context Compression Pipeline  v2")
    print("  ChromaDB + Google text-embedding-004 + Gemini 2.5 Pro")
    print("=" * 64)

    agent, llm, vs = build_agent(model_name=model_name, api_key=api_key)
    history: list[BaseMessage] = []

    print(f"\n  Embedding model : {CFG.embedding_model}")
    print(f"  Vector store    : {CFG.chroma_dir}  (ChromaDB persistent)")
    print(f"  Chunk size      : {CFG.chunk_size} chars  |  overlap: {CFG.chunk_overlap} chars")
    print(f"  Context window  : {CFG.model_context_window:,} tokens")
    print(f"  Compress at     : {CFG.compression_threshold:.0%} fill\n")

    # ── STEP 1 — Prompt for document ──────────────────────────────────────

    print("=" * 64)
    print("  STEP 1: Load a document into the vector store")
    print("=" * 64)
    print("  [1] Use the built-in sample document  (climate & agriculture)")
    print("  [2] Provide a file path               (.txt / .md / .py / etc.)")
    print("  [3] Paste raw text                    (type END on a blank line to finish)")

    choice = input("\nYour choice [1/2/3]: ").strip()

    if choice == "1":
        print("\nUsing sample document: 'Climate Change and Agriculture'")
        _, history = chat(
            agent, history,
            f'Ingest the following document with source_name="climate_agriculture":\n\n{SAMPLE_DOCUMENT}',
        )

    elif choice == "2":
        fp = input("File path: ").strip()
        _, history = chat(agent, history, f'Ingest the file at: {fp}')

    elif choice == "3":
        print("Paste text below. Enter 'END' on its own line when done:")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        raw = "\n".join(lines)
        name = input("Document name (e.g. my_report): ").strip() or "user_text"
        _, history = chat(agent, history,
                          f'Ingest this document with source_name="{name}":\n\n{raw}')

    else:
        print("Invalid choice — using sample document.")
        _, history = chat(
            agent, history,
            f'Ingest the following document with source_name="climate_agriculture":\n\n{SAMPLE_DOCUMENT}',
        )

    # ── STEP 2 — Q&A loop ─────────────────────────────────────────────────

    print("\n" + "=" * 64)
    print("  STEP 2: Ask questions  (type 'quit' to exit)")
    print("=" * 64)

    while True:
        try:
            query = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue

        _, history = chat(agent, history, query)

    STATS.report()
    print("\nSession complete. Goodbye!")


# =============================================================================
# 11.  MAIN  (CLI — unchanged)
# =============================================================================

def main():
    api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        print("=" * 64)
        print("  Context Compression Pipeline  v2")
        print("  ChromaDB + Google text-embedding-004 + Gemini 2.5 Pro")
        print("=" * 64)
        print("\n  GOOGLE_API_KEY not set.")
        print("  export GOOGLE_API_KEY='your-key-here'")
        print("  python context_compression_v2.py\n")
        print("Pipeline configuration:")
        print(f"  Embedding model   : {CFG.embedding_model}")
        print(f"  ChromaDB dir      : {CFG.chroma_dir}")
        print(f"  Collection name   : {CFG.collection_name}")
        print(f"  Chunk size        : {CFG.chunk_size} chars")
        print(f"  Chunk overlap     : {CFG.chunk_overlap} chars")
        print(f"  Min chunk size    : {CFG.min_chunk_size} chars")
        print(f"  Context window    : {CFG.model_context_window:,} tokens")
        print(f"  Compress at       : {CFG.compression_threshold:.0%} fill")
        print(f"  L1 offload at     : >{CFG.tool_result_offload_threshold:,} tokens")
        print(f"  Recent tail kept  : {CFG.recent_messages_keep:.0%} of window\n")
        print("Compression layers:")
        print("  Layer 1  Tool Result Offloading  (always-on, per message)")
        print("  Layer 2  Tool Input Eviction     (fires at 85% context fill)")
        print("  Layer 3  LLM Summarization       (fires when L2 insufficient)")
        print("  Bonus    Autonomous Compact Tool  (agent calls at task boundaries)\n")
        print("Document search flow:")
        print("  User provides doc → ingest_document() / ingest_file_path()")
        print("  → chunked into 800-char pieces with 150-char overlap")
        print("  → embedded with Google text-embedding-004")
        print("  → stored in ChromaDB (persistent)")
        print("  → search_documents() performs semantic retrieval")
        return

    run_interactive_session(api_key=api_key)


# =============================================================================
# 12.  STREAMLIT UI
# =============================================================================

def _in_streamlit() -> bool:
    """Return True when executed via `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


class _QueueWriter:
    """Redirect print() output from a worker thread into a Queue."""

    encoding = "utf-8"   # prevents ascii-codec errors on Windows
    errors   = "replace"

    def __init__(self, q):
        self._q, self._buf = q, ""

    def write(self, text) -> int:
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._q.put(("log", line.strip()))
        return len(text)

    def flush(self):
        if self._buf.strip():
            self._q.put(("log", self._buf.strip()))
            self._buf = ""


def _agent_worker(fn, q):
    """Run fn() in a thread, capturing stdout into the queue."""
    writer = _QueueWriter(q)
    old, sys.stdout = sys.stdout, writer
    try:
        q.put(("done", fn()))
    except Exception as exc:
        q.put(("error", str(exc)))
    finally:
        sys.stdout = old
        writer.flush()


def _classify_log(line: str) -> str:
    lo = line.lower()
    if "[layer 1]"     in lo: return "l1"
    if "[layer 2]"     in lo: return "l2"
    if "[layer 3]"     in lo: return "l3"
    if "[autonomous]"  in lo: return "auto"
    if "[vectorstore]" in lo: return "store"
    if "compression triggered" in lo: return "ctx"
    if "saved" in lo and "tokens" in lo: return "ctx"
    if "fill" in lo and "%" in lo: return "ctx"
    if "context:" in lo and "tokens" in lo: return "ctx"
    return "info"


_EVT_ICON = {
    "l1": "💾", "l2": "✂️", "l3": "🗜️",
    "auto": "⚡", "store": "📦", "ctx": "📊", "info": "›",
}
_EVT_COLOR = {
    "l1": "#d29922", "l2": "#bc8cff", "l3": "#f85149",
    "auto": "#39d0d8", "store": "#3fb950", "ctx": "#58a6ff", "info": "#8b949e",
}

_UI_DEFAULTS: dict = {
    "google_api_key": "",
    "model_name":     "gemini-2.5-pro",
    "agent":          None,
    "history":        [],
    "doc_loaded":     False,
    "doc_name":       "",
    "chunks_stored":  0,
    "activity_log":   [],
    "result":         "",
    "running":        False,
    "queue":          None,
    "phase":          "idle",
    "ingest_meta":    {},
    "ui_stats":       {"l1": 0, "l2": 0, "l3": 0, "auto": 0, "saved": 0},
    "ctx_fill":       0.0,
    "ctx_tokens":     0,
}


def streamlit_app():
    """
    Streamlit UI for the Context Compression Pipeline v2.
    Run with:  streamlit run context_compression_v2.py
    """
    # Streamlit re-executes the entire script on every rerun, which resets
    # module-level globals (_VS, _DOCUMENTS_LOADED) back to None/False.
    # We persist them in session_state and restore them here on every rerun.
    global _VS, _GLOBAL_LLM, _DOCUMENTS_LOADED

    import tempfile
    import streamlit as st
    from queue import Queue, Empty
    from threading import Thread

    st.set_page_config(
        page_title="Context Compression v2",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');
:root {
    --bg:#0d1117; --surf:#161b22; --surf2:#21262d; --border:rgba(48,54,61,.9);
    --blue:#58a6ff; --green:#3fb950; --amber:#d29922; --red:#f85149;
    --purple:#bc8cff; --cyan:#39d0d8; --text:#c9d1d9; --muted:#8b949e;
    --mono:'JetBrains Mono',monospace;
}
* { font-family:'Inter',sans-serif; }
.stApp { background:var(--bg); }
.cc-header {
    background:linear-gradient(135deg,#161b22,#0d1117);
    border:1px solid var(--border); border-top:3px solid var(--blue);
    border-radius:12px; padding:24px 34px; margin-bottom:20px;
}
.cc-header h1 { font-size:1.8rem; font-weight:600; color:#f0f6fc; margin:0 0 4px; letter-spacing:-.02em; }
.cc-header p  { color:var(--muted); font-size:.84rem; margin:0; }
.step-lbl {
    font-size:.71rem; font-weight:600; font-family:var(--mono);
    color:var(--blue); text-transform:uppercase; letter-spacing:.08em; margin-bottom:9px;
    display:inline-block;
}
.evt {
    display:flex; align-items:flex-start; gap:9px;
    padding:7px 11px; border-radius:6px; margin-bottom:5px;
    font-size:.78rem; border-left:3px solid transparent;
    background:var(--surf); animation:fadeIn .2s ease;
}
.evt.l1   {border-color:var(--amber);}  .evt.l2   {border-color:var(--purple);}
.evt.l3   {border-color:var(--red);}    .evt.auto {border-color:var(--cyan);}
.evt.store{border-color:var(--green);}  .evt.ctx  {border-color:var(--blue);}
.evt.info {border-color:var(--muted);}
.evt-msg  {color:var(--text);line-height:1.45;word-break:break-word;}
.evt-time {font-family:var(--mono);font-size:.65rem;color:var(--muted);white-space:nowrap;}
.stat-chip{background:var(--surf);border:1px solid var(--border);border-radius:8px;padding:9px 12px;text-align:center;}
.stat-chip .val{font-family:var(--mono);font-size:1.2rem;font-weight:600;}
.stat-chip .lbl{font-size:.67rem;color:var(--muted);margin-top:2px;}
.ctx-wrap{background:var(--surf);border:1px solid var(--border);border-radius:8px;padding:10px 14px;margin-bottom:13px;}
.ctx-track{height:8px;background:var(--surf2);border-radius:4px;overflow:hidden;margin:5px 0 3px;}
.ctx-fill-bar{height:100%;border-radius:4px;transition:width .4s ease;}
.result-box{
    background:var(--surf);border:1px solid var(--border);border-radius:10px;
    padding:16px;font-size:.85rem;line-height:1.7;color:var(--text);
    white-space:pre-wrap;max-height:480px;overflow-y:auto;
}
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea{
    background:var(--surf)!important;border-color:var(--border)!important;color:var(--text)!important;
}
.stButton button{
    background:var(--blue)!important;color:#0d1117!important;
    border:none!important;border-radius:6px!important;font-weight:600!important;
}
.stButton button:disabled{opacity:.4!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding-top:1.1rem;padding-bottom:2rem;}
@keyframes fadeIn{from{opacity:0;transform:translateX(-4px)}to{opacity:1;transform:translateX(0)}}
</style>
""", unsafe_allow_html=True)

    # ── session state ─────────────────────────────────────────────────────────
    for k, v in _UI_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Pre-populate API key widget from environment on very first run
    if "api_key_widget" not in st.session_state:
        env_key = os.environ.get("GOOGLE_API_KEY", "")
        st.session_state["api_key_widget"] = env_key
        if env_key:
            st.session_state.google_api_key = env_key

    # Restore module-level singletons that Streamlit reset by re-executing the script
    if st.session_state.get("_cc_vs") is not None:
        _VS = st.session_state["_cc_vs"]
    if st.session_state.get("_cc_docs_loaded", False):
        _DOCUMENTS_LOADED = True

    def _now():
        return datetime.now().strftime("%H:%M:%S")

    def add_event(msg, kind="info"):
        st.session_state.activity_log.append({"msg": msg, "kind": kind, "time": _now()})

    def start_thread(fn, phase, meta=None):
        q = Queue()
        st.session_state.queue        = q
        st.session_state.running      = True
        st.session_state.phase        = phase
        st.session_state.activity_log = []
        st.session_state.result       = ""
        st.session_state.ingest_meta  = meta or {}
        Thread(target=_agent_worker, args=(fn, q), daemon=True).start()
        st.rerun()

    # ── sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🔑 Configuration")

        # Use key= so Streamlit persists the typed value across reruns.
        # value= is intentionally omitted — it would reset the field every rerun.
        # Pre-seed from env var is done above (before widget renders).
        api_key_in = st.text_input(
            "Google API Key", type="password",
            placeholder="AIza…",
            key="api_key_widget",
            help="Paste your key then press Enter (or Tab away) to confirm.",
        )
        # Also read directly from session_state in case the widget didn't rerun
        _raw_key = api_key_in or st.session_state.get("api_key_widget", "")
        if _raw_key:
            st.session_state.google_api_key = _raw_key
            os.environ["GOOGLE_API_KEY"] = _raw_key

        model_sel = st.selectbox(
            "Model", ["gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-pro"],
            key="model_sel_widget",
        )
        st.session_state.model_name = model_sel

        agent_ready = st.session_state.agent is not None

        if st.session_state.google_api_key and not agent_ready:
            if st.button("🚀 Initialize Agent", use_container_width=True):
                with st.spinner("Building agent + ChromaDB…"):
                    try:
                        os.environ["GOOGLE_API_KEY"] = st.session_state.google_api_key
                        ag, _, _ = build_agent(
                            model_name=st.session_state.model_name,
                            api_key=st.session_state.google_api_key,
                        )
                        st.session_state.agent    = ag
                        st.session_state["_cc_vs"] = _VS   # persist so reruns restore it
                        add_event("Agent + ChromaDB initialized ✓", "store")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Init error: {e}")
        elif agent_ready:
            st.success("✓ Agent ready — upload a document to begin")
            with st.expander("Advanced"):
                if st.button("🔄 Clear session & reset", use_container_width=True):
                    # Preserve API key and model so user doesn't have to re-enter
                    saved_key   = st.session_state.google_api_key
                    saved_model = st.session_state.model_name
                    for k, v in _UI_DEFAULTS.items():
                        st.session_state[k] = v
                    st.session_state.google_api_key      = saved_key
                    st.session_state.model_name          = saved_model
                    st.session_state["_cc_vs"]           = None
                    st.session_state["_cc_docs_loaded"]  = False
                    st.rerun()
        else:
            st.info("Enter your Google API key above, then click **Initialize Agent**.")

        st.markdown("---")
        st.markdown("### ⚙️ Pipeline Config")
        st.markdown(f"""
<div style="font-size:.77rem;color:#8b949e;line-height:2.1">
<b style="color:#c9d1d9">Context window</b>: {CFG.model_context_window:,} tokens<br>
<b style="color:#c9d1d9">Compress at</b>: {CFG.compression_threshold:.0%} fill<br>
<b style="color:#c9d1d9">L1 offload at</b>: &gt;{CFG.tool_result_offload_threshold:,} tokens<br>
<b style="color:#c9d1d9">Chunk size</b>: {CFG.chunk_size} chars / {CFG.chunk_overlap} overlap<br>
<b style="color:#c9d1d9">Keep recent</b>: {CFG.recent_messages_keep:.0%} of window
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
### 🗜️ Compression Layers
<div style="font-size:.77rem;color:#8b949e;line-height:2.3">
<span style="color:#d29922">●</span> <b>L1</b> — Tool result offload &gt;2k tokens<br>
<span style="color:#bc8cff">●</span> <b>L2</b> — Tool input eviction @ 85%<br>
<span style="color:#f85149">●</span> <b>L3</b> — LLM summarization (fallback)<br>
<span style="color:#39d0d8">●</span> <b>Auto</b> — Agent-triggered compact
</div>""", unsafe_allow_html=True)

    # ── header ────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="cc-header">
  <h1>🧠 Context Compression Pipeline
    <span style="font-size:.84rem;color:#8b949e;font-weight:400">&nbsp;v2</span>
  </h1>
  <p>Google Gemini 2.5 Pro &nbsp;·&nbsp; ChromaDB vector store &nbsp;·&nbsp;
     3-Layer automatic compression &nbsp;·&nbsp; Autonomous compact tool</p>
</div>""", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    # ── LEFT — document + query ───────────────────────────────────────────────
    with col_left:
        st.markdown('<div class="step-lbl">📄 Step 1 — Load Document</div>',
                    unsafe_allow_html=True)

        if st.session_state.doc_loaded:
            st.success(
                f"✓ **{st.session_state.doc_name}** "
                f"({st.session_state.chunks_stored} chunks in ChromaDB)"
            )
            if st.button("🔄 Load different document"):
                st.session_state.doc_loaded     = False
                st.session_state.doc_name       = ""
                st.session_state.history        = []
                st.session_state.chunks_stored  = 0
                st.rerun()
        else:
            tab_up, tab_fp, tab_txt = st.tabs(
                ["📁 Upload File", "🗂️ File Path", "📋 Paste Text"]
            )

            with tab_up:
                uploaded = st.file_uploader(
                    "Upload a UTF-8 text file",
                    type=["txt", "md", "py", "json", "csv", "log"],
                    label_visibility="collapsed",
                )
                if uploaded:
                    if not agent_ready:
                        st.warning("Initialize the agent first (sidebar).")
                    elif not st.session_state.running and not st.session_state.doc_loaded:
                        # Auto-ingest as soon as the file is dropped
                        suffix = Path(uploaded.name).suffix or ".txt"
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=suffix, mode="wb"
                        ) as f:
                            f.write(uploaded.read())
                            tmp = f.name
                        _ag, _h, _m = (
                            st.session_state.agent,
                            list(st.session_state.history),
                            f"Ingest the file at: {tmp}",
                        )
                        start_thread(
                            fn=lambda: chat(_ag, _h, _m),
                            phase="ingesting",
                            meta={"name": uploaded.name, "tmp": tmp},
                        )
                    elif st.session_state.running:
                        st.info(f"⏳ Ingesting **{uploaded.name}**…")

            with tab_fp:
                fp_in = st.text_input(
                    "File path", placeholder="C:/path/to/document.txt",
                    label_visibility="collapsed",
                )
                if fp_in:
                    if not agent_ready:
                        st.warning("Initialize the agent first (sidebar).")
                    elif st.button("📥 Ingest Path", use_container_width=True,
                                   disabled=st.session_state.running):
                        _ag, _h, _m = (
                            st.session_state.agent,
                            list(st.session_state.history),
                            f"Ingest the file at: {fp_in}",
                        )
                        start_thread(
                            fn=lambda: chat(_ag, _h, _m),
                            phase="ingesting",
                            meta={"name": Path(fp_in).name, "tmp": None},
                        )

            with tab_txt:
                pasted  = st.text_area("Paste document text", height=130,
                                       placeholder="Paste content here…",
                                       label_visibility="collapsed")
                doc_lbl = st.text_input("Document name", placeholder="my_document",
                                        label_visibility="collapsed")
                if pasted:
                    if not agent_ready:
                        st.warning("Initialize the agent first (sidebar).")
                    elif st.button("📥 Ingest Text", use_container_width=True,
                                   disabled=st.session_state.running):
                        _name = doc_lbl.strip() or "pasted_text"
                        _ag, _h = st.session_state.agent, list(st.session_state.history)
                        _m = (f'Ingest the following document '
                              f'with source_name="{_name}":\n\n{pasted}')
                        start_thread(
                            fn=lambda: chat(_ag, _h, _m),
                            phase="ingesting",
                            meta={"name": _name, "tmp": None},
                        )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="step-lbl">🔍 Step 2 — Ask a Query</div>',
                    unsafe_allow_html=True)
        query = st.text_area(
            "Query",
            placeholder="e.g. What are the main adaptation strategies mentioned?",
            height=90, label_visibility="collapsed",
            disabled=not st.session_state.doc_loaded,
        )
        if st.button(
            "🔍 Run Query", use_container_width=True,
            disabled=(not st.session_state.doc_loaded
                      or st.session_state.running
                      or not query.strip()),
        ):
            _ag, _h, _q = (
                st.session_state.agent,
                list(st.session_state.history),
                query.strip(),
            )
            start_thread(fn=lambda: chat(_ag, _h, _q), phase="querying")

        if st.session_state.history:
            turns = sum(1 for m in st.session_state.history
                        if m.__class__.__name__ == "HumanMessage")
            st.caption(f"💬 {turns} turn(s) · {st.session_state.ctx_tokens:,} tokens in context")

    # ── RIGHT — activity feed + results ──────────────────────────────────────
    with col_right:
        fill = st.session_state.ctx_fill
        bar_c = "#3fb950" if fill < .5 else "#d29922" if fill < .85 else "#f85149"
        st.markdown(f"""
<div class="ctx-wrap">
  <div style="display:flex;justify-content:space-between;font-size:.73rem;color:#8b949e">
    <span>Context Window</span>
    <span style="font-family:var(--mono);color:#c9d1d9">
      {st.session_state.ctx_tokens:,} tokens &nbsp;·&nbsp; {fill:.1%} full
    </span>
  </div>
  <div class="ctx-track">
    <div class="ctx-fill-bar" style="width:{min(fill*100,100):.1f}%;background:{bar_c}"></div>
  </div>
  <div style="font-size:.66rem;color:#8b949e">
    Compression triggers at 85% &nbsp;·&nbsp; L1 always-on for large tool results
  </div>
</div>""", unsafe_allow_html=True)

        s = st.session_state.ui_stats
        chips = [
            (s["l1"],           "#d29922", "Layer 1"),
            (s["l2"],           "#bc8cff", "Layer 2"),
            (s["l3"],           "#f85149", "Layer 3"),
            (s["auto"],         "#39d0d8", "Auto"),
            (f'{s["saved"]:,}', "#3fb950", "Tokens ↓"),
        ]
        for col, (val, color, lbl) in zip(st.columns(5), chips):
            with col:
                st.markdown(
                    f'<div class="stat-chip">'
                    f'<div class="val" style="color:{color}">{val}</div>'
                    f'<div class="lbl">{lbl}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="step-lbl">⚡ Compression Activity</div>',
                    unsafe_allow_html=True)
        act_ph = st.empty()
        log = st.session_state.activity_log
        if not log:
            act_ph.markdown(
                '<div style="color:#484f58;font-size:.81rem;padding:6px 0">'
                'Compression events will appear here during processing…</div>',
                unsafe_allow_html=True,
            )
        else:
            html = "".join(
                f'<div class="evt {e["kind"]}">'
                f'<span>{_EVT_ICON.get(e["kind"], "›")}</span>'
                f'<div><div class="evt-msg">{e["msg"][:130]}</div>'
                f'<div class="evt-time">{e["time"]}</div></div></div>'
                for e in log[-35:]
            )
            act_ph.markdown(html, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="step-lbl">📋 Result</div>', unsafe_allow_html=True)
        res_ph = st.empty()
        if st.session_state.result:
            res_ph.markdown(
                f'<div class="result-box">{st.session_state.result}</div>',
                unsafe_allow_html=True,
            )
        else:
            res_ph.markdown(
                '<div style="color:#484f58;font-size:.84rem;padding:6px 0">'
                'Query results will appear here after processing.</div>',
                unsafe_allow_html=True,
            )

    # ── queue polling ─────────────────────────────────────────────────────────
    if st.session_state.running and st.session_state.queue is not None:
        q = st.session_state.queue

        while True:
            try:
                msg_type, payload = q.get_nowait()

                if msg_type == "log":
                    kind = _classify_log(payload)
                    add_event(payload, kind)
                    lo = payload.lower()
                    if   "[layer 1]"     in lo: st.session_state.ui_stats["l1"]   += 1
                    elif "[layer 2]"     in lo: st.session_state.ui_stats["l2"]   += 1
                    elif "[layer 3]"     in lo: st.session_state.ui_stats["l3"]   += 1
                    elif "[autonomous]"  in lo: st.session_state.ui_stats["auto"] += 1
                    m = re.search(r"saved\s*~?([\d,]+)\s*tokens", payload, re.I)
                    if m:
                        st.session_state.ui_stats["saved"] += int(m.group(1).replace(",", ""))
                    m = re.search(r"Context:\s*([\d,]+)\s*tokens\s*\(([\d.]+)%", payload)
                    if m:
                        st.session_state.ctx_tokens = int(m.group(1).replace(",", ""))
                        st.session_state.ctx_fill   = float(m.group(2)) / 100
                    m = re.search(r"fill[=\s]*([\d.]+)%", payload, re.I)
                    if m:
                        st.session_state.ctx_fill = float(m.group(1)) / 100
                    m = re.search(r"Ingested\s+(\d+)\s+chunks", payload, re.I)
                    if m:
                        st.session_state.chunks_stored += int(m.group(1))

                elif msg_type == "done":
                    resp_text, new_history = payload      # chat() → (str, list)
                    st.session_state.history  = new_history
                    st.session_state.running  = False
                    st.session_state.queue    = None
                    if st.session_state.phase == "ingesting":
                        meta = st.session_state.ingest_meta or {}
                        doc_name = meta.get("name", "document")
                        st.session_state.doc_loaded            = True
                        st.session_state.doc_name              = doc_name
                        st.session_state.result                = resp_text
                        st.session_state["_cc_docs_loaded"]    = True
                        st.session_state["_cc_vs"]             = _VS  # re-save in case VS updated
                        add_event(f"✅ '{doc_name}' ingested & indexed", "store")
                        tmp = meta.get("tmp")
                        if tmp:
                            try: Path(tmp).unlink()
                            except OSError: pass
                    else:
                        st.session_state.result = resp_text
                        add_event("✅ Query complete", "store")
                    st.session_state.phase = "idle"
                    st.rerun()
                    break

                elif msg_type == "error":
                    st.session_state.running = False
                    st.session_state.queue   = None
                    st.session_state.phase   = "idle"
                    add_event(f"❌ {payload}", "l3")
                    st.error(f"Error: {payload}")
                    break

            except Empty:
                break

        if st.session_state.running:
            time.sleep(0.4)
            st.rerun()


# =============================================================================
# 13.  ENTRY POINT — CLI or Streamlit, detected automatically
# =============================================================================

if _in_streamlit():
    streamlit_app()
elif __name__ == "__main__":
    main()
